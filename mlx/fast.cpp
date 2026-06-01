// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string_view>

#include "mlx/fast.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/kernels/turbo_quant_attention_jit.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (j < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (weight.has_value()) ? result_type(x, (*weight)) : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback =
      [has_weight, eps, out_type, s](const std::vector<array>& inputs) {
        auto x = astype(inputs[0], float32, s);
        x = multiply(
            x,
            rsqrt(
                add(mean(square(x, s), -1, /* keepdims */ true, s),
                    array(eps, float32),
                    s),
                s),
            s);
        x = astype(x, out_type, s);

        if (has_weight) {
          x = multiply(x, inputs[1], s);
        }

        return std::vector<array>{x};
      };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);

  if (!RMSNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight});
  }
  return fallback({x, passed_weight})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    // df/dx
    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();
  bool has_bias = bias.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[layer_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (has_bias) {
    if ((*bias).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have 1 dimension but has "
          << (*bias).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*bias).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have the same size as the last dimension of"
             " x but has "
          << (*bias).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (has_weight)
      ? ((has_bias) ? result_type(x, *weight, *bias) : result_type(x, *weight))
      : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[layer_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [has_weight, has_bias, eps, out_type, s](
                      const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);

    auto mu = mean(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto xc = subtract(x, mu, s);
    auto v = mean(square(xc, s), /* axis= */ -1, /* keepdims= */ true, s);

    x = multiply(xc, rsqrt(add(v, array(eps, float32), s), s));
    x = astype(x, out_type, s);

    // If the LN is affine then transform x according to the weight and bias
    if (has_weight) {
      x = multiply(x, inputs[1], s);
    }
    if (has_bias) {
      x = add(x, inputs[2], s);
    }

    return std::vector<array>{x};
  };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);
  auto passed_bias =
      (has_bias) ? astype(*bias, out_type, s) : array(0, out_type);

  if (!LayerNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<LayerNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight, passed_bias});
  }
  return fallback({x, passed_weight, passed_bias})[0];
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& b = inputs[2];
    auto& g = inputs[3];

    std::vector<array> vjps;

    auto norm = number_of_elements(x, {-1}, true, x.dtype(), s);
    auto sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto mu = multiply(sumx, norm, s);
    auto mu2 = multiply(sumx2, norm, s);
    auto var = subtract(mu2, square(mu, s), s);
    auto n = rsqrt(add(var, array(eps, x.dtype()), s));
    auto n3 = power(n, array(3, x.dtype()), s);
    auto x_c = subtract(x, mu, s);

    // df/dx
    auto wg = multiply(w, g, s);
    auto sumwg =
        multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
    auto sumwgxc = multiply(
        sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
        norm,
        s);
    auto t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
    auto t2 = multiply(subtract(wg, sumwg, s), n, s);
    vjps.push_back(subtract(t2, t1, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x_c, n, s), s), axes, /* keepdims= */ false, s));
    }

    // df/db
    if (b.ndim() == 0) {
      vjps.push_back(zeros_like(b, s));
    } else {
      vjps.push_back(sum(g, axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape(), primals[2].shape()},
      {primals[0].dtype(), primals[1].dtype(), primals[2].dtype()},
      std::make_shared<LayerNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& a_other = static_cast<const LayerNorm&>(other);
  return eps_ == a_other.eps_;
}

bool LayerNormVJP::is_equivalent(const Primitive& other) const {
  const LayerNormVJP& a_other = static_cast<const LayerNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  auto& offset = inputs[1];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[rope] Input must be a floating type but got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.ndim() > 1) {
    std::ostringstream msg;
    msg << "[rope] offset must have at most one dimension but has shape "
        << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.size() != 1 && offset.size() != x.shape(0)) {
    std::ostringstream msg;
    msg << "[rope] offset must be a scalar or vector with " << x.shape(0)
        << " elements but has shape " << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(offset.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[rope] offset must be an integer but got type " << offset.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.dtype().size() != 4) {
    inputs[1] = astype(offset, int32, s);
  }
  if (dims <= 0) {
    std::ostringstream msg;
    msg << "[rope] dims must be positive but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }
  if (dims % 2 != 0) {
    std::ostringstream msg;
    msg << "[rope] dims must be even but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }
  if (dims > x.shape(-1)) {
    std::ostringstream msg;
    msg << "[rope] dims must not exceed the input's last dimension ("
        << x.shape(-1) << ") but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }

  if (inputs.size() == 3 &&
      (inputs[2].ndim() != 1 || inputs[2].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[2].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, forward, s](
                      std::vector<array> inputs) {
    auto x = inputs[0];
    auto shape = x.shape();
    if (x.ndim() == 3) {
      x = expand_dims(x, 1, s);
    } else if (x.ndim() > 4) {
      x = flatten(x, 1, 1 + (x.ndim() - 4), s);
    }

    auto B = x.shape(0);
    auto N = x.shape(1);
    auto T = x.shape(2);
    auto t = x.dtype();
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto offset = inputs[1];
    if (offset.size() > 1) {
      offset = expand_dims(offset, {-1, -2}, s);
    }
    auto positions = multiply(
        add(arange(x.shape(2), float32, s), offset, s),
        array(scale, float32),
        s);

    auto default_inv_freqs = [&s, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, float32, s),
              array(std::log(base) / half_dims, float32),
              s),
          s);
    };

    auto inv_freqs =
        inputs.size() == 3 ? reciprocal(inputs[2], s) : default_inv_freqs();
    auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
    auto coss = astype(cos(theta, s), t, s);
    auto sins = astype(sin(theta, s), t, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, -1, s);
      }
      auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
      if (dims < x.shape(-1)) {
        out =
            concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, -1, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (!RoPE::use_fallback(stream)) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x, offset};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument(
          "[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      true,
      s);
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return rope(
      x, dims, traditional, base, scale, array(offset, int32), freqs, s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{
        rope(std::move(inputs), dims, traditional, base, scale, !forward, s)};
  };
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[RoPE::vjp] vjp for offset or frequencies not supported");
  }
  auto inputs = std::vector<array>{cotangents[0], primals[1]};
  if (primals.size() == 3) {
    inputs.push_back(primals[2]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      forward_ == a_other.forward_);
}

std::pair<array, bool> prepare_sdpa_array_mask(
    const array& mask,
    Dtype out_type,
    const Shape& full_mask_shape,
    std::string_view tag,
    Stream s) {
  bool has_bool_mask = mask.dtype() == bool_;
  if (!has_bool_mask && promote_types(mask.dtype(), out_type) != out_type) {
    std::ostringstream msg;
    msg << "[" << tag << "] Mask type must promote to output type " << out_type
        << ".";
    throw std::invalid_argument(msg.str());
  }
  auto prepared_mask = has_bool_mask ? mask : astype(mask, out_type, s);
  return {broadcast_to(prepared_mask, full_mask_shape, s), has_bool_mask};
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode /* = "" */,
    std::optional<array> mask_arr /* = {} */,
    const std::optional<array>& sinks /* = {} */,
    StreamOrDevice s /* = {}*/) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }
  // Check valid mask
  if (mask_mode != "" && mask_mode != "causal" && mask_mode != "array") {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Invalid mask_mode " << mask_mode
        << ". mask_mode must be 'causal', 'array' or ''.";
    throw std::invalid_argument(msg.str());
  }

  bool do_causal = false;
  bool has_mask = false;
  bool has_arr_mask = false;
  bool has_bool_mask = false;

  if (mask_mode == "causal") {
    has_mask = true;
    do_causal = true;

    if (mask_arr) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Invalid mask_arr for mask_mode "
          << "'casusal'. No array mask should be passed.";
      throw std::invalid_argument(msg.str());
    }
  } else if (mask_arr) {
    has_mask = true;
    has_arr_mask = true;
  }

  if (has_arr_mask && mask_arr->ndim() > 4) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] the mask with shape "
        << mask_arr->shape() << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Q, K must have matching last dims (d_k aka 'head_dim');
  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // K, V must have matching number of heads (n_kv_heads);
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  bool has_sinks = sinks.has_value();

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_sinks,
                   has_arr_mask,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (has_arr_mask || do_causal) {
      // Mask must be broadcast-compatible with [B, n_q_heads, L_q, L_kv]
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int offset = kL - qL;
          auto q_idx = arange(offset, qL + offset, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          return greater_equal(q_idx, k_idx, s);
        }
        return inputs[3];
      };
      auto mask = make_or_fetch_mask();

      if (n_repeats > 1 && mask.ndim() >= 3) {
        if (mask.shape(-3) == 1) {
          mask = expand_dims(mask, -3, s);
        } else {
          mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (mask.dtype() == bool_) {
        scores = where(
            mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, mask, s);
      }
    }
    if (has_sinks) {
      auto sinks = inputs.back();
      // scores has shape B N_q N_k L_q L_k
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      // Slice off scores
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  std::vector<array> inputs = {q, k, v};
  if (has_arr_mask) {
    auto mask_shape = queries.shape();
    mask_shape.back() = keys.shape(-2);
    auto [prepared_mask, prepared_bool_mask] = prepare_sdpa_array_mask(
        *mask_arr,
        final_type,
        mask_shape,
        "scaled_dot_product_attention",
        stream);
    has_bool_mask = prepared_bool_mask;
    inputs.push_back(std::move(prepared_mask));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  bool is_training = detail::in_grad_tracing();
  bool has_fast_vjp = !ScaledDotProductAttentionVJP::use_fallback(q, stream);
  bool output_logsumexp = is_training && has_fast_vjp;
  if (!ScaledDotProductAttention::use_fallback(
          q,
          k,
          v,
          has_mask,
          has_arr_mask,
          do_causal,
          is_training,
          output_logsumexp,
          stream)) {
    if (has_bool_mask && !ScaledDotProductAttention::supports_bool_mask()) {
      // Convert bool mask to additive mask.
      float inf = std::numeric_limits<float>::infinity();
      array& mask = inputs[3];
      mask = where(
          mask,
          full_like(mask, 0, final_type, s),
          full_like(mask, -inf, final_type, s));
    }
    Shape out_shape{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
    auto primitive = std::make_shared<ScaledDotProductAttention>(
        stream, fallback, scale, do_causal, has_sinks, output_logsumexp);
    if (output_logsumexp) {
      return array::make_arrays(
          {std::move(out_shape), Shape{q.shape(0), q.shape(1), q.shape(2), 1}},
          {final_type, float32},
          primitive,
          std::move(inputs))[0];
    } else {
      return array(
          std::move(out_shape), final_type, primitive, std::move(inputs));
    }
  }
  return fallback(std::move(inputs))[0];
}

array quantized_scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& key_scales,
    const std::optional<array>& key_biases,
    const array& values,
    const array& value_scales,
    const std::optional<array>& value_biases,
    const float scale,
    const std::optional<array>& mask /* = std::nullopt */,
    const std::optional<array>& sinks /* = std::nullopt */,
    const std::optional<int> group_size_ /* = std::nullopt */,
    const std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "mxfp4" */,
    bool causal /* = false */,
    StreamOrDevice s /* = {} */) {
  constexpr const char* tag = "quantized_scaled_dot_product_attention";

  // Parse mode and get parameters
  auto qmode = string_to_quantization_mode(mode, tag);
  bool is_affine = qmode == QuantizationMode::Affine;
  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);

  // Validate mode-specific group_size and bits
  if (is_affine) {
    if (group_size != 32 && group_size != 64) {
      std::ostringstream msg;
      msg << "[" << tag << "] Affine mode supports group_size 32 or 64 "
          << "but received " << group_size << ".";
      throw std::invalid_argument(msg.str());
    }
    if (bits != 4 && bits != 6 && bits != 8) {
      std::ostringstream msg;
      msg << "[" << tag
          << "] Affine mode supports bits 4, 6, or 8 but received " << bits
          << ".";
      throw std::invalid_argument(msg.str());
    }
    if (!key_biases.has_value() || !value_biases.has_value()) {
      throw std::invalid_argument(
          "[quantized_scaled_dot_product_attention] Affine mode requires "
          "key_biases and value_biases.");
    }
  } else {
    // FP modes have fixed params - verify if user overrode them incorrectly
    auto [expected_gs, expected_bits] =
        quantization_params_from_mode(qmode, std::nullopt, std::nullopt);
    if (group_size != expected_gs || bits != expected_bits) {
      std::ostringstream msg;
      msg << "[" << tag << "] Mode '" << mode << "' requires group_size "
          << expected_gs << " and bits " << expected_bits << ".";
      throw std::invalid_argument(msg.str());
    }
    if (key_biases.has_value() || value_biases.has_value()) {
      throw std::invalid_argument(
          "[quantized_scaled_dot_product_attention] Biases should only be "
          "provided for affine mode.");
    }
  }

  // Validate rank 4 for all inputs
  for (const auto& t : {queries, keys, key_scales, values, value_scales}) {
    if (t.ndim() != 4) {
      std::ostringstream msg;
      msg << "[" << tag << "] input with shape " << t.shape()
          << " expected to be rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (is_affine && (key_biases->ndim() != 4 || value_biases->ndim() != 4)) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Biases must be rank 4.");
  }

  // Validate dtypes
  auto final_type = queries.dtype();
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[" << tag << "] queries must be floating type but got "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!(final_type == float16 || final_type == bfloat16 ||
        final_type == float32)) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] queries must be float16, bfloat16, or float32 for quantized "
           "attention; received "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.dtype() != uint32 || values.dtype() != uint32) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Keys and values must be "
        "uint32.");
  }
  if (!is_affine &&
      (key_scales.dtype() != uint8 || value_scales.dtype() != uint8)) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Scales must be uint8 for fp "
        "quantization.");
  }

  // Compute and validate dimensions
  auto key_head_dim = (keys.shape(-1) * 32) / bits;
  auto value_head_dim = (values.shape(-1) * 32) / bits;
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (queries.shape(0) != keys.shape(0) ||
      queries.shape(0) != values.shape(0)) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Batch dimensions must match.");
  }
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] n_heads must be a multiple of n_kv_heads, found "
        << n_q_heads << " vs " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.shape(-3) != values.shape(-3)) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Keys and values must have "
        "matching n_kv_heads.");
  }
  if (queries.shape(-1) != key_head_dim ||
      queries.shape(-1) != value_head_dim) {
    std::ostringstream msg;
    msg << "[" << tag << "] Query head dim " << queries.shape(-1)
        << " must match key (" << key_head_dim << ") and value ("
        << value_head_dim << ").";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(-1) % group_size != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] Head dim " << queries.shape(-1)
        << " must be divisible by group_size " << group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  // Validate scale/bias shapes
  auto expected_scale_dim = queries.shape(-1) / group_size;
  for (const auto& [qdata, scale, bias, name] :
       {std::tuple{&keys, &key_scales, &key_biases, "key"},
        std::tuple{&values, &value_scales, &value_biases, "value"}}) {
    if (scale->shape(-1) != expected_scale_dim ||
        scale->shape(-3) != qdata->shape(-3) ||
        scale->shape(-2) != qdata->shape(-2)) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name << " scale shape mismatch.";
      throw std::invalid_argument(msg.str());
    }
    if (is_affine && bias->has_value() && (*bias)->shape() != scale->shape()) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " bias shape must match scale shape.";
      throw std::invalid_argument(msg.str());
    }
  }

  // Validate mask
  bool do_causal = causal;
  bool has_arr_mask = mask.has_value();
  bool has_sinks = sinks.has_value();
  if (do_causal && has_arr_mask) {
    throw std::invalid_argument(
        "[quantized_scaled_dot_product_attention] Received both causal=true "
        "and an array mask. Please provide only one mask type.");
  }
  if (has_arr_mask && mask->ndim() > 4) {
    std::ostringstream msg;
    msg << "[" << tag << "] Mask with shape " << mask->shape()
        << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  auto q = astype(queries, final_type, s);
  // Inputs layout:
  // [q, k, k_scales, k_biases (if affine), v, v_scales, v_biases (if affine),
  // mask (if present), sinks (if present)]
  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_arr_mask,
                   has_sinks,
                   is_affine,
                   group_size,
                   bits,
                   mode,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;

    auto k = inputs[1];
    auto k_scales = inputs[2];
    std::optional<array> k_biases = std::nullopt;
    int idx = 3;
    if (is_affine) {
      k_biases = inputs[idx++];
    }
    auto v = inputs[idx++];
    auto v_scales = inputs[idx++];
    std::optional<array> v_biases = std::nullopt;
    if (is_affine) {
      v_biases = inputs[idx++];
    }

    std::optional<array> arr_mask =
        has_arr_mask ? std::optional<array>{inputs[idx++]} : std::nullopt;
    std::optional<array> sinks_opt =
        has_sinks ? std::optional<array>{inputs[idx++]} : std::nullopt;

    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      k_scales = expand_dims(k_scales, 2, s);
      if (k_biases) {
        k_biases = expand_dims(*k_biases, 2, s);
      }
      v = expand_dims(v, 2, s);
      v_scales = expand_dims(v_scales, 2, s);
      if (v_biases) {
        v_biases = expand_dims(*v_biases, 2, s);
      }
    }

    auto scores = quantized_matmul(
        q,
        k,
        k_scales,
        k_biases,
        /*transpose=*/true,
        group_size,
        bits,
        mode,
        s);
    if (has_arr_mask || do_causal) {
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int offset = kL - qL;
          auto q_idx = arange(offset, qL + offset, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          return greater_equal(q_idx, k_idx, s);
        }
        return *arr_mask;
      };
      auto m = make_or_fetch_mask();
      if (n_repeats > 1 && m.ndim() >= 3) {
        if (m.shape(-3) == 1) {
          m = expand_dims(m, -3, s);
        } else {
          m = unflatten(m, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (m.dtype() == bool_) {
        scores = where(
            m, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, m, s);
      }
    }

    if (has_sinks) {
      auto sinks = *sinks_opt;
      // scores has shape B N_q N_k L_q L_k
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = quantized_matmul(
        scores,
        v,
        v_scales,
        v_biases,
        /*transpose=*/false,
        group_size,
        bits,
        mode,
        s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  Shape full_mask_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), keys.shape(-2)};

  std::vector<array> inputs = {q, keys, key_scales};
  if (is_affine) {
    inputs.push_back(*key_biases);
  }
  inputs.push_back(values);
  inputs.push_back(value_scales);
  if (is_affine) {
    inputs.push_back(*value_biases);
  }
  if (has_arr_mask) {
    auto prepared_mask = prepare_sdpa_array_mask(
        *mask, final_type, full_mask_shape, tag, stream);
    inputs.push_back(std::move(prepared_mask.first));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[" << tag << "] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[" << tag << "] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  int out_dim = value_head_dim;
  Shape out_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), out_dim};

  if (QuantizedScaledDotProductAttention::use_fallback(
          q, keys, detail::in_grad_tracing(), stream)) {
    return fallback(std::move(inputs))[0];
  }

  auto primitive = std::make_shared<QuantizedScaledDotProductAttention>(
      stream,
      fallback,
      scale,
      has_arr_mask,
      has_sinks,
      do_causal,
      group_size,
      bits,
      group_size,
      bits,
      qmode);
  return array(std::move(out_shape), final_type, primitive, std::move(inputs));
}

array mixed_quantized_scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& key_scales,
    const std::optional<array>& key_biases,
    const array& values,
    const array& value_scales,
    const std::optional<array>& value_biases,
    const float scale,
    const std::optional<array>& mask /* = std::nullopt */,
    const std::optional<array>& sinks /* = std::nullopt */,
    int key_group_size /* = 64 */,
    int key_bits /* = 8 */,
    int value_group_size /* = 32 */,
    int value_bits /* = 4 */,
    bool causal /* = false */,
    StreamOrDevice s /* = {} */) {
  constexpr const char* tag = "mixed_quantized_scaled_dot_product_attention";
  auto qmode = QuantizationMode::Affine;

  auto validate_affine_params = [&](std::string_view name,
                                    int group_size,
                                    int bits) {
    if (group_size != 32 && group_size != 64) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " affine group_size must be 32 or 64 but received "
          << group_size << ".";
      throw std::invalid_argument(msg.str());
    }
    if (bits != 4 && bits != 6 && bits != 8) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " affine bits must be 4, 6, or 8 but received " << bits << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  validate_affine_params("key", key_group_size, key_bits);
  validate_affine_params("value", value_group_size, value_bits);
  if (key_bits != 8 || value_bits != 4) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] native mixed affine SDPA supports only K8/V4 quantization; "
        << "received key_bits=" << key_bits
        << " and value_bits=" << value_bits << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!key_biases.has_value() || !value_biases.has_value()) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Affine mode requires "
        "key_biases and value_biases.");
  }

  for (const auto& t : {queries, keys, key_scales, values, value_scales}) {
    if (t.ndim() != 4) {
      std::ostringstream msg;
      msg << "[" << tag << "] input with shape " << t.shape()
          << " expected to be rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (key_biases->ndim() != 4 || value_biases->ndim() != 4) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Biases must be rank 4.");
  }

  auto final_type = queries.dtype();
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[" << tag << "] queries must be floating type but got "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!(final_type == float16 || final_type == bfloat16 ||
        final_type == float32)) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] queries must be float16, bfloat16, or float32; received "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.dtype() != uint32 || values.dtype() != uint32) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Keys and values must "
        "be uint32.");
  }

  auto key_head_dim = (keys.shape(-1) * 32) / key_bits;
  auto value_head_dim = (values.shape(-1) * 32) / value_bits;
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);
  auto query_sequence_length = queries.shape(-2);
  auto key_sequence_length = keys.shape(-2);
  auto head_dim = queries.shape(-1);

  if (queries.shape(0) <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 ||
      query_sequence_length <= 0 || key_sequence_length <= 0 ||
      head_dim <= 0) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] inputs must have "
        "positive batch, head, sequence, and head dimensions.");
  }

  if (queries.shape(0) != keys.shape(0) ||
      queries.shape(0) != values.shape(0)) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Batch dimensions must match.");
  }
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] n_heads must be a multiple of n_kv_heads, found "
        << n_q_heads << " vs " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }
  if (keys.shape(-3) != values.shape(-3) ||
      keys.shape(-2) != values.shape(-2)) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Keys and values must "
        "have matching n_kv_heads and sequence length.");
  }
  if (queries.shape(-1) != key_head_dim ||
      queries.shape(-1) != value_head_dim) {
    std::ostringstream msg;
    msg << "[" << tag << "] Query head dim " << queries.shape(-1)
        << " must match key (" << key_head_dim << ") and value ("
        << value_head_dim << ").";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(-1) % key_group_size != 0 ||
      queries.shape(-1) % value_group_size != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] Head dim " << queries.shape(-1)
        << " must be divisible by key_group_size " << key_group_size
        << " and value_group_size " << value_group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  auto validate_scale = [&](const array& qdata,
                            const array& scales,
                            const std::optional<array>& biases,
                            int group_size,
                            std::string_view name) {
    auto expected_scale_dim = queries.shape(-1) / group_size;
    if (scales.shape(-1) != expected_scale_dim ||
        scales.shape(0) != qdata.shape(0) ||
        scales.shape(-3) != qdata.shape(-3) ||
        scales.shape(-2) != qdata.shape(-2)) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name << " scale shape mismatch.";
      throw std::invalid_argument(msg.str());
    }
    if (biases.has_value() && biases->shape() != scales.shape()) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " bias shape must match scale shape.";
      throw std::invalid_argument(msg.str());
    }
  };
  validate_scale(keys, key_scales, key_biases, key_group_size, "key");
  validate_scale(values, value_scales, value_biases, value_group_size, "value");
  auto validate_affine_dtype = [&](const array& param, std::string_view name) {
    if (!issubdtype(param.dtype(), floating)) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " must be floating type but got " << param.dtype() << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  validate_affine_dtype(key_scales, "key scales");
  validate_affine_dtype(*key_biases, "key biases");
  validate_affine_dtype(value_scales, "value scales");
  validate_affine_dtype(*value_biases, "value biases");

  bool do_causal = causal;
  bool has_arr_mask = mask.has_value();
  bool has_sinks = sinks.has_value();
  if (do_causal && has_arr_mask) {
    throw std::invalid_argument(
        "[mixed_quantized_scaled_dot_product_attention] Received both "
        "causal=true and an array mask.");
  }
  if (has_arr_mask && mask->ndim() > 4) {
    std::ostringstream msg;
    msg << "[" << tag << "] Mask with shape " << mask->shape()
        << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto gqa_factor = n_q_heads / n_kv_heads;
  auto validate_native_support = [&]() {
    if (query_sequence_length > 32) {
      std::ostringstream msg;
      msg << "[" << tag << "] query sequence length "
          << query_sequence_length << " exceeds native K8/V4 limit 32.";
      throw std::invalid_argument(msg.str());
    }
    if (query_sequence_length > key_sequence_length) {
      std::ostringstream msg;
      msg << "[" << tag << "] query sequence length "
          << query_sequence_length
          << " must not exceed key sequence length " << key_sequence_length
          << " for native K8/V4 SDPA.";
      throw std::invalid_argument(msg.str());
    }
    if (!(head_dim == 64 || head_dim == 128 || head_dim == 256 ||
          head_dim == 512)) {
      std::ostringstream msg;
      msg << "[" << tag << "] head dimension " << head_dim
          << " is not native K8/V4 certified; expected one of "
             "{64, 128, 256, 512}.";
      throw std::invalid_argument(msg.str());
    }
    if (gqa_factor > 32) {
      std::ostringstream msg;
      msg << "[" << tag << "] GQA factor " << gqa_factor
          << " exceeds native K8/V4 limit 32.";
      throw std::invalid_argument(msg.str());
    }
    if (stream.device != Device::gpu || !metal::is_available()) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] native K8/V4 SDPA "
          "requires an available Metal GPU stream.");
    }
    if (detail::in_grad_tracing()) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] native K8/V4 SDPA "
          "does not support gradient tracing.");
    }
  };
  validate_native_support();

  auto q = astype(queries, final_type, stream);
  auto key_scales_cast = astype(key_scales, final_type, stream);
  auto key_biases_cast = astype(*key_biases, final_type, stream);
  auto value_scales_cast = astype(value_scales, final_type, stream);
  auto value_biases_cast = astype(*value_biases, final_type, stream);
  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_arr_mask,
                   has_sinks,
                   key_group_size,
                   key_bits,
                   value_group_size,
                   value_bits,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;

    auto k = inputs[1];
    auto k_scales = inputs[2];
    std::optional<array> k_biases = inputs[3];
    auto v = inputs[4];
    auto v_scales = inputs[5];
    std::optional<array> v_biases = inputs[6];
    int idx = 7;

    std::optional<array> arr_mask =
        has_arr_mask ? std::optional<array>{inputs[idx++]} : std::nullopt;
    std::optional<array> sinks_opt =
        has_sinks ? std::optional<array>{inputs[idx++]} : std::nullopt;

    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      k_scales = expand_dims(k_scales, 2, s);
      k_biases = expand_dims(*k_biases, 2, s);
      v = expand_dims(v, 2, s);
      v_scales = expand_dims(v_scales, 2, s);
      v_biases = expand_dims(*v_biases, 2, s);
    }

    auto scores = quantized_matmul(
        q,
        k,
        k_scales,
        k_biases,
        /*transpose=*/true,
        key_group_size,
        key_bits,
        "affine",
        s);
    if (has_arr_mask || do_causal) {
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int offset = kL - qL;
          auto q_idx = arange(offset, qL + offset, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          return greater_equal(q_idx, k_idx, s);
        }
        return *arr_mask;
      };
      auto m = make_or_fetch_mask();
      if (n_repeats > 1 && m.ndim() >= 3) {
        if (m.shape(-3) == 1) {
          m = expand_dims(m, -3, s);
        } else {
          m = unflatten(m, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (m.dtype() == bool_) {
        scores = where(
            m, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, m, s);
      }
    }

    if (has_sinks) {
      auto sinks = *sinks_opt;
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = quantized_matmul(
        scores,
        v,
        v_scales,
        v_biases,
        /*transpose=*/false,
        value_group_size,
        value_bits,
        "affine",
        s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  Shape full_mask_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), keys.shape(-2)};

  std::vector<array> inputs = {q,
                               keys,
                               key_scales_cast,
                               key_biases_cast,
                               values,
                               value_scales_cast,
                               value_biases_cast};
  if (has_arr_mask) {
    auto prepared_mask = prepare_sdpa_array_mask(
        *mask, final_type, full_mask_shape, tag, stream);
    inputs.push_back(std::move(prepared_mask.first));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[" << tag << "] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[" << tag << "] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  Shape out_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), value_head_dim};

  auto primitive = std::make_shared<QuantizedScaledDotProductAttention>(
      stream,
      fallback,
      scale,
      has_arr_mask,
      has_sinks,
      do_causal,
      key_group_size,
      key_bits,
      value_group_size,
      value_bits,
      qmode);
  return array(std::move(out_shape), final_type, primitive, std::move(inputs));
}

namespace {

constexpr const char* tq_sdpa_tag = "turbo_quant_segmented_attention";

bool tq_experimental_jit_available(Stream stream) {
  return stream.device == Device::gpu && metal::is_available();
}

TurboQuantSegmentedAttentionBackend tq_segmented_attention_backend(
    bool allow_experimental_jit,
    Stream stream) {
  if (TurboQuantScaledDotProductAttention::native_backend_available(stream)) {
    return TurboQuantSegmentedAttentionBackend::NativeFused;
  }
  if (allow_experimental_jit && tq_experimental_jit_available(stream)) {
    return TurboQuantSegmentedAttentionBackend::ExperimentalJit;
  }
  return TurboQuantSegmentedAttentionBackend::Unavailable;
}

void check_tq_rank(const array& x, std::string_view name, int rank) {
  if (x.ndim() != rank) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] " << name << " with shape " << x.shape()
        << " expected rank " << rank << ".";
    throw std::invalid_argument(msg.str());
  }
}

void check_tq_dtype(
    const array& x,
    std::string_view name,
    const std::vector<Dtype>& allowed) {
  if (std::find(allowed.begin(), allowed.end(), x.dtype()) == allowed.end()) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] " << name << " has unsupported dtype "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
}

void check_tq_shape(
    const array& x,
    std::string_view name,
    const Shape& expected) {
  if (x.shape() != expected) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] " << name << " shape " << x.shape()
        << " expected " << expected << ".";
    throw std::invalid_argument(msg.str());
  }
}

void check_tq_bitset_shape(
    const array& x,
    std::string_view name,
    const Shape& expected,
    bool compact_allowed) {
  if (x.shape() == expected || (compact_allowed && x.shape() == Shape{1})) {
    return;
  }
  std::ostringstream msg;
  msg << "[" << tq_sdpa_tag << "] " << name << " shape " << x.shape()
      << " expected " << expected;
  if (compact_allowed) {
    msg << " or compact [1]";
  }
  msg << ".";
  throw std::invalid_argument(msg.str());
}

int tq_next_power_of_two_clamped(int minimum, int maximum) {
  int target = std::max(1, std::min(maximum, minimum));
  int width = 1;
  while (width < target) {
    width <<= 1;
  }
  return width;
}

int tq_recommended_block_width(
    int logical_length,
    int head_dimension,
    int query_length,
    int requested_split_blocks) {
  if (query_length != 1) {
    return 0;
  }

  int minimum = std::max(head_dimension, 512);
  if (requested_split_blocks > 1) {
    minimum = std::max(
        minimum,
        (logical_length + requested_split_blocks - 1) / requested_split_blocks);
  } else if (logical_length < 4096) {
    return 0;
  } else if (logical_length > 512 * 512) {
    minimum = std::max(
        minimum,
        static_cast<int>(std::ceil(std::sqrt(static_cast<double>(logical_length)))));
  }

  int block_width = tq_next_power_of_two_clamped(minimum, 512);
  int active_blocks = (logical_length + block_width - 1) / block_width;
  if (active_blocks <= 1 || active_blocks > block_width) {
    return 0;
  }
  return block_width;
}

std::vector<std::pair<std::string, TemplateArg>> tq_seed_template(
    std::string_view prefix,
    uint64_t seed) {
  std::string name(prefix);
  return {
      {name + "_3", static_cast<int>((seed >> 48) & 0xffff)},
      {name + "_2", static_cast<int>((seed >> 32) & 0xffff)},
      {name + "_1", static_cast<int>((seed >> 16) & 0xffff)},
      {name + "_0", static_cast<int>(seed & 0xffff)}};
}

void tq_append_seed_template(
    std::vector<std::pair<std::string, TemplateArg>>& args,
    std::string_view prefix,
    uint64_t seed) {
  auto seed_args = tq_seed_template(prefix, seed);
  args.insert(args.end(), seed_args.begin(), seed_args.end());
}

std::vector<std::pair<std::string, TemplateArg>>
tq_runtime_layout_attention_template(
    const array& queries,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    Dtype output_dtype) {
  std::vector<std::pair<std::string, TemplateArg>> args = {
      {"BATCH_SIZE", layout.batch_size},
      {"KV_HEADS", layout.kv_head_count},
      {"QUERY_HEADS", queries.shape(1)},
      {"CAPACITY", layout.capacity},
      {"QUERY_LENGTH", queries.shape(2)},
      {"HEAD_DIM", layout.head_dimension},
      {"GROUP_SIZE", precision.group_size},
      {"GROUPS_PER_VECTOR", layout.groups_per_vector},
      {"BASE_BITS", precision.key_base_bits + 1},
      {"HIGH_BITS", precision.key_high_bits + 1},
      {"HIGH_NUMERATOR", precision.high_precision_numerator},
      {"HIGH_DENOMINATOR", precision.high_precision_denominator},
      {"KEY_BASE_BITS", precision.key_base_bits},
      {"KEY_HIGH_BITS", precision.key_high_bits},
      {"MAG_WORDS_PER_GROUP", layout.magnitude_words_per_group},
      {"BITSET_WORDS_PER_GROUP", layout.bitset_words_per_group},
      {"VALUE_BITS", precision.value_bits},
      {"SCALES_PER_GROUP", precision.key_scales_per_group},
      {"LAYOUT_VERSION", layout.layout_version},
      {"DETERMINISTIC_HIGH_MASK", false},
      {"ROLE", 0},
      {"OUTPUT_DTYPE", output_dtype},
      {"DO_CAUSAL", options.causal}};
  tq_append_seed_template(args, "SEED", precision.key_seed);
  return args;
}

std::vector<std::pair<std::string, TemplateArg>> tq_attention_value_template(
    const array& queries,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    Dtype output_dtype) {
  auto args = tq_runtime_layout_attention_template(
      queries, layout, precision, options, output_dtype);
  args.push_back(
      {"VALUE_MAG_WORDS_PER_GROUP", precision.value_magnitude_words_per_group});
  args.push_back({"VALUE_SCALES_PER_GROUP", precision.value_scales_per_group});
  tq_append_seed_template(args, "VALUE_SEED", precision.value_seed);
  return args;
}

const CustomKernelFunction& tq_fused_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_fused_decode_runtime_layout_native",
      {"q",
       "k_packed",
       "k_signs",
       "k_high_mask",
       "k_residual_signs",
       "k_scales",
       "v_packed",
       "v_signs",
       "v_high_mask",
       "v_residual_signs",
       "v_scales",
       "runtime_logical_length",
       "runtime_ring_offset",
       "runtime_pinned_prefix_length",
       "runtime_attention_scale"},
      {"out"},
      std::string(turbo_quant_detail::turbo_quant_fused_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_fused_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_fused_decode_runtime_layout_native",
      {"q",
       "k_packed",
       "k_signs",
       "k_high_mask",
       "k_residual_signs",
       "k_scales",
       "v_packed",
       "v_signs",
       "v_high_mask",
       "v_residual_signs",
       "v_scales",
       "runtime_logical_length",
       "runtime_ring_offset",
       "runtime_pinned_prefix_length",
       "runtime_attention_scale",
       "runtime_sparse_v_threshold"},
      {"out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_fused_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_block_partials_kernel(bool grouped_query) {
  static CustomKernelFunction generic_kernel = metal_kernel(
      "turboquant_attention_fused_block_partials_runtime_layout_native",
      {"q",
       "k_packed",
       "k_signs",
       "k_high_mask",
       "k_residual_signs",
       "k_scales",
       "v_packed",
       "v_signs",
       "v_high_mask",
       "v_residual_signs",
       "v_scales",
       "runtime_logical_length",
       "runtime_ring_offset",
       "runtime_pinned_prefix_length",
       "runtime_attention_scale"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_block_partials_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  static CustomKernelFunction gqa_kernel = metal_kernel(
      "turboquant_attention_fused_gqa_block_partials_runtime_layout_native",
      {"q",
       "k_packed",
       "k_signs",
       "k_high_mask",
       "k_residual_signs",
       "k_scales",
       "v_packed",
       "v_signs",
       "v_high_mask",
       "v_residual_signs",
       "v_scales",
       "runtime_logical_length",
       "runtime_ring_offset",
       "runtime_pinned_prefix_length",
       "runtime_attention_scale"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_gqa_block_partials_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return grouped_query ? gqa_kernel : generic_kernel;
}

const CustomKernelFunction& tq_block_reduce_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_fused_block_reduce_native",
      {"partial_stats", "partial_out"},
      {"out"},
      std::string(turbo_quant_detail::turbo_quant_block_reduce_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_diagnostics_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_diagnostics_native",
      {"sparse_stats"},
      {"diagnostics"},
      std::string(turbo_quant_detail::turbo_quant_sparse_diagnostics_source),
      "",
      false);
  return kernel;
}

std::vector<array> tq_native_inputs(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantAttentionOptions& options) {
  return {
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      array(layout.logical_length, int32),
      array(layout.ring_offset, int32),
      array(layout.pinned_prefix_length, int32),
      array(options.scale, float32)};
}

std::vector<array> tq_native_sparse_inputs(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantAttentionOptions& options) {
  auto inputs = tq_native_inputs(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      options);
  inputs.push_back(array(options.sparse_v_threshold, float32));
  return inputs;
}

array tq_diagnostics_array(
    int backend_version,
    int kernel_kind,
    int active_blocks,
    int block_tokens,
    int sparse_skipped_tokens,
    int sparse_total_tokens,
    int fallback_code,
    int flags) {
  return array(
      {backend_version,
       kernel_kind,
       active_blocks,
       block_tokens,
       sparse_skipped_tokens,
       sparse_total_tokens,
       fallback_code,
       flags},
      int32);
}

bool tq_cooperative_gqa_path_allowed(
    const array& queries,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision) {
  int repeats = queries.shape(1) / layout.kv_head_count;
  if (repeats != 4 || layout.head_dimension != 256 || precision.group_size <= 0 ||
      layout.logical_length < 32768 || layout.layout_version < 6) {
    return false;
  }
  bool uniform = precision.key_base_bits == precision.key_high_bits;
  bool split = precision.key_high_bits == precision.key_base_bits + 1;
  int chunk = layout.head_dimension / 4;
  return (uniform || split) && chunk <= precision.group_size &&
      precision.group_size % chunk == 0;
}

std::vector<array> tq_dispatch_native_jit(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    bool output_diagnostics,
    StreamOrDevice stream) {
  Dtype output_dtype = queries.dtype();
  Shape output_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), layout.head_dimension};
  int row_count = queries.shape(0) * queries.shape(1) * queries.shape(2);
  int repeats = queries.shape(1) / layout.kv_head_count;
  int flags = (options.causal ? 1 : 0);

  if (options.sparse_v_threshold > 0.0f) {
    int threadgroup_width =
        tq_next_power_of_two_clamped(std::max(layout.head_dimension, 256), 256);
    auto sparse_template =
        tq_attention_value_template(queries, layout, precision, options, output_dtype);
    sparse_template.push_back({"THREADS_PER_ROW", threadgroup_width});
    auto sparse_outputs = tq_sparse_fused_attention_kernel()(
        tq_native_sparse_inputs(
            queries,
            key_packed,
            key_signs,
            key_high_precision_mask,
            key_residual_signs,
            key_scales,
            value_packed,
            value_signs,
            value_high_precision_mask,
            value_residual_signs,
            value_scales,
            layout,
            options),
        {output_shape, Shape{row_count, 2}},
        {output_dtype, uint32},
        {row_count * threadgroup_width, 1, 1},
        {threadgroup_width, 1, 1},
        std::move(sparse_template),
        std::nullopt,
        false,
        stream);
    flags |= 16;
    if (output_diagnostics) {
      auto diagnostics = tq_sparse_diagnostics_kernel()(
          {sparse_outputs[1]},
          {Shape{8}},
          {int32},
          {256, 1, 1},
          {256, 1, 1},
          {{"ROW_COUNT", row_count},
           {"BACKEND_VERSION", options.backend_version},
           {"KERNEL_KIND", 5},
           {"ACTIVE_BLOCKS", 1},
           {"BLOCK_TOKENS", threadgroup_width},
           {"FALLBACK_CODE", 0},
           {"FLAGS", flags}},
          std::nullopt,
          false,
          stream)[0];
      return {sparse_outputs[0], diagnostics};
    }
    return {sparse_outputs[0]};
  }

  auto inputs = tq_native_inputs(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      options);

  int block_width = tq_recommended_block_width(
      layout.logical_length,
      layout.head_dimension,
      queries.shape(2),
      options.split_k_blocks);
  if (block_width > 0) {
    int active_blocks = (layout.logical_length + block_width - 1) / block_width;
    bool grouped_query = repeats > 1 && repeats <= 4;
    bool coop = grouped_query &&
        tq_cooperative_gqa_path_allowed(queries, layout, precision);
    int partial_rows = grouped_query
        ? queries.shape(0) * layout.kv_head_count * queries.shape(2)
        : row_count;
    auto partial_template =
        tq_attention_value_template(queries, layout, precision, options, output_dtype);
    partial_template.push_back({"THREADS_PER_BLOCK", block_width});
    partial_template.push_back({"BLOCK_TOKENS", block_width});
    partial_template.push_back({"BLOCK_COUNT", active_blocks});
    partial_template.push_back({"GQA_REPEATS", grouped_query ? repeats : 1});
    partial_template.push_back({"LANES_PER_TOKEN", coop ? 4 : 1});

    Dtype partial_dtype = output_dtype == float32 ? float32 : output_dtype;
    auto partials = tq_block_partials_kernel(grouped_query)(
        inputs,
        {Shape{row_count, active_blocks, 2},
         Shape{row_count, active_blocks, layout.head_dimension}},
        {float32, partial_dtype},
        {partial_rows * active_blocks * block_width, 1, 1},
        {block_width, 1, 1},
        std::move(partial_template),
        std::nullopt,
        false,
        stream);

    int reduce_width = tq_next_power_of_two_clamped(
        std::max(active_blocks, layout.head_dimension), 512);
    auto output = tq_block_reduce_kernel()(
        partials,
        {output_shape},
        {output_dtype},
        {row_count * reduce_width, 1, 1},
        {reduce_width, 1, 1},
        {{"ROW_COUNT", row_count},
         {"HEAD_DIM", layout.head_dimension},
         {"BLOCK_COUNT", active_blocks},
         {"THREADS_PER_BLOCK", reduce_width},
         {"OUTPUT_DTYPE", output_dtype}},
        std::nullopt,
        false,
        stream)[0];

    int kernel_kind = coop ? 4 : (grouped_query ? 3 : 2);
    flags |= 2 | (grouped_query ? 4 : 0) | (coop ? 8 : 0);
    if (output_diagnostics) {
      return {output,
              tq_diagnostics_array(
                  options.backend_version,
                  kernel_kind,
                  active_blocks,
                  block_width,
                  0,
                  0,
                  0,
                  flags)};
    }
    return {output};
  }

  int threadgroup_width =
      tq_next_power_of_two_clamped(std::max(layout.head_dimension, 256), 256);
  auto fused_template =
      tq_attention_value_template(queries, layout, precision, options, output_dtype);
  fused_template.push_back({"THREADS_PER_ROW", threadgroup_width});
  auto output = tq_fused_attention_kernel()(
      inputs,
      {output_shape},
      {output_dtype},
      {row_count * threadgroup_width, 1, 1},
      {threadgroup_width, 1, 1},
      std::move(fused_template),
      std::nullopt,
      false,
      stream)[0];

  if (output_diagnostics) {
    return {output,
            tq_diagnostics_array(
                options.backend_version,
                1,
                1,
                threadgroup_width,
                0,
                0,
                0,
                flags)};
  }
  return {output};
}

void validate_tq_descriptors(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options) {
  check_tq_rank(queries, "queries", 4);
  check_tq_dtype(queries, "queries", {float16, bfloat16, float32});
  if (queries.dtype() == bfloat16) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] bfloat16 native output is "
        "not enabled before the rollout gate.");
  }

  if (layout.layout_version != 4 && layout.layout_version != 5 &&
      layout.layout_version != 6) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] unsupported layout version "
        << layout.layout_version << ".";
    throw std::invalid_argument(msg.str());
  }
  if (layout.batch_size <= 0 || layout.kv_head_count <= 0 ||
      layout.capacity <= 0 || layout.logical_length < 0 ||
      layout.logical_length > layout.capacity || layout.head_dimension <= 0 ||
      layout.groups_per_vector <= 0 || layout.magnitude_words_per_group <= 0 ||
      layout.bitset_words_per_group <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] invalid non-positive or "
        "inconsistent layout descriptor.");
  }
  if (layout.ring_offset < 0 || layout.ring_offset >= layout.capacity) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] ring_offset must be within "
        "compressed capacity.");
  }
  if (layout.pinned_prefix_length < 0 ||
      layout.pinned_prefix_length > layout.logical_length ||
      layout.pinned_prefix_length > layout.capacity) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] pinned_prefix_length is "
        "outside the logical/cache range.");
  }
  if (!(layout.head_dimension == 64 || layout.head_dimension == 128 ||
        layout.head_dimension == 256)) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag
        << "] native compressed attention supports head dimension 64, 128, "
           "or 256; received "
        << layout.head_dimension << ".";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(0) != layout.batch_size ||
      queries.shape(3) != layout.head_dimension) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] query shape " << queries.shape()
        << " is incompatible with layout batch/head dimension.";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(2) <= 0 || queries.shape(2) > 8) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag
        << "] native compressed attention supports qLen 1...8; received "
        << queries.shape(2) << ".";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(1) <= 0 ||
      queries.shape(1) % layout.kv_head_count != 0) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] query heads " << queries.shape(1)
        << " must be a positive multiple of KV heads "
        << layout.kv_head_count << ".";
    throw std::invalid_argument(msg.str());
  }
  if (precision.group_size <= 0 ||
      layout.groups_per_vector !=
          (layout.head_dimension + precision.group_size - 1) /
              precision.group_size) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] group_size and "
        "groups_per_vector are inconsistent.");
  }
  if (precision.key_base_bits <= 0 || precision.key_high_bits <= 0 ||
      precision.key_high_bits < precision.key_base_bits ||
      precision.value_bits <= 0 || precision.value_bits > 8 ||
      precision.high_precision_numerator < 0 ||
      precision.high_precision_denominator <= 0 ||
      precision.key_scales_per_group <= 0 ||
      precision.value_scales_per_group <= 0 ||
      precision.value_magnitude_words_per_group <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] invalid precision policy.");
  }
  if (!std::isfinite(options.scale) || options.scale <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] scale must be finite and "
        "positive.");
  }
  if (options.split_k_blocks < 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] split_k_blocks must be "
        "non-negative.");
  }
  if (!std::isfinite(options.sparse_v_threshold) ||
      options.sparse_v_threshold < 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] sparse_v_threshold must be "
        "finite and non-negative.");
  }

  Shape key_packed_shape{
      layout.batch_size,
      layout.kv_head_count,
      layout.capacity,
      layout.groups_per_vector,
      layout.magnitude_words_per_group};
  Shape value_packed_shape{
      layout.batch_size,
      layout.kv_head_count,
      layout.capacity,
      layout.groups_per_vector,
      precision.value_magnitude_words_per_group};
  Shape bitset_shape{
      layout.batch_size,
      layout.kv_head_count,
      layout.capacity,
      layout.groups_per_vector,
      layout.bitset_words_per_group};
  Shape key_scales_shape{
      layout.batch_size,
      layout.kv_head_count,
      layout.capacity,
      layout.groups_per_vector,
      precision.key_scales_per_group};
  Shape value_scales_shape{
      layout.batch_size,
      layout.kv_head_count,
      layout.capacity,
      layout.groups_per_vector,
      precision.value_scales_per_group};

  for (const auto& x : {key_packed, value_packed}) {
    check_tq_rank(x, "packed plane", 5);
    check_tq_dtype(x, "packed plane", {uint32});
  }
  check_tq_shape(key_packed, "key_packed", key_packed_shape);
  check_tq_shape(value_packed, "value_packed", value_packed_shape);
  for (const auto& x :
       {key_signs,
        key_high_precision_mask,
        key_residual_signs,
        value_signs,
        value_high_precision_mask,
        value_residual_signs}) {
    check_tq_dtype(x, "bitset plane", {uint32});
  }
  check_tq_bitset_shape(key_signs, "key_signs", bitset_shape, false);
  check_tq_bitset_shape(
      key_high_precision_mask, "key_high_precision_mask", bitset_shape, true);
  check_tq_bitset_shape(
      key_residual_signs, "key_residual_signs", bitset_shape, true);
  check_tq_bitset_shape(value_signs, "value_signs", bitset_shape, true);
  check_tq_bitset_shape(
      value_high_precision_mask, "value_high_precision_mask", bitset_shape, true);
  check_tq_bitset_shape(
      value_residual_signs, "value_residual_signs", bitset_shape, true);

  check_tq_rank(key_scales, "key_scales", 5);
  check_tq_rank(value_scales, "value_scales", 5);
  check_tq_dtype(key_scales, "key_scales", {float16, bfloat16, float32});
  check_tq_dtype(value_scales, "value_scales", {float16, bfloat16, float32});
  check_tq_shape(key_scales, "key_scales", key_scales_shape);
  check_tq_shape(value_scales, "value_scales", value_scales_shape);
}

std::vector<array> turbo_quant_scaled_dot_product_attention_impl(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    bool output_diagnostics,
    StreamOrDevice s) {
  validate_tq_descriptors(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options);

  auto stream = to_stream(s);
  bool native_enabled =
      TurboQuantScaledDotProductAttention::native_backend_available(stream) ||
      tq_experimental_jit_available(stream);
  if (detail::in_grad_tracing()) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag
        << "] native MLX TurboQuant attention is unavailable while tracing "
           "gradients.";
    throw TurboQuantNativeAttentionUnavailable(msg.str());
  }
  if (TurboQuantScaledDotProductAttention::use_fallback(
          queries, detail::in_grad_tracing(), stream, native_enabled)) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag
        << "] native MLX TurboQuant attention is unavailable";
    if (stream.device == Device::cpu) {
      msg << " on CPU streams";
    } else {
      msg << " for this dtype or backend";
    }
    msg << ".";
    throw TurboQuantNativeAttentionUnavailable(msg.str());
  }

  return tq_dispatch_native_jit(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options,
      output_diagnostics,
      stream);
}

} // namespace

TurboQuantSegmentedAttentionBackend turbo_quant_segmented_attention_backend(
    bool allow_experimental_jit,
    StreamOrDevice s) {
  return tq_segmented_attention_backend(allow_experimental_jit, to_stream(s));
}

bool turbo_quant_segmented_attention_is_available(
    bool allow_experimental_jit,
    StreamOrDevice s) {
  return turbo_quant_segmented_attention_backend(allow_experimental_jit, s) !=
      TurboQuantSegmentedAttentionBackend::Unavailable;
}

array turbo_quant_segmented_attention(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s) {
  return turbo_quant_scaled_dot_product_attention_impl(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options,
      false,
      s)[0];
}

std::vector<array> turbo_quant_segmented_attention_with_diagnostics(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s) {
  return turbo_quant_scaled_dot_product_attention_impl(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options,
      true,
      s);
}

array turbo_quant_scaled_dot_product_attention(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s) {
  return turbo_quant_segmented_attention(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options,
      s);
}

std::vector<array> turbo_quant_scaled_dot_product_attention_with_diagnostics(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s) {
  return turbo_quant_segmented_attention_with_diagnostics(
      queries,
      key_packed,
      key_signs,
      key_high_precision_mask,
      key_residual_signs,
      key_scales,
      value_packed,
      value_signs,
      value_high_precision_mask,
      value_residual_signs,
      value_scales,
      layout,
      precision,
      options,
      s);
}

std::vector<array> ScaledDotProductAttention::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() >= 3);
  assert(cotangents.size() == outputs.size());

  auto s = stream();
  if (ScaledDotProductAttentionVJP::use_fallback(primals[0], s)) {
    assert(outputs.size() == 1);
    return Custom::vjp(primals, cotangents, argnums, outputs);
  }

  auto fallback = [sdpa = fallback_, s](const std::vector<array>& inputs) {
    std::vector<array> primals(inputs.begin(), std::prev(inputs.end()));
    auto [_, vjps] = mlx::core::vjp(sdpa, primals, {inputs.back()});
    return vjps;
  };

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (int i = 0; i < /* outputs size */ 3; ++i) {
    shapes.push_back(primals[i].shape());
    dtypes.push_back(primals[i].dtype());
  }
  auto primitive = std::make_shared<ScaledDotProductAttentionVJP>(
      s, fallback, scale_, do_causal_, has_sinks_);
  std::vector<array> inputs = primals;
  inputs.push_back(outputs[0]);
  inputs.push_back(outputs[1]);
  inputs.push_back(cotangents[0]);
  auto vjps = array::make_arrays(std::move(shapes), dtypes, primitive, inputs);

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 3) {
      throw std::invalid_argument(
          "[scale_dot_product_attention] Does not support VJP with respect "
          " to mask or attention sinks.");
    }
    returned_vjps.push_back(std::move(vjps[arg]));
  }
  return returned_vjps;
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other =
      static_cast<const ScaledDotProductAttention&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_ &&
      output_logsumexp_ == a_other.output_logsumexp_;
}

bool QuantizedScaledDotProductAttention::is_equivalent(
    const Primitive& other) const {
  const QuantizedScaledDotProductAttention& a_other =
      static_cast<const QuantizedScaledDotProductAttention&>(other);
  return scale_ == a_other.scale_ && has_arr_mask_ == a_other.has_arr_mask_ &&
      has_sinks_ == a_other.has_sinks_ && do_causal_ == a_other.do_causal_ &&
      key_group_size_ == a_other.key_group_size_ &&
      key_bits_ == a_other.key_bits_ &&
      value_group_size_ == a_other.value_group_size_ &&
      value_bits_ == a_other.value_bits_ &&
      mode_ == a_other.mode_;
}

bool ScaledDotProductAttentionVJP::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttentionVJP& a_other =
      static_cast<const ScaledDotProductAttentionVJP&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_;
}

bool Quantize::is_equivalent(const Primitive& other) const {
  const Quantize& p_other = static_cast<const Quantize&>(other);
  return (
      p_other.group_size_ == group_size_ && p_other.bits_ == bits_ &&
      p_other.mode_ == mode_ && p_other.dequantize_ == dequantize_);
}

std::vector<Shape> Quantize::output_shapes(const std::vector<array>& inputs) {
  auto& w = inputs[0];
  if (dequantize_) {
    auto out_size = w.shape(-1) * 32 / bits_;
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return {std::move(out_shape)};
  } else {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) * bits_ / 32;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size_;
    if (inputs.size() == 2) {
      return {std::move(wq_shape), std::move(sshape)};
    } else {
      auto bshape = sshape;
      return {std::move(wq_shape), std::move(sshape), std::move(bshape)};
    }
  }
}

bool ConvertFP8::is_equivalent(const Primitive& other) const {
  const ConvertFP8& a_other = static_cast<const ConvertFP8&>(other);
  return to_fp8_ == a_other.to_fp8_;
}

} // namespace mlx::core::fast
