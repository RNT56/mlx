// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
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

namespace {

std::vector<array> mixed_quantized_scaled_dot_product_attention_impl(
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
    float sparse_v_threshold,
    bool output_diagnostics,
    StreamOrDevice s /* = {} */) {
  constexpr const char* tag = "mixed_quantized_scaled_dot_product_attention";
  auto qmode = QuantizationMode::Affine;

  auto validate_affine_group_size = [&](std::string_view name,
                                        int group_size) {
    if (group_size != 32 && group_size != 64) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << name
          << " affine group_size must be 32 or 64 but received "
          << group_size << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  validate_affine_group_size("key", key_group_size);
  validate_affine_group_size("value", value_group_size);
  if (key_bits != 8) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] native mixed affine SDPA supports only 8-bit keys for the "
           "K8/Vx path; received key_bits="
        << key_bits << ".";
    throw std::invalid_argument(msg.str());
  }
  if (value_bits != 2 && value_bits != 3 && value_bits != 4) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] native mixed affine SDPA supports value_bits 2, 3, or 4 for "
           "the K8/Vx path; received value_bits="
        << value_bits << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!std::isfinite(sparse_v_threshold) || sparse_v_threshold < 0.0f) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] sparse_v_threshold must be finite and non-negative.";
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
  if (sparse_v_threshold > 0.0f || output_diagnostics) {
    if (query_sequence_length != 1) {
      std::ostringstream msg;
      msg << "[" << tag
          << "] Sparse-V is implemented for decode-only q_seq_len == 1; "
             "received q_seq_len="
          << query_sequence_length << ".";
      throw std::invalid_argument(msg.str());
    }
    if (has_arr_mask) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] Sparse-V does not "
          "support array masks in the native decode path.");
    }
    if (has_sinks) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] Sparse-V does not "
          "support attention sinks in the native decode path.");
    }
  }

  auto stream = to_stream(s);
  auto gqa_factor = n_q_heads / n_kv_heads;
  auto validate_native_support = [&]() {
    if (query_sequence_length > 32) {
      std::ostringstream msg;
      msg << "[" << tag << "] query sequence length "
          << query_sequence_length << " exceeds native K8/Vx limit 32.";
      throw std::invalid_argument(msg.str());
    }
    if (query_sequence_length > key_sequence_length) {
      std::ostringstream msg;
      msg << "[" << tag << "] query sequence length "
          << query_sequence_length
          << " must not exceed key sequence length " << key_sequence_length
          << " for native K8/Vx SDPA.";
      throw std::invalid_argument(msg.str());
    }
    if (!(head_dim == 64 || head_dim == 128 || head_dim == 256 ||
          head_dim == 512)) {
      std::ostringstream msg;
      msg << "[" << tag << "] head dimension " << head_dim
          << " is not native K8/Vx certified; expected one of "
             "{64, 128, 256, 512}.";
      throw std::invalid_argument(msg.str());
    }
    if (gqa_factor > 32) {
      std::ostringstream msg;
      msg << "[" << tag << "] GQA factor " << gqa_factor
          << " exceeds native K8/Vx limit 32.";
      throw std::invalid_argument(msg.str());
    }
    if (stream.device != Device::gpu || !metal::is_available()) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] native K8/Vx SDPA "
          "requires an available Metal GPU stream.");
    }
    if (detail::in_grad_tracing()) {
      throw std::invalid_argument(
          "[mixed_quantized_scaled_dot_product_attention] native K8/Vx SDPA "
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
                   sparse_v_threshold,
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
    if (sparse_v_threshold > 0.0f) {
      scores = where(
          greater_equal(
              scores, array(sparse_v_threshold, scores.dtype()), s),
          scores,
          zeros_like(scores, s),
          s);
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
      qmode,
      sparse_v_threshold,
      output_diagnostics);
  if (output_diagnostics) {
    auto rows = queries.shape(0) * queries.shape(1) * queries.shape(2);
    return array::make_arrays(
        {std::move(out_shape), Shape{rows, 2}},
        {final_type, uint32},
        primitive,
        inputs);
  }
  return {array(std::move(out_shape), final_type, primitive, std::move(inputs))};
}

} // namespace

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
  return mixed_quantized_scaled_dot_product_attention_impl(
      queries,
      keys,
      key_scales,
      key_biases,
      values,
      value_scales,
      value_biases,
      scale,
      mask,
      sinks,
      key_group_size,
      key_bits,
      value_group_size,
      value_bits,
      causal,
      0.0f,
      false,
      s)[0];
}

std::vector<array> mixed_quantized_scaled_dot_product_attention_with_diagnostics(
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
    float sparse_v_threshold /* = 0.0f */,
    StreamOrDevice s /* = {} */) {
  return mixed_quantized_scaled_dot_product_attention_impl(
      queries,
      keys,
      key_scales,
      key_biases,
      values,
      value_scales,
      value_biases,
      scale,
      mask,
      sinks,
      key_group_size,
      key_bits,
      value_group_size,
      value_bits,
      causal,
      sparse_v_threshold,
      true,
      s);
}

std::vector<array> quantize_append_kv(
    const array& k_new,
    const array& v_new,
    const array& k_codes,
    const array& k_scales,
    const array& k_biases,
    const array& v_codes,
    const array& v_scales,
    const array& v_biases,
    int seq_offset,
    int steps,
    int key_group_size,
    int key_bits,
    int value_group_size,
    int value_bits,
    StreamOrDevice s /* = {} */) {
  constexpr const char* tag = "quantize_append_kv";

  // Supported specialization set: K8/gs{64,128}, V4/gs{32,64,128}, power-of-two
  // bits only (the 3/5/6-bit byte-splitting branches are intentionally not
  // ported). Anything else fails closed with an invalid_argument.
  if (key_bits != 8 || (key_group_size != 64 && key_group_size != 128)) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] only key_bits=8 with key_group_size in {64,128} is supported; "
           "received key_bits="
        << key_bits << ", key_group_size=" << key_group_size << ".";
    throw std::invalid_argument(msg.str());
  }
  if (value_bits != 4 ||
      (value_group_size != 32 && value_group_size != 64 &&
       value_group_size != 128)) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] only value_bits=4 with value_group_size in {32,64,128} is "
           "supported; received value_bits="
        << value_bits << ", value_group_size=" << value_group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = k_new.dtype();
  if (!(final_type == float16 || final_type == bfloat16 ||
        final_type == float32)) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] k_new must be float16, bfloat16, or float32; received "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  if (v_new.dtype() != final_type) {
    std::ostringstream msg;
    msg << "[" << tag << "] v_new dtype " << v_new.dtype()
        << " must match k_new dtype " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  for (const auto& t : {k_new, v_new}) {
    if (t.ndim() != 4) {
      std::ostringstream msg;
      msg << "[" << tag << "] incoming rows with shape " << t.shape()
          << " expected to be rank 4 [B, n_kv_heads, steps, head_dim].";
      throw std::invalid_argument(msg.str());
    }
  }
  for (const auto& t : {k_codes, k_scales, k_biases, v_codes, v_scales,
                        v_biases}) {
    if (t.ndim() != 4) {
      std::ostringstream msg;
      msg << "[" << tag << "] cache plane with shape " << t.shape()
          << " expected to be rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (k_codes.dtype() != uint32 || v_codes.dtype() != uint32) {
    throw std::invalid_argument(
        "[quantize_append_kv] code planes must be uint32.");
  }
  if (k_scales.dtype() != final_type || k_biases.dtype() != final_type ||
      v_scales.dtype() != final_type || v_biases.dtype() != final_type) {
    throw std::invalid_argument(
        "[quantize_append_kv] scale/bias planes must match k_new dtype.");
  }

  if (steps <= 0 || seq_offset < 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] steps must be positive and seq_offset "
        << "non-negative; received steps=" << steps
        << ", seq_offset=" << seq_offset << ".";
    throw std::invalid_argument(msg.str());
  }

  auto key_head_dim = k_new.shape(-1);
  auto value_head_dim = v_new.shape(-1);
  if (key_head_dim % key_group_size != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] key head_dim=" << key_head_dim
        << " must be divisible by key_group_size=" << key_group_size << ".";
    throw std::invalid_argument(msg.str());
  }
  if (value_head_dim % value_group_size != 0) {
    std::ostringstream msg;
    msg << "[" << tag << "] value head_dim=" << value_head_dim
        << " must be divisible by value_group_size=" << value_group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  // Incoming rows must carry exactly `steps` rows on the sequence axis (dim -2).
  if (k_new.shape(-2) != steps || v_new.shape(-2) != steps) {
    std::ostringstream msg;
    msg << "[" << tag << "] k_new/v_new seq axis must equal steps=" << steps
        << " but got k_new=" << k_new.shape(-2)
        << ", v_new=" << v_new.shape(-2) << ".";
    throw std::invalid_argument(msg.str());
  }

  // Validate plane geometry: codes packed head_dim*bits/32, scales/biases
  // head_dim/group_size, and matching batch/head dims and capacity headroom.
  auto check_plane = [&](const array& plane,
                         const char* which,
                         int head_dim,
                         int last_dim) {
    if (plane.shape(0) != k_new.shape(0) ||
        plane.shape(1) != k_new.shape(1)) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << which << " batch/head dims "
          << plane.shape() << " must match k_new " << k_new.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    if (plane.shape(-1) != last_dim) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << which << " last dim " << plane.shape(-1)
          << " expected " << last_dim << " for head_dim=" << head_dim << ".";
      throw std::invalid_argument(msg.str());
    }
    if (seq_offset + steps > plane.shape(-2)) {
      std::ostringstream msg;
      msg << "[" << tag << "] " << which << " capacity " << plane.shape(-2)
          << " too small for seq_offset+steps=" << (seq_offset + steps) << ".";
      throw std::invalid_argument(msg.str());
    }
  };
  int key_words = key_head_dim * key_bits / 32;
  int key_groups = key_head_dim / key_group_size;
  int value_words = value_head_dim * value_bits / 32;
  int value_groups = value_head_dim / value_group_size;
  check_plane(k_codes, "k_codes", key_head_dim, key_words);
  check_plane(k_scales, "k_scales", key_head_dim, key_groups);
  check_plane(k_biases, "k_biases", key_head_dim, key_groups);
  check_plane(v_codes, "v_codes", value_head_dim, value_words);
  check_plane(v_scales, "v_scales", value_head_dim, value_groups);
  check_plane(v_biases, "v_biases", value_head_dim, value_groups);

  // Bit-identical fallback: express the exact per-tensor quantize +
  // slice_update ladder in ops. Used off-GPU and when TQ_QAPPEND=0.
  auto fallback = [seq_offset,
                   steps,
                   key_group_size,
                   key_bits,
                   value_group_size,
                   value_bits,
                   key_words,
                   key_groups,
                   value_words,
                   value_groups,
                   s](const std::vector<array>& inputs) -> std::vector<array> {
    const auto& kn = inputs[0];
    const auto& vn = inputs[1];
    auto kq = quantize(kn, key_group_size, key_bits, "affine", std::nullopt, s);
    auto vq =
        quantize(vn, value_group_size, value_bits, "affine", std::nullopt, s);

    auto update_plane = [&](const array& plane,
                            const array& update,
                            int last_dim) {
      Shape start(plane.ndim(), 0);
      start[plane.ndim() - 2] = seq_offset;
      Shape stop = plane.shape();
      stop[plane.ndim() - 2] = seq_offset + steps;
      stop[plane.ndim() - 1] = last_dim;
      return slice_update(plane, update, std::move(start), std::move(stop), s);
    };

    std::vector<array> out;
    out.reserve(6);
    out.push_back(update_plane(inputs[2], kq[0], key_words)); // k_codes
    out.push_back(update_plane(inputs[3], kq[1], key_groups)); // k_scales
    out.push_back(update_plane(inputs[4], kq[2], key_groups)); // k_biases
    out.push_back(update_plane(inputs[5], vq[0], value_words)); // v_codes
    out.push_back(update_plane(inputs[6], vq[1], value_groups)); // v_scales
    out.push_back(update_plane(inputs[7], vq[2], value_groups)); // v_biases
    return out;
  };

  std::vector<array> inputs = {
      k_new, v_new, k_codes, k_scales, k_biases, v_codes, v_scales, v_biases};

  // TQ_QAPPEND=0 is a fail-closed kill switch that returns the bit-identical op
  // ladder directly (observable as a different graph shape). Default (unset or
  // any value != "0") uses the fused primitive.
  if (const char* env = std::getenv("TQ_QAPPEND");
      env && std::string_view(env) == "0") {
    return fallback(inputs);
  }

  auto stream = to_stream(s);
  auto primitive = std::make_shared<QuantizeAppendKV>(
      stream,
      fallback,
      seq_offset,
      steps,
      key_group_size,
      key_bits,
      value_group_size,
      value_bits);
  return array::make_arrays(
      {k_codes.shape(),
       k_scales.shape(),
       k_biases.shape(),
       v_codes.shape(),
       v_scales.shape(),
       v_biases.shape()},
      {uint32, final_type, final_type, uint32, final_type, final_type},
      primitive,
      inputs);
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

TurboQuantSegmentedAttentionBackend tq_segmented_attention_backend_for_codec(
    TurboQuantSegmentedAttentionCodec codec,
    bool allow_experimental_jit,
    Stream stream) {
  switch (codec) {
    case TurboQuantSegmentedAttentionCodec::PolarQJL:
      return tq_segmented_attention_backend(allow_experimental_jit, stream);
    case TurboQuantSegmentedAttentionCodec::PolarWHT:
    case TurboQuantSegmentedAttentionCodec::HybridK8PolarWHTValue:
    default:
      return TurboQuantSegmentedAttentionBackend::Unavailable;
  }
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

constexpr int tq_sparse_v_selection_off = 0;
constexpr int tq_sparse_v_selection_threshold = 1;
constexpr int tq_sparse_v_selection_top_k = 2;
constexpr int tq_sparse_v_selection_cumulative_mass = 3;
constexpr int tq_sparse_v_selection_hybrid_cumulative_mass_top_k = 4;
constexpr int tq_sparse_v_selection_block_threshold = 5;
constexpr int tq_sparse_v_selection_page_top_k = 6;
constexpr int tq_sparse_v_selection_candidate_sparse = 7;
constexpr int tq_candidate_sparse_page_tokens = 512;
constexpr int tq_candidate_sparse_sketch_width = 64;
constexpr int tq_candidate_sparse_default_recent_tokens = 1024;
constexpr int tq_candidate_sparse_default_candidate_pages = 8;

int tq_sparse_v_selection_mode(const TurboQuantAttentionOptions& options) {
  if (options.sparse_v_selection_mode != tq_sparse_v_selection_off) {
    return options.sparse_v_selection_mode;
  }
  return options.sparse_v_threshold > 0.0f ? tq_sparse_v_selection_threshold
                                           : tq_sparse_v_selection_off;
}

bool tq_sparse_v_enabled(const TurboQuantAttentionOptions& options) {
  switch (tq_sparse_v_selection_mode(options)) {
    case tq_sparse_v_selection_threshold:
    case tq_sparse_v_selection_block_threshold:
      return options.sparse_v_threshold > 0.0f;
    case tq_sparse_v_selection_top_k:
    case tq_sparse_v_selection_page_top_k:
    case tq_sparse_v_selection_candidate_sparse:
      return options.sparse_v_top_k > 0;
    case tq_sparse_v_selection_cumulative_mass:
      return options.sparse_v_cumulative_mass > 0.0f;
    case tq_sparse_v_selection_hybrid_cumulative_mass_top_k:
      return options.sparse_v_max_top_k > 0 &&
          options.sparse_v_cumulative_mass > 0.0f;
    case tq_sparse_v_selection_off:
    default:
      return false;
  }
}

bool tq_sparse_v_selection_requires_split(
    const TurboQuantAttentionOptions& options) {
  int mode = tq_sparse_v_selection_mode(options);
  return mode == tq_sparse_v_selection_top_k ||
      mode == tq_sparse_v_selection_cumulative_mass ||
      mode == tq_sparse_v_selection_hybrid_cumulative_mass_top_k ||
      mode == tq_sparse_v_selection_block_threshold ||
      mode == tq_sparse_v_selection_page_top_k ||
      mode == tq_sparse_v_selection_candidate_sparse;
}

int tq_candidate_sparse_recent_tokens(const TurboQuantAttentionOptions& options) {
  return options.sparse_v_recent_tokens > 0
      ? options.sparse_v_recent_tokens
      : tq_candidate_sparse_default_recent_tokens;
}

int tq_candidate_sparse_candidate_pages(const TurboQuantAttentionOptions& options) {
  return options.sparse_v_candidate_pages > 0
      ? options.sparse_v_candidate_pages
      : tq_candidate_sparse_default_candidate_pages;
}

std::string tq_replace_all(
    std::string source,
    std::string_view needle,
    std::string_view replacement) {
  std::size_t pos = 0;
  while ((pos = source.find(needle, pos)) != std::string::npos) {
    source.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
  return source;
}

std::string tq_sparse_selected_partials_source(bool grouped_query) {
  std::string source = grouped_query
      ? std::string(turbo_quant_detail::turbo_quant_sparse_gqa_block_partials_source)
      : std::string(turbo_quant_detail::turbo_quant_sparse_block_partials_source);
  source = tq_replace_all(
      std::move(source),
      "bool skipped = active && (block_threshold_mode\n"
      "            ? block_mass < sparse_v_threshold\n"
      "            : final_weight < sparse_v_threshold);",
      "float selection_cutoff = selection_stats[row * 4u];\n"
      "            bool skipped = active && final_weight < selection_cutoff;");
  return tq_replace_all(
      std::move(source),
      "bool skipped = active && (block_threshold_mode\n"
      "                ? block_mass < sparse_v_threshold\n"
      "                : final_weight < sparse_v_threshold);",
      "float selection_cutoff = selection_stats[row * 4u];\n"
      "            bool skipped = active && final_weight < selection_cutoff;");
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
      "turboquant_attention_fused_decode_runtime_layout_native_s2",
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
      "turboquant_attention_sparse_fused_decode_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k"},
      {"out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_fused_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_page_topk_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_page_topk_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "page_score_tiles"},
      {"out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_page_topk_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

std::string tq_sparse_page_summary_fused_page_topk_attention_source() {
  auto source =
      std::string(turbo_quant_detail::turbo_quant_sparse_page_topk_attention_source);

  auto replace_once = [](std::string& target,
                         const std::string& needle,
                         const std::string& replacement) {
    auto position = target.find(needle);
    if (position == std::string::npos) {
      throw std::runtime_error(
          "[turbo_quant_segmented_attention] failed to specialize fused "
          "pageTopK kernel source.");
    }
    target.replace(position, needle.size(), replacement);
  };

  replace_once(
      source,
      R"TQMLX(        threadgroup float page_scores[THREADS_PER_ROW];
        threadgroup uint page_tokens[THREADS_PER_ROW];
        threadgroup uint retained_pages[8];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];
)TQMLX",
      R"TQMLX(        threadgroup float page_scores[THREADS_PER_ROW];
        threadgroup uint page_tokens[THREADS_PER_ROW];
        threadgroup uint retained_pages[8];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float q_group_abs[GROUPS_PER_VECTOR];
        threadgroup float output_accum[HEAD_DIM];
)TQMLX");

  replace_once(
      source,
      R"TQMLX(        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        if (lane < page_count) {
            page_scores[lane] = page_score_tiles[(row * uint(BLOCK_COUNT)) + lane];
            page_tokens[lane] = lane;
        } else {
            page_scores[lane] = -INFINITY;
            page_tokens[lane] = 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
)TQMLX",
      R"TQMLX(        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        if (lane < uint(GROUPS_PER_VECTOR)) {
            q_group_abs[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(GROUPS_PER_VECTOR)) {
            uint group_start = lane * uint(GROUP_SIZE);
            uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
            float group_abs = 0.0f;
            for (uint local = 0u; local < count; local++) {
                group_abs += fabs(query_cache[group_start + local]);
            }
            q_group_abs[lane] = group_abs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < page_count) {
            float page_score = -INFINITY;
            if (lane < uint(PAGE_CAPACITY)) {
                page_score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint offset =
                        (((batch * uint(KV_HEADS) + kv_head) * uint(PAGE_CAPACITY)
                            + lane) * uint(GROUPS_PER_VECTOR)) + group;
                    page_score += q_group_abs[group] * float(page_summary[offset]);
                }
            }
            page_scores[lane] = page_score;
            page_tokens[lane] = lane;
        } else {
            page_scores[lane] = -INFINITY;
            page_tokens[lane] = 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
)TQMLX");

  return source;
}

const CustomKernelFunction& tq_sparse_page_summary_fused_page_topk_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_page_summary_fused_page_topk_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "page_summary"},
      {"out", "sparse_stats"},
      tq_sparse_page_summary_fused_page_topk_attention_source(),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_candidate_sparse_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_candidate_sparse_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "runtime_sparse_v_recent_tokens",
       "runtime_sparse_v_candidate_pages",
       "key_candidate_sketch"},
      {"out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_candidate_sparse_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_candidate_sparse_gqa_attention_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_candidate_sparse_gqa_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "runtime_sparse_v_recent_tokens",
       "runtime_sparse_v_candidate_pages",
       "key_candidate_sketch"},
      {"out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_candidate_sparse_gqa_attention_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_page_scores_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_page_scores_runtime_layout_native_s2",
      {"q",
       "k_scales",
       "runtime_logical_length",
       "runtime_ring_offset",
       "runtime_pinned_prefix_length"},
      {"page_score_tiles"},
      std::string(turbo_quant_detail::turbo_quant_sparse_page_scores_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_page_summary_scores_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_page_summary_scores_runtime_layout_native_s2",
      {"q",
       "page_summary",
       "runtime_logical_length"},
      {"page_score_tiles"},
      std::string(turbo_quant_detail::turbo_quant_sparse_page_summary_scores_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_block_stats_kernel(bool grouped_query) {
  static CustomKernelFunction generic_kernel = metal_kernel(
      "turboquant_attention_sparse_block_stats_runtime_layout_native_s2",
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
      {"partial_stats", "score_tiles"},
      std::string(turbo_quant_detail::turbo_quant_sparse_block_stats_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  static CustomKernelFunction gqa_kernel = metal_kernel(
      "turboquant_attention_sparse_gqa_block_stats_runtime_layout_native_s2",
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
      {"partial_stats", "score_tiles"},
      std::string(turbo_quant_detail::turbo_quant_sparse_gqa_block_stats_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return grouped_query ? gqa_kernel : generic_kernel;
}

const CustomKernelFunction& tq_sparse_gqa_block_stats_topk_candidates_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_gqa_block_stats_topk_candidates_runtime_layout_native_s2",
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
      {"partial_stats", "candidate_scores", "candidate_tokens"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_gqa_block_stats_topk_candidates_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_block_global_stats_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_block_global_stats_native",
      {"partial_stats"},
      {"global_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_block_global_stats_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_block_selection_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_block_selection_native",
      {"score_tiles",
       "global_stats",
       "runtime_logical_length",
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k"},
      {"selection_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_block_selection_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_local_candidates_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_local_candidates_native",
      {"score_tiles", "runtime_logical_length"},
      {"candidate_scores", "candidate_tokens"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_local_candidates_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_global_selection_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_global_selection_native",
      {"score_tiles", "global_stats", "runtime_logical_length"},
      {"selection_stats", "selected_tokens"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_global_selection_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_compact_selection_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_compact_selection_native",
      {"candidate_scores",
       "candidate_tokens",
       "global_stats",
       "runtime_logical_length"},
      {"selection_stats", "selected_tokens"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_compact_selection_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_global_compact_output_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_global_compact_output_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats"},
      {"out", "sparse_stats"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_global_compact_output_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_candidate_compact_output_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_candidate_compact_output_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "candidate_scores",
       "candidate_tokens",
       "global_stats"},
      {"out", "sparse_stats"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_candidate_compact_output_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_topk_compact_output_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_topk_compact_output_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats",
       "selected_tokens"},
      {"out", "sparse_stats"},
      std::string(
          turbo_quant_detail::turbo_quant_sparse_topk_compact_output_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_sparse_block_partials_kernel(bool grouped_query) {
  static CustomKernelFunction generic_kernel = metal_kernel(
      "turboquant_attention_sparse_block_partials_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats"},
      {"partial_out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_block_partials_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  static CustomKernelFunction gqa_kernel = metal_kernel(
      "turboquant_attention_sparse_gqa_block_partials_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats"},
      {"partial_out", "sparse_stats"},
      std::string(turbo_quant_detail::turbo_quant_sparse_gqa_block_partials_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return grouped_query ? gqa_kernel : generic_kernel;
}

const CustomKernelFunction& tq_sparse_block_selected_partials_kernel(
    bool grouped_query) {
  static CustomKernelFunction generic_kernel = metal_kernel(
      "turboquant_attention_sparse_block_selected_partials_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats",
       "selection_stats"},
      {"partial_out", "sparse_stats"},
      tq_sparse_selected_partials_source(false),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  static CustomKernelFunction gqa_kernel = metal_kernel(
      "turboquant_attention_sparse_gqa_block_selected_partials_runtime_layout_native_s2",
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
       "runtime_sparse_v_threshold",
       "runtime_sparse_v_selection_mode",
       "runtime_sparse_v_top_k",
       "runtime_sparse_v_cumulative_mass",
       "runtime_sparse_v_max_top_k",
       "score_tiles",
       "global_stats",
       "selection_stats"},
      {"partial_out", "sparse_stats"},
      tq_sparse_selected_partials_source(true),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return grouped_query ? gqa_kernel : generic_kernel;
}

const CustomKernelFunction& tq_sparse_block_sum_reduce_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_block_sum_reduce_native",
      {"partial_out"},
      {"out"},
      std::string(turbo_quant_detail::turbo_quant_sparse_block_sum_reduce_source),
      "",
      false);
  return kernel;
}

const CustomKernelFunction& tq_block_partials_kernel(bool grouped_query) {
  static CustomKernelFunction generic_kernel = metal_kernel(
      "turboquant_attention_fused_block_partials_runtime_layout_native_rtu1_s2",
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
       "runtime_block_count"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_block_partials_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  static CustomKernelFunction gqa_kernel = metal_kernel(
      "turboquant_attention_fused_gqa_block_partials_runtime_layout_native_rtu1_s2_rf1",
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
       "runtime_block_count"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_gqa_block_partials_rf1_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return grouped_query ? gqa_kernel : generic_kernel;
}

const CustomKernelFunction& tq_gqa_block_partials_kernel_v7() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_fused_gqa_block_partials_runtime_layout_native_rtu1_s2_v7",
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
       "runtime_block_count"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_gqa_block_partials_v7_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header) +
          std::string(turbo_quant_detail::turbo_quant_attention_header_v7_ext),
      false);
  return kernel;
}

// T2.2 H16 tgmem diet: same source (and back-half algorithm) as the strided/coop
// v6-family GQA block-partials kernels, but partial/tile_scores are staged as half
// and tile_has_weight is a 32-lane bitset, halving static threadgroup memory (see
// mlx-swift Source/MLX/TurboQuant.swift fusedAttentionGQABlockPartialsH16Source /
// roadmap T2.2). Added here for source-copy symmetry with the Swift registration;
// native dispatch wiring (selecting this kernel from the native SDPA call path) is
// NOT implemented in this campaign -- the online-fused decode path used by real
// models routes through the Swift MLXFast.metalKernel registrations, not this C++
// entry point. A future native-route campaign can wire selection here.
const CustomKernelFunction& tq_gqa_block_partials_kernel_h16() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_fused_gqa_block_partials_runtime_layout_native_rtu1_s2_h16_rf1",
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
       "runtime_block_count"},
      {"partial_stats", "partial_out"},
      std::string(turbo_quant_detail::turbo_quant_gqa_block_partials_h16_rf1_source),
      std::string(turbo_quant_detail::turbo_quant_attention_header),
      false);
  return kernel;
}

const CustomKernelFunction& tq_block_reduce_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_fused_block_reduce_native_rtu1_s2",
      {"partial_stats", "partial_out", "runtime_block_count"},
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

const CustomKernelFunction& tq_sparse_extended_diagnostics_kernel() {
  static CustomKernelFunction kernel = metal_kernel(
      "turboquant_attention_sparse_extended_diagnostics_native",
      {"sparse_stats"},
      {"diagnostics"},
      std::string(turbo_quant_detail::turbo_quant_sparse_extended_diagnostics_source),
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
  inputs.push_back(array(tq_sparse_v_selection_mode(options), int32));
  inputs.push_back(array(options.sparse_v_top_k, int32));
  inputs.push_back(array(options.sparse_v_cumulative_mass, float32));
  inputs.push_back(array(options.sparse_v_max_top_k, int32));
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
  // TQ_COOP=0 is a fail-closed kill switch for the C++ coop route. Default
  // (unset or any value != "0") preserves the current env-free behavior.
  if (const char* v = std::getenv("TQ_COOP"); v && std::string_view(v) == "0") {
    return false;
  }
  int repeats = queries.shape(1) / layout.kv_head_count;
  // Coop is excluded from layout v7 (its offsets are hardcoded to the v6 token-major
  // packed/bitset/scale layout): gated to exactly v6, not "v6 or newer".
  bool hd_ok = (layout.head_dimension == 128 || layout.head_dimension == 256);
  // COOPW (T2.4 stage-2): widened in lockstep with the Swift-side
  // turboQuantCooperativeQuadDecodeActive guard now that the shared coop kernel source
  // clamps its r-loops to repeat_count instead of hardcoding r<4u (repeats 2/3/4 all
  // correct from one source). UNTESTED by this campaign: this native C++ route is not
  // exercised by the Swift-route online-fused decode path this campaign validates; the
  // native gqa_kernel selector in fast.cpp still dispatches the same
  // turbo_quant_gqa_block_partials_rf1_source for every admitted repeat count, so no separate
  // native _coopw kernel/name was added here -- flag for a future native-route campaign
  // to verify whether MLX's native kernel cache keys on template params the way the Swift
  // MLXFast.metalKernel path does before relying on this.
  // Context floor. Default 32768 (coop pays off once KV traffic dominates the
  // decode; strided is neutral/better at short context). TQ_COOP_MIN_CONTEXT
  // is the measurement override, mirroring the Swift-side
  // turboQuantCooperativeQuadDecodeActive gate: it lets an A/B engage the coop
  // (LANES_PER_TOKEN=4, kernel kind 4) branch below the production floor at
  // tractable prefill contexts. Unset => behavior identical to the previous
  // hardcoded 32768. NOTE: this gate also feeds the sparse block-stats call
  // site; measurement runs must keep sparse off.
  int coop_min_context = 32768;
  if (const char* v = std::getenv("TQ_COOP_MIN_CONTEXT")) {
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed > 0 && parsed <= (1 << 30)) {
      coop_min_context = static_cast<int>(parsed);
    }
  }
  if (repeats < 2 || repeats > 4 || !hd_ok || precision.group_size <= 0 ||
      layout.logical_length < coop_min_context || layout.layout_version != 6) {
    return false;
  }
  bool uniform = precision.key_base_bits == precision.key_high_bits;
  bool split = precision.key_high_bits == precision.key_base_bits + 1;
  int chunk = layout.head_dimension / 4;
  return (uniform || split) && chunk <= precision.group_size &&
      precision.group_size % chunk == 0;
}

bool tq_valid_page_summary(
    const array* page_summary,
    const TurboQuantAttentionLayoutDescriptor& layout,
    int active_blocks) {
  if (page_summary == nullptr || page_summary->dtype() != float32 ||
      page_summary->ndim() != 4 || layout.ring_offset != 0 ||
      layout.pinned_prefix_length != 0) {
    return false;
  }
  return page_summary->shape(0) == layout.batch_size &&
      page_summary->shape(1) == layout.kv_head_count &&
      page_summary->shape(2) >= active_blocks &&
      page_summary->shape(3) == layout.groups_per_vector;
}

bool tq_valid_candidate_sketch(
    const array* candidate_sketch,
    const TurboQuantAttentionLayoutDescriptor& layout,
    int active_pages) {
  if (candidate_sketch == nullptr || candidate_sketch->dtype() != float32 ||
      candidate_sketch->ndim() != 4 || layout.ring_offset != 0) {
    return false;
  }
  return candidate_sketch->shape(0) == layout.batch_size &&
      candidate_sketch->shape(1) == layout.kv_head_count &&
      candidate_sketch->shape(2) >= active_pages &&
      candidate_sketch->shape(3) == tq_candidate_sparse_sketch_width;
}

bool tq_sparse_page_fused_enabled() {
  const char* value = std::getenv("TURBOQUANT_SPARSE_V_PAGE_FUSED");
  return value == nullptr || std::string_view(value) != "0";
}

bool tq_candidate_sparse_fused_enabled() {
  const char* value = std::getenv("TURBOQUANT_CANDIDATE_SPARSE_FUSED");
  return value != nullptr && std::string_view(value) == "1";
}

int tq_sparse_page_recent_tokens() {
  const char* value = std::getenv("TURBOQUANT_SPARSE_V_PAGE_RECENT_TOKENS");
  if (value == nullptr) {
    return 0;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 0;
  }
  return static_cast<int>(std::min<long>(parsed, 1 << 20));
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
    const array* key_page_summary,
    const array* key_candidate_sketch,
    bool output_diagnostics,
    StreamOrDevice stream) {
  Dtype output_dtype = queries.dtype();
  Shape output_shape{
      queries.shape(0), queries.shape(1), queries.shape(2), layout.head_dimension};
  int row_count = queries.shape(0) * queries.shape(1) * queries.shape(2);
  int repeats = queries.shape(1) / layout.kv_head_count;
  int flags = (options.causal ? 1 : 0);

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
  int sparse_selection_mode = tq_sparse_v_selection_mode(options);
  bool sparse_requires_split = tq_sparse_v_selection_requires_split(options);
  bool sparse_uses_selection_stats =
      sparse_selection_mode == tq_sparse_v_selection_top_k ||
      sparse_selection_mode == tq_sparse_v_selection_cumulative_mass ||
      sparse_selection_mode ==
          tq_sparse_v_selection_hybrid_cumulative_mass_top_k;
  if (sparse_requires_split && block_width == 0 && queries.shape(2) == 1) {
    block_width = tq_next_power_of_two_clamped(
        std::max(layout.head_dimension, 512), 512);
  }
  bool sparse_single_block_topk = sparse_selection_mode == tq_sparse_v_selection_top_k &&
      block_width > 0 && queries.shape(2) == 1 &&
      layout.logical_length <= block_width;
  bool sparse_topk_compact = sparse_selection_mode == tq_sparse_v_selection_top_k &&
      block_width > 0 && queries.shape(2) == 1 &&
      options.sparse_v_top_k > 0 &&
      options.sparse_v_top_k < layout.logical_length &&
      options.sparse_v_top_k <= 512;
  bool sparse_page_topk = sparse_selection_mode == tq_sparse_v_selection_page_top_k &&
      block_width > 0 && queries.shape(2) == 1 &&
      options.sparse_v_top_k > 0;
  bool sparse_candidate_requested =
      sparse_selection_mode == tq_sparse_v_selection_candidate_sparse &&
      tq_sparse_v_enabled(options);
  int sparse_candidate_pages = sparse_candidate_requested
      ? (layout.logical_length + tq_candidate_sparse_page_tokens - 1) /
          tq_candidate_sparse_page_tokens
      : 0;
  int sparse_dense_fallback_code = 0;
  if (sparse_candidate_requested) {
    int candidate_pages = tq_candidate_sparse_candidate_pages(options);
    int older_top_k = options.sparse_v_top_k;
    bool valid_candidate_shape =
        tq_valid_candidate_sketch(key_candidate_sketch, layout, sparse_candidate_pages);
    bool valid_candidate_limits = queries.shape(2) == 1 &&
        candidate_pages > 0 && candidate_pages <= 16 &&
        older_top_k > 0 && older_top_k <= 4096 &&
        sparse_candidate_pages > 0 && sparse_candidate_pages <= 512;
    if (!valid_candidate_shape) {
      sparse_dense_fallback_code = 1;
    } else if (!valid_candidate_limits) {
      sparse_dense_fallback_code = 2;
    }
  }

  if (sparse_candidate_requested && sparse_dense_fallback_code == 0) {
    int candidate_pages = std::min(tq_candidate_sparse_candidate_pages(options), 16);
    int older_top_k = std::min(options.sparse_v_top_k, 4096);
    int candidate_token_limit = candidate_pages * tq_candidate_sparse_page_tokens;
    int threadgroup_width = tq_next_power_of_two_clamped(
        std::max({layout.head_dimension, tq_candidate_sparse_page_tokens, 256}), 512);
    bool cooperative_candidate =
        tq_candidate_sparse_fused_enabled() && repeats > 1 &&
        queries.shape(2) == 1 && candidate_pages <= 2 &&
        older_top_k <= 512 && layout.head_dimension <= 256;
    if (cooperative_candidate) {
      int repeat_group_count = (repeats + 3) / 4;
      int gqa_row_count = queries.shape(0) * layout.kv_head_count * queries.shape(2);
      auto candidate_template =
          tq_attention_value_template(queries, layout, precision, options, output_dtype);
      candidate_template.push_back({"THREADS_PER_ROW", threadgroup_width});
      candidate_template.push_back({"CANDIDATE_PAGE_LIMIT", candidate_pages});
      candidate_template.push_back({"CANDIDATE_TOKEN_LIMIT", candidate_token_limit});
      candidate_template.push_back({"TOPK_LIMIT", older_top_k});
      candidate_template.push_back({"PAGE_CAPACITY", key_candidate_sketch->shape(2)});
      candidate_template.push_back({"GQA_REPEATS", repeats});
      candidate_template.push_back({"REPEAT_GROUP_COUNT", repeat_group_count});
      candidate_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
      auto candidate_inputs = tq_native_sparse_inputs(
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
      candidate_inputs.push_back(
          array(tq_candidate_sparse_recent_tokens(options), int32));
      candidate_inputs.push_back(
          array(tq_candidate_sparse_candidate_pages(options), int32));
      candidate_inputs.push_back(*key_candidate_sketch);
      auto candidate_outputs = tq_candidate_sparse_gqa_attention_kernel()(
          candidate_inputs,
          {output_shape, Shape{row_count, 8}},
          {output_dtype, uint32},
          {gqa_row_count * repeat_group_count * threadgroup_width, 1, 1},
          {threadgroup_width, 1, 1},
          std::move(candidate_template),
          std::nullopt,
          false,
          stream);
      flags |= 16 | ((sparse_selection_mode & 7) << 5);
      if (output_diagnostics) {
        auto diagnostics = tq_sparse_extended_diagnostics_kernel()(
            {candidate_outputs[1]},
            {Shape{16}},
            {int32},
            {256, 1, 1},
            {256, 1, 1},
            {{"ROW_COUNT", row_count},
             {"BACKEND_VERSION", options.backend_version},
             {"KERNEL_KIND", 20},
             {"ACTIVE_BLOCKS", sparse_candidate_pages},
             {"BLOCK_TOKENS", tq_candidate_sparse_page_tokens},
             {"FALLBACK_CODE", 0},
             {"FLAGS", flags}},
            std::nullopt,
            false,
            stream)[0];
        return {candidate_outputs[0], diagnostics};
      }
      return {candidate_outputs[0]};
    }
    auto candidate_template =
        tq_attention_value_template(queries, layout, precision, options, output_dtype);
    candidate_template.push_back({"THREADS_PER_ROW", threadgroup_width});
    candidate_template.push_back({"CANDIDATE_PAGE_LIMIT", candidate_pages});
    candidate_template.push_back({"CANDIDATE_TOKEN_LIMIT", candidate_token_limit});
    candidate_template.push_back({"TOPK_LIMIT", older_top_k});
    candidate_template.push_back({"PAGE_CAPACITY", key_candidate_sketch->shape(2)});
    candidate_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
    auto candidate_inputs = tq_native_sparse_inputs(
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
    candidate_inputs.push_back(
        array(tq_candidate_sparse_recent_tokens(options), int32));
    candidate_inputs.push_back(
        array(tq_candidate_sparse_candidate_pages(options), int32));
    candidate_inputs.push_back(*key_candidate_sketch);
    auto candidate_outputs = tq_candidate_sparse_attention_kernel()(
        candidate_inputs,
        {output_shape, Shape{row_count, 8}},
        {output_dtype, uint32},
        {row_count * threadgroup_width, 1, 1},
        {threadgroup_width, 1, 1},
        std::move(candidate_template),
        std::nullopt,
        false,
        stream);
    flags |= 16 | ((sparse_selection_mode & 7) << 5);
    if (output_diagnostics) {
      auto diagnostics = tq_sparse_extended_diagnostics_kernel()(
          {candidate_outputs[1]},
          {Shape{16}},
          {int32},
          {256, 1, 1},
          {256, 1, 1},
          {{"ROW_COUNT", row_count},
           {"BACKEND_VERSION", options.backend_version},
           {"KERNEL_KIND", 19},
           {"ACTIVE_BLOCKS", sparse_candidate_pages},
           {"BLOCK_TOKENS", tq_candidate_sparse_page_tokens},
           {"FALLBACK_CODE", 0},
           {"FLAGS", flags}},
          std::nullopt,
          false,
          stream)[0];
      return {candidate_outputs[0], diagnostics};
    }
    return {candidate_outputs[0]};
  }

  if (sparse_candidate_requested && sparse_dense_fallback_code != 0) {
    flags |= 16 | ((sparse_selection_mode & 7) << 5);
  }

  if (tq_sparse_v_enabled(options) && !sparse_candidate_requested) {
    if (sparse_page_topk) {
      int active_blocks = (layout.logical_length + block_width - 1) / block_width;
      int page_recent_tokens = tq_sparse_page_recent_tokens();
      bool use_page_summary =
          tq_valid_page_summary(key_page_summary, layout, active_blocks);
      bool use_fused_page_summary = use_page_summary &&
          tq_sparse_page_fused_enabled() &&
          options.sparse_v_top_k <= 8 &&
          active_blocks <= 512;
      std::optional<array> page_scores;
      if (use_page_summary && !use_fused_page_summary) {
        auto page_score_template =
            tq_runtime_layout_attention_template(
                queries, layout, precision, options, float32);
        page_score_template.push_back({"THREADS_PER_BLOCK", block_width});
        page_score_template.push_back({"BLOCK_TOKENS", block_width});
        page_score_template.push_back({"BLOCK_COUNT", active_blocks});
        page_score_template.push_back({"PAGE_CAPACITY", key_page_summary->shape(2)});
        page_scores = tq_sparse_page_summary_scores_kernel()(
            {queries,
             *key_page_summary,
             array(layout.logical_length, int32)},
            {Shape{row_count, active_blocks}},
            {float32},
            {row_count * active_blocks * block_width, 1, 1},
            {block_width, 1, 1},
            std::move(page_score_template),
            std::nullopt,
            false,
            stream)[0];
      } else if (!use_page_summary) {
        auto page_score_template =
            tq_runtime_layout_attention_template(
                queries, layout, precision, options, float32);
        page_score_template.push_back({"THREADS_PER_BLOCK", block_width});
        page_score_template.push_back({"BLOCK_TOKENS", block_width});
        page_score_template.push_back({"BLOCK_COUNT", active_blocks});
        page_score_template.push_back({"PAGE_SCORE_SAMPLES", 8});
        page_scores = tq_sparse_page_scores_kernel()(
            {queries,
             key_scales,
             array(layout.logical_length, int32),
             array(layout.ring_offset, int32),
             array(layout.pinned_prefix_length, int32)},
            {Shape{row_count, active_blocks}},
            {float32},
            {row_count * active_blocks * block_width, 1, 1},
            {block_width, 1, 1},
            std::move(page_score_template),
            std::nullopt,
            false,
            stream)[0];
      }
      int page_width = tq_next_power_of_two_clamped(
          std::max(
              std::max(layout.head_dimension, block_width),
              std::max(active_blocks, 256)),
          512);
      auto page_template =
          tq_attention_value_template(queries, layout, precision, options, output_dtype);
      page_template.push_back({"THREADS_PER_ROW", page_width});
      page_template.push_back({"BLOCK_TOKENS", block_width});
      page_template.push_back({"BLOCK_COUNT", active_blocks});
      page_template.push_back({"PAGE_RECENT_TOKENS", page_recent_tokens});
      page_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
      auto page_inputs = tq_native_sparse_inputs(
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
      std::vector<array> page_outputs;
      if (use_fused_page_summary) {
        page_template.push_back({"PAGE_CAPACITY", key_page_summary->shape(2)});
        page_inputs.push_back(*key_page_summary);
        page_outputs = tq_sparse_page_summary_fused_page_topk_attention_kernel()(
            page_inputs,
            {output_shape, Shape{row_count, 2}},
            {output_dtype, uint32},
            {row_count * page_width, 1, 1},
            {page_width, 1, 1},
            std::move(page_template),
            std::nullopt,
            false,
            stream);
      } else {
        page_inputs.push_back(*page_scores);
        page_outputs = tq_sparse_page_topk_attention_kernel()(
            page_inputs,
            {output_shape, Shape{row_count, 2}},
            {output_dtype, uint32},
            {row_count * page_width, 1, 1},
            {page_width, 1, 1},
            std::move(page_template),
            std::nullopt,
            false,
            stream);
      }
      flags |= 16 | ((sparse_selection_mode & 7) << 5);
      if (output_diagnostics) {
        int kernel_kind = use_fused_page_summary
            ? (page_recent_tokens > 0 ? 18 : 15)
            : (use_page_summary
                   ? (page_recent_tokens > 0 ? 17 : 14)
                   : (page_recent_tokens > 0 ? 16 : 13));
        auto diagnostics = tq_sparse_diagnostics_kernel()(
            {page_outputs[1]},
            {Shape{8}},
            {int32},
            {256, 1, 1},
            {256, 1, 1},
            {{"ROW_COUNT", row_count},
             {"BACKEND_VERSION", options.backend_version},
             {"KERNEL_KIND", kernel_kind},
             {"ACTIVE_BLOCKS", active_blocks},
             {"BLOCK_TOKENS", block_width},
             {"FALLBACK_CODE", 0},
             {"FLAGS", flags}},
            std::nullopt,
            false,
            stream)[0];
        return {page_outputs[0], diagnostics};
      }
      return {page_outputs[0]};
    }

    if (block_width > 0 && !sparse_single_block_topk) {
      int active_blocks = (layout.logical_length + block_width - 1) / block_width;
      bool grouped_query = repeats > 1 && repeats <= 4;
      bool coop = grouped_query &&
          tq_cooperative_gqa_path_allowed(queries, layout, precision);
      bool sparse_topk_fused_candidate_stats = false;
      int partial_rows = grouped_query
          ? queries.shape(0) * layout.kv_head_count * queries.shape(2)
          : row_count;
      auto stats_template =
          tq_attention_value_template(queries, layout, precision, options, output_dtype);
      stats_template.push_back({"THREADS_PER_BLOCK", block_width});
      stats_template.push_back({"BLOCK_TOKENS", block_width});
      stats_template.push_back({"BLOCK_COUNT", active_blocks});
      stats_template.push_back({"GQA_REPEATS", grouped_query ? repeats : 1});
      stats_template.push_back({"LANES_PER_TOKEN", coop ? 4 : 1});
      std::vector<array> stats_outputs;
      if (sparse_topk_fused_candidate_stats) {
        int topk_limit = std::min(options.sparse_v_top_k, layout.logical_length);
        stats_template.push_back({"TOPK_LIMIT", topk_limit});
        stats_outputs = tq_sparse_gqa_block_stats_topk_candidates_kernel()(
            inputs,
            {Shape{row_count, active_blocks, 2},
             Shape{row_count, active_blocks, topk_limit},
             Shape{row_count, active_blocks, topk_limit}},
            {float32, float32, int32},
            {partial_rows * active_blocks * block_width, 1, 1},
            {block_width, 1, 1},
            std::move(stats_template),
            std::nullopt,
            false,
            stream);
      } else {
        stats_outputs = tq_sparse_block_stats_kernel(grouped_query)(
            inputs,
            {Shape{row_count, active_blocks, 2},
             Shape{row_count, active_blocks, block_width}},
            {float32, float32},
            {partial_rows * active_blocks * block_width, 1, 1},
            {block_width, 1, 1},
            std::move(stats_template),
            std::nullopt,
            false,
            stream);
      }
      auto partial_stats = stats_outputs[0];
      std::optional<array> score_tiles;
      if (!sparse_topk_fused_candidate_stats) {
        score_tiles = stats_outputs[1];
      }

      int stats_reduce_width =
          tq_next_power_of_two_clamped(std::max(active_blocks, 256), 512);
      auto global_stats = tq_sparse_block_global_stats_kernel()(
          {partial_stats},
          {Shape{row_count, 2}},
          {float32},
	          {row_count * stats_reduce_width, 1, 1},
	          {stats_reduce_width, 1, 1},
	          {{"ROW_COUNT", row_count},
	           {"BLOCK_COUNT", active_blocks},
	           {"THREADS_PER_BLOCK", stats_reduce_width}},
	          std::nullopt,
	          false,
	          stream)[0];

	      if (sparse_topk_compact) {
	        int topk_limit = std::min(options.sparse_v_top_k, layout.logical_length);
	        auto compact_template =
	            tq_attention_value_template(queries, layout, precision, options, output_dtype);
	        int compact_width = tq_next_power_of_two_clamped(
	            std::max(layout.head_dimension, 256), 512);
	        compact_template.push_back({"THREADS_PER_ROW", compact_width});
	        compact_template.push_back({"TOPK_LIMIT", topk_limit});
	        compact_template.push_back({"BLOCK_COUNT", active_blocks});
	        compact_template.push_back({"BLOCK_TOKENS", block_width});
	        compact_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
	        std::vector<array> compact_outputs;
	        if (active_blocks >= 16) {
	          std::vector<array> local_candidates;
	          if (sparse_topk_fused_candidate_stats) {
	            local_candidates = {stats_outputs[1], stats_outputs[2]};
	          } else {
	            local_candidates = tq_sparse_topk_local_candidates_kernel()(
	                {*score_tiles, array(layout.logical_length, int32)},
	                {Shape{row_count, active_blocks, topk_limit},
	                 Shape{row_count, active_blocks, topk_limit}},
	                {float32, int32},
	                {row_count * active_blocks * block_width, 1, 1},
	                {block_width, 1, 1},
	                {{"ROW_COUNT", row_count},
	                 {"BLOCK_COUNT", active_blocks},
	                 {"BLOCK_TOKENS", block_width},
	                 {"THREADS_PER_BLOCK", block_width},
	                 {"TOPK_LIMIT", topk_limit}},
	                std::nullopt,
	                false,
	                stream);
	          }
	          auto compact_inputs = tq_native_sparse_inputs(
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
	          compact_inputs.push_back(local_candidates[0]);
	          compact_inputs.push_back(local_candidates[1]);
	          compact_inputs.push_back(global_stats);
	          compact_outputs = tq_sparse_topk_candidate_compact_output_kernel()(
	              compact_inputs,
	              {output_shape, Shape{row_count, 2}},
	              {output_dtype, uint32},
	              {row_count * compact_width, 1, 1},
	              {compact_width, 1, 1},
	              compact_template,
	              std::nullopt,
	              false,
	              stream);
	        } else {
	          auto compact_inputs = tq_native_sparse_inputs(
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
	          compact_inputs.push_back(*score_tiles);
	          compact_inputs.push_back(global_stats);
	          compact_outputs = tq_sparse_topk_global_compact_output_kernel()(
	              compact_inputs,
	              {output_shape, Shape{row_count, 2}},
	              {output_dtype, uint32},
	              {row_count * compact_width, 1, 1},
	              {compact_width, 1, 1},
	              std::move(compact_template),
	              std::nullopt,
	              false,
	              stream);
	        }

	        flags |= 2 | 16 | ((sparse_selection_mode & 7) << 5) |
	            (grouped_query ? 4 : 0);
	        if (output_diagnostics) {
	          auto diagnostics = tq_sparse_diagnostics_kernel()(
	              {compact_outputs[1]},
	              {Shape{8}},
	              {int32},
	              {256, 1, 1},
	              {256, 1, 1},
	              {{"ROW_COUNT", row_count},
	               {"BACKEND_VERSION", options.backend_version},
	               {"KERNEL_KIND", 12},
	               {"ACTIVE_BLOCKS", active_blocks},
	               {"BLOCK_TOKENS", block_width},
	               {"FALLBACK_CODE", 0},
	               {"FLAGS", flags}},
	              std::nullopt,
	              false,
	              stream)[0];
	          return {compact_outputs[0], diagnostics};
	        }
	        return {compact_outputs[0]};
	      }

	      std::optional<array> selection_stats;
      if (sparse_uses_selection_stats) {
        int selection_width = 512;
        selection_stats = tq_sparse_block_selection_kernel()(
            {*score_tiles,
             global_stats,
             array(layout.logical_length, int32),
             array(options.sparse_v_threshold, float32),
             array(sparse_selection_mode, int32),
             array(options.sparse_v_top_k, int32),
             array(options.sparse_v_cumulative_mass, float32),
             array(options.sparse_v_max_top_k, int32)},
            {Shape{row_count, 4}},
            {float32},
            {row_count * selection_width, 1, 1},
            {selection_width, 1, 1},
            {{"ROW_COUNT", row_count},
             {"BLOCK_COUNT", active_blocks},
             {"BLOCK_TOKENS", block_width},
             {"THREADS_PER_BLOCK", selection_width}},
            std::nullopt,
            false,
            stream)[0];
      }

      auto partial_template =
          tq_attention_value_template(queries, layout, precision, options, output_dtype);
      partial_template.push_back({"THREADS_PER_BLOCK", block_width});
      partial_template.push_back({"BLOCK_TOKENS", block_width});
      partial_template.push_back({"BLOCK_COUNT", active_blocks});
      partial_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
      partial_template.push_back({"GQA_REPEATS", grouped_query ? repeats : 1});
      Dtype partial_dtype = output_dtype == float32 ? float32 : output_dtype;
      auto sparse_inputs = tq_native_sparse_inputs(
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
      sparse_inputs.push_back(*score_tiles);
      sparse_inputs.push_back(global_stats);
      const auto& partial_kernel = sparse_uses_selection_stats
          ? tq_sparse_block_selected_partials_kernel(grouped_query)
          : tq_sparse_block_partials_kernel(grouped_query);
      if (sparse_uses_selection_stats) {
        sparse_inputs.push_back(*selection_stats);
      }
      auto partials = partial_kernel(
          sparse_inputs,
          {Shape{row_count, active_blocks, layout.head_dimension},
           Shape{row_count, active_blocks, 2}},
          {partial_dtype, uint32},
          {partial_rows * active_blocks * block_width, 1, 1},
          {block_width, 1, 1},
          std::move(partial_template),
          std::nullopt,
          false,
          stream);

      int reduce_width = tq_next_power_of_two_clamped(
          std::max(active_blocks, layout.head_dimension), 512);
      auto output = tq_sparse_block_sum_reduce_kernel()(
          {partials[0]},
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

      flags |= 2 | 16 | ((sparse_selection_mode & 7) << 5) |
          (grouped_query ? 4 : 0) | (coop ? 8 : 0);
      if (output_diagnostics) {
        int kernel_kind = sparse_uses_selection_stats
            ? (coop ? 11 : (grouped_query ? 10 : 9))
            : (coop ? 8 : (grouped_query ? 7 : 6));
        auto diagnostics = tq_sparse_diagnostics_kernel()(
            {partials[1]},
            {Shape{8}},
            {int32},
            {256, 1, 1},
            {256, 1, 1},
            {{"ROW_COUNT", row_count * active_blocks},
             {"BACKEND_VERSION", options.backend_version},
             {"KERNEL_KIND", kernel_kind},
             {"ACTIVE_BLOCKS", active_blocks},
             {"BLOCK_TOKENS", block_width},
             {"FALLBACK_CODE", 0},
             {"FLAGS", flags}},
            std::nullopt,
            false,
            stream)[0];
        return {output, diagnostics};
      }
      return {output};
    }

    if (sparse_requires_split && !sparse_single_block_topk) {
      throw std::invalid_argument(
          "[turbo_quant_segmented_attention] split Sparse-V selection modes "
          "require decode-only split-K native attention.");
    }

    int threadgroup_width = sparse_single_block_topk
        ? block_width
        : tq_next_power_of_two_clamped(std::max(layout.head_dimension, 256), 256);
    auto sparse_template =
        tq_attention_value_template(queries, layout, precision, options, output_dtype);
    sparse_template.push_back({"THREADS_PER_ROW", threadgroup_width});
    sparse_template.push_back({"OUTPUT_SPARSE_STATS", output_diagnostics});
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
    flags |= 16 | ((sparse_selection_mode & 7) << 5);
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
  if (block_width > 0) {
    int active_blocks = (layout.logical_length + block_width - 1) / block_width;
    bool tile_transposed_v7 = layout.layout_version == 7;
    bool grouped_query = repeats > 1 && repeats <= 4;
    bool coop = !tile_transposed_v7 && grouped_query &&
        tq_cooperative_gqa_path_allowed(queries, layout, precision);
    if (tile_transposed_v7 && !grouped_query) {
      throw std::invalid_argument(
          "[turbo_quant_segmented_attention] layout v7 requires the "
          "grouped-query block-partials kernel (query head repeats 2...4).");
    }
    int partial_rows = grouped_query
        ? queries.shape(0) * layout.kv_head_count * queries.shape(2)
        : row_count;
    auto partial_template =
        tq_attention_value_template(queries, layout, precision, options, output_dtype);
    partial_template.push_back({"THREADS_PER_BLOCK", block_width});
    partial_template.push_back({"BLOCK_TOKENS", block_width});
    partial_template.push_back({"GQA_REPEATS", grouped_query ? repeats : 1});
    if (!tile_transposed_v7) {
      partial_template.push_back({"LANES_PER_TOKEN", coop ? 4 : 1});
    }

    Dtype partial_dtype = output_dtype == float32 ? float32 : output_dtype;
    auto partial_inputs = inputs;
    partial_inputs.push_back(array(active_blocks, int32));
    auto partials = (tile_transposed_v7
                          ? tq_gqa_block_partials_kernel_v7()
                          : tq_block_partials_kernel(grouped_query))(
        partial_inputs,
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
    std::vector<array> reduce_inputs = partials;
    reduce_inputs.push_back(array(active_blocks, int32));
    auto output = tq_block_reduce_kernel()(
        reduce_inputs,
        {output_shape},
        {output_dtype},
        {row_count * reduce_width, 1, 1},
        {reduce_width, 1, 1},
        {{"ROW_COUNT", row_count},
         {"HEAD_DIM", layout.head_dimension},
         {"THREADS_PER_BLOCK", reduce_width},
         {"OUTPUT_DTYPE", output_dtype}},
        std::nullopt,
        false,
        stream)[0];

    int kernel_kind =
        tile_transposed_v7 ? 12 : (coop ? 4 : (grouped_query ? 3 : 2));
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
                  sparse_dense_fallback_code,
                  flags)};
    }
    return {output};
  }

  if (layout.layout_version == 7) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] layout v7 is only ported to the "
        "block-partials path; the fused single-pass kernel is not ported.");
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
                sparse_dense_fallback_code,
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
      layout.layout_version != 6 && layout.layout_version != 7) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] unsupported layout version "
        << layout.layout_version << ".";
    throw std::invalid_argument(msg.str());
  }
  if (layout.layout_version == 7 && layout.capacity % 32 != 0) {
    std::ostringstream msg;
    msg << "[" << tq_sdpa_tag << "] layout v7 requires capacity to be a "
        << "multiple of 32; received " << layout.capacity << ".";
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
  int sparse_selection_mode = tq_sparse_v_selection_mode(options);
  if (sparse_selection_mode < tq_sparse_v_selection_off ||
      sparse_selection_mode > tq_sparse_v_selection_candidate_sparse) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] sparse_v_selection_mode must be "
        "0 (off), 1 (threshold), 2 (top-k), 3 (cumulative mass), or 4 "
        "(hybrid cumulative mass plus max top-k), 5 (block threshold), or "
        "6 (page top-k), or 7 (candidate sparse).");
  }
  if (layout.layout_version == 7 &&
      (sparse_selection_mode != 0 || options.sparse_v_threshold > 0)) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] sparse paths are not ported to "
        "layout v7.");
  }
  if (options.sparse_v_top_k < 0 || options.sparse_v_max_top_k < 0 ||
      options.sparse_v_recent_tokens < 0 ||
      options.sparse_v_candidate_pages < 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] sparse_v_top_k, "
        "sparse_v_max_top_k, sparse_v_recent_tokens, and "
        "sparse_v_candidate_pages must be non-negative.");
  }
  if (!std::isfinite(options.sparse_v_cumulative_mass) ||
      options.sparse_v_cumulative_mass < 0.0f ||
      options.sparse_v_cumulative_mass > 1.0f) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] sparse_v_cumulative_mass must be "
        "finite and in [0, 1].");
  }
  if (sparse_selection_mode == tq_sparse_v_selection_top_k &&
      options.sparse_v_top_k <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] top-k Sparse-V requires "
        "sparse_v_top_k > 0.");
  }
  if (sparse_selection_mode == tq_sparse_v_selection_page_top_k &&
      options.sparse_v_top_k <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] page top-k Sparse-V requires "
        "sparse_v_top_k > 0.");
  }
  if (sparse_selection_mode == tq_sparse_v_selection_cumulative_mass &&
      options.sparse_v_cumulative_mass <= 0.0f) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] cumulative Sparse-V requires "
        "sparse_v_cumulative_mass > 0.");
  }
  if (sparse_selection_mode ==
          tq_sparse_v_selection_hybrid_cumulative_mass_top_k &&
      (options.sparse_v_cumulative_mass <= 0.0f ||
       options.sparse_v_max_top_k <= 0)) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] hybrid Sparse-V requires "
        "sparse_v_cumulative_mass > 0 and sparse_v_max_top_k > 0.");
  }
  if (sparse_selection_mode == tq_sparse_v_selection_block_threshold &&
      options.sparse_v_threshold <= 0.0f) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] block-threshold Sparse-V requires "
        "sparse_v_threshold > 0.");
  }
  if (sparse_selection_mode == tq_sparse_v_selection_candidate_sparse &&
      options.sparse_v_top_k <= 0) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] candidateSparse Sparse-V requires "
        "sparse_v_top_k > 0.");
  }
  if (tq_sparse_v_selection_requires_split(options) && queries.shape(2) != 1) {
    throw std::invalid_argument(
        "[turbo_quant_segmented_attention] split Sparse-V selection modes are "
        "decode-only and require qLen == 1.");
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
    const array* key_page_summary,
    const array* key_candidate_sketch,
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
      key_page_summary,
      key_candidate_sketch,
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

TurboQuantSegmentedAttentionBackend
turbo_quant_segmented_attention_backend_for_codec(
    TurboQuantSegmentedAttentionCodec codec,
    bool allow_experimental_jit,
    StreamOrDevice s) {
  return tq_segmented_attention_backend_for_codec(
      codec, allow_experimental_jit, to_stream(s));
}

bool turbo_quant_segmented_attention_is_available_for_codec(
    TurboQuantSegmentedAttentionCodec codec,
    bool allow_experimental_jit,
    StreamOrDevice s) {
  return turbo_quant_segmented_attention_backend_for_codec(
             codec, allow_experimental_jit, s) !=
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
      nullptr,
      nullptr,
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
      nullptr,
      nullptr,
      true,
      s);
}

array turbo_quant_segmented_attention_with_page_summaries(
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
    const array& key_page_summary,
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
      &key_page_summary,
      nullptr,
      false,
      s)[0];
}

std::vector<array>
turbo_quant_segmented_attention_with_page_summaries_and_diagnostics(
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
    const array& key_page_summary,
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
      &key_page_summary,
      nullptr,
      true,
      s);
}

array turbo_quant_segmented_attention_with_candidate_sketches(
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
    const array& key_candidate_sketch,
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
      nullptr,
      &key_candidate_sketch,
      false,
      s)[0];
}

std::vector<array>
turbo_quant_segmented_attention_with_candidate_sketches_and_diagnostics(
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
    const array& key_candidate_sketch,
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
      nullptr,
      &key_candidate_sketch,
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
      mode_ == a_other.mode_ &&
      sparse_v_threshold_ == a_other.sparse_v_threshold_ &&
      output_diagnostics_ == a_other.output_diagnostics_;
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

bool QuantizeAppendKV::is_equivalent(const Primitive& other) const {
  const QuantizeAppendKV& p_other =
      static_cast<const QuantizeAppendKV&>(other);
  return (
      p_other.seq_offset_ == seq_offset_ && p_other.steps_ == steps_ &&
      p_other.key_group_size_ == key_group_size_ &&
      p_other.key_bits_ == key_bits_ &&
      p_other.value_group_size_ == value_group_size_ &&
      p_other.value_bits_ == value_bits_);
}

std::vector<Shape> QuantizeAppendKV::output_shapes(
    const std::vector<array>& inputs) {
  // The six updated planes keep the shapes of the six input planes
  // (inputs[2..7]); inputs[0]/inputs[1] are the incoming k_new/v_new rows.
  return {
      inputs[2].shape(),
      inputs[3].shape(),
      inputs[4].shape(),
      inputs[5].shape(),
      inputs[6].shape(),
      inputs[7].shape()};
}

bool ConvertFP8::is_equivalent(const Primitive& other) const {
  const ConvertFP8& a_other = static_cast<const ConvertFP8&>(other);
  return to_fp8_ == a_other.to_fp8_;
}

} // namespace mlx::core::fast
