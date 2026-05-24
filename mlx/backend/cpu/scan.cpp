// Copyright © 2023 Apple Inc.

#include <cassert>
#include <type_traits>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

struct ScanSum {
  static constexpr bool vectorizable = true;

  template <typename U, typename T>
  auto operator()(U y, T x) const {
    return y + x;
  }

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) const {
    return y + x;
  }
};

struct ScanProd {
  static constexpr bool vectorizable = true;

  template <typename U, typename T>
  auto operator()(U y, T x) const {
    return y * x;
  }

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) const {
    return y * x;
  }
};

struct ScanMin {
  static constexpr bool vectorizable = true;

  template <typename U, typename T>
  auto operator()(U y, T x) const {
    return x < y ? x : y;
  }

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) const {
    return simd::select(x < y, x, y);
  }
};

struct ScanMax {
  static constexpr bool vectorizable = true;

  template <typename U, typename T>
  auto operator()(U y, T x) const {
    return x < y ? y : x;
  }

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) const {
    return simd::select(x < y, y, x);
  }
};

struct ScanLogAddExp {
  static constexpr bool vectorizable = false;

  template <typename U, typename T>
  auto operator()(U y, T x) const {
    return detail::LogAddExp{}(y, static_cast<U>(x));
  }

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) const {
    return detail::LogAddExp{}(y, x);
  }
};

template <typename T, typename U, typename Op>
void strided_scan_step(
    const U* previous,
    const T* input,
    U* output,
    int stride,
    const Op& op) {
  if constexpr (
      Op::vectorizable && std::is_same_v<T, U> &&
      !std::is_same_v<U, bool>) {
    constexpr int N = simd::max_size<U>;
    while (stride >= N) {
      simd::store(
          output,
          op(simd::load<U, N>(previous), simd::load<T, N>(input)));
      previous += N;
      input += N;
      output += N;
      stride -= N;
    }
  }
  while (stride-- > 0) {
    *output++ = op(*previous++, *input++);
  }
}

template <typename T, typename U, typename Op>
void contiguous_scan(
    const T* input,
    U* output,
    int count,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  if (!reverse) {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        *output = *input;
        for (int j = 1; j < stride; j++) {
          input++;
          output++;
          *output = op(*(output - 1), *input);
        }
        output++;
        input++;
      }
    } else {
      for (int i = 0; i < count; i++) {
        *output = init;
        for (int j = 1; j < stride; j++) {
          *(output + 1) = op(*output, *input);
          input++;
          output++;
        }
        output++;
        input++;
      }
    }
  } else {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        output += stride - 1;
        input += stride - 1;
        *output = *input;
        for (int j = 1; j < stride; j++) {
          input--;
          output--;
          *output = op(*(output + 1), *input);
        }
        output += stride;
        input += stride;
      }
    } else {
      for (int i = 0; i < count; i++) {
        output += stride - 1;
        input += stride - 1;
        *output = init;
        for (int j = 1; j < stride; j++) {
          *(output - 1) = op(*output, *input);
          input--;
          output--;
        }
        output += stride;
        input += stride;
      }
    }
  }
};

template <typename T, typename U, typename Op>
void strided_scan(
    const T* input,
    U* output,
    int count,
    int size,
    int stride,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  if (!reverse) {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        std::copy(input, input + stride, output);
        output += stride;
        input += stride;
        for (int j = 1; j < size; j++) {
          strided_scan_step(output - stride, input, output, stride, op);
          output += stride;
          input += stride;
        }
      }
    } else {
      for (int i = 0; i < count; i++) {
        std::fill(output, output + stride, init);
        output += stride;
        input += stride;
        for (int j = 1; j < size; j++) {
          strided_scan_step(output - stride, input - stride, output, stride, op);
          output += stride;
          input += stride;
        }
      }
    }
  } else {
    if (inclusive) {
      for (int i = 0; i < count; i++) {
        auto input_block = input + (size - 1) * stride;
        auto output_block = output + (size - 1) * stride;
        std::copy(input_block, input_block + stride, output_block);
        for (int j = 1; j < size; j++) {
          input_block -= stride;
          output_block -= stride;
          strided_scan_step(
              output_block + stride, input_block, output_block, stride, op);
        }
        output += size * stride;
        input += size * stride;
      }
    } else {
      for (int i = 0; i < count; i++) {
        auto input_block = input + (size - 1) * stride;
        auto output_block = output + (size - 1) * stride;
        std::fill(output_block, output_block + stride, init);
        for (int j = 1; j < size; j++) {
          input_block -= stride;
          output_block -= stride;
          strided_scan_step(
              output_block + stride,
              input_block + stride,
              output_block,
              stride,
              op);
        }
        output += size * stride;
        input += size * stride;
      }
    }
  }
};

template <typename T, typename U, typename Op>
void scan_op(
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive,
    const Op& op,
    U init) {
  if (in.flags().row_contiguous) {
    if (in.strides()[axis] == 1) {
      contiguous_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis),
          in.shape(axis),
          reverse,
          inclusive,
          op,
          init);
    } else {
      strided_scan(
          in.data<T>(),
          out.data<U>(),
          in.size() / in.shape(axis) / in.strides()[axis],
          in.shape(axis),
          in.strides()[axis],
          reverse,
          inclusive,
          op,
          init);
    }
  } else {
    throw std::runtime_error("Scan op supports only contiguous inputs");
  }
}

template <typename T, typename U>
void scan_dispatch(
    Scan::ReduceType rtype,
    const array& in,
    array& out,
    int axis,
    bool reverse,
    bool inclusive) {
  switch (rtype) {
    case Scan::Sum: {
      auto init = static_cast<U>(0);
      scan_op<T, U>(in, out, axis, reverse, inclusive, ScanSum{}, init);
      break;
    }
    case Scan::Prod: {
      auto init = static_cast<U>(1);
      scan_op<T, U>(in, out, axis, reverse, inclusive, ScanProd{}, init);
      break;
    }
    case Scan::Min: {
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::max();
      scan_op<T, U>(in, out, axis, reverse, inclusive, ScanMin{}, init);
      break;
    }
    case Scan::Max: {
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::min();
      scan_op<T, U>(in, out, axis, reverse, inclusive, ScanMax{}, init);
      break;
    }
    case Scan::LogAddExp: {
      auto init = (issubdtype(in.dtype(), floating))
          ? static_cast<U>(-std::numeric_limits<float>::infinity())
          : std::numeric_limits<U>::min();
      scan_op<T, U>(in, out, axis, reverse, inclusive, ScanLogAddExp{}, init);
      break;
    }
  }
}

} // namespace

void Scan::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& encoder = cpu::get_command_encoder(stream());

  // Ensure contiguity
  auto in = inputs[0];
  if (!in.flags().row_contiguous) {
    in = contiguous_copy_cpu(in, stream());
    encoder.add_temporary(in);
  }
  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    axis_ = axis_,
                    reduce_type_ = reduce_type_,
                    reverse_ = reverse_,
                    inclusive_ = inclusive_]() mutable {
    switch (in.dtype()) {
      case bool_: {
        // We could do a full dtype x dtype switch but this is the only case
        // where we accumulate in a different type, for now.
        //
        // TODO: If we add the option to accumulate floats in higher precision
        //       floats perhaps we should add the full all-to-all dispatch.
        if (reduce_type_ == Scan::Sum && out.dtype() == int32) {
          scan_dispatch<bool, int32_t>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        } else {
          scan_dispatch<bool, bool>(
              reduce_type_, in, out, axis_, reverse_, inclusive_);
        }
        break;
      }
      case uint8:
        scan_dispatch<uint8_t, uint8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint16:
        scan_dispatch<uint16_t, uint16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint32:
        scan_dispatch<uint32_t, uint32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case uint64:
        scan_dispatch<uint64_t, uint64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int8:
        scan_dispatch<int8_t, int8_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int16:
        scan_dispatch<int16_t, int16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int32:
        scan_dispatch<int32_t, int32_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case int64:
        scan_dispatch<int64_t, int64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float16:
        scan_dispatch<float16_t, float16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float32:
        scan_dispatch<float, float>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case float64:
        scan_dispatch<double, double>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case bfloat16:
        scan_dispatch<bfloat16_t, bfloat16_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
      case complex64:
        scan_dispatch<complex64_t, complex64_t>(
            reduce_type_, in, out, axis_, reverse_, inclusive_);
        break;
    }
  });
}

} // namespace mlx::core
