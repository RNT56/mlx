// Copyright © 2026 Apple Inc.

#pragma once

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void general_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& reductions [[buffer(2)]],
    const constant int* shape [[buffer(3)]],
    const constant int64_t* strides [[buffer(4)]],
    const constant int& ndim [[buffer(5)]],
    const constant int* reduce_shape [[buffer(6)]],
    const constant int64_t* reduce_strides [[buffer(7)]],
    const constant int& reduce_ndim [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]]) {
  Op op;
  U total = Op::init;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  loop.next(simd_lane_id, reduce_shape, reduce_strides);
  for (IdxT r = simd_lane_id; r < IdxT(reductions); r += simd_size) {
    total = op(static_cast<U>(in[loop.location()]), total);
    loop.next(simd_size, reduce_shape, reduce_strides);
  }

  total = op.simd_reduce(total);

  if (simd_lane_id == 0) {
    out[out_idx] = total;
  }
}
