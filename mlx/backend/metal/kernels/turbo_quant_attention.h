// Copyright © 2026 RNT56.

#pragma once

#include "mlx/backend/metal/kernels/defines.h"

template <typename T>
[[kernel]] void turbo_quant_attention_unavailable(
    const device T* q [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  (void)q;
  (void)out;
  (void)index;
}
