// Copyright © 2026 RNT56.

#include "mlx/backend/metal/kernels/turbo_quant_attention.h"

template [[host_name("turbo_quant_attention_unavailable_float")]] [[kernel]]
void turbo_quant_attention_unavailable<float>(
    const device float* q,
    device float* out,
    uint index);

template [[host_name("turbo_quant_attention_unavailable_float16")]] [[kernel]]
void turbo_quant_attention_unavailable<half>(
    const device half* q,
    device half* out,
    uint index);
