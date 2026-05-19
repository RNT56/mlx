// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

bool quantized_sdpa_uses_fallback(int qsl, int gqa, int head_dim) {
  auto q = zeros({1, gqa, qsl, head_dim}, float16);
  auto k = zeros({1, 1, 128, head_dim / 8}, uint32);
  return fast::QuantizedScaledDotProductAttention::use_fallback(
      q, k, /* is_training = */ false, Stream(0, Device::gpu));
}

} // namespace

TEST_CASE("quantized sdpa supports verifier batch shapes") {
  for (int qsl : {1, 8, 9, 16, 32}) {
    for (int head_dim : {64, 128, 256, 512}) {
      CHECK_FALSE(quantized_sdpa_uses_fallback(qsl, /* gqa = */ 8, head_dim));
    }
  }
}

TEST_CASE("quantized sdpa fallback rejects unsupported gate shapes") {
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 33, /* gqa = */ 8, 128));
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 9, /* gqa = */ 8, 96));
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 9, /* gqa = */ 33, 128));
}
