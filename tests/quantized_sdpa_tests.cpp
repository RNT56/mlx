// Copyright © 2026 Apple Inc.

#include <optional>

#include "doctest/doctest.h"

#include "mlx/fast.h"
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

bool quantized_sdpa_uses_fallback(
    int qsl,
    int query_heads,
    int kv_heads,
    int head_dim) {
  auto q = zeros({1, query_heads, qsl, head_dim}, float16);
  auto k = zeros({1, kv_heads, 128, head_dim / 8}, uint32);
  return fast::QuantizedScaledDotProductAttention::use_fallback(
      q, k, /* is_training = */ false, Stream(0, Device::gpu));
}

array make_mixed_k8v4_sdpa(
    int qsl,
    int query_heads,
    int kv_heads,
    int key_sequence_length,
    int head_dim,
    int key_group_size = 64,
    int key_bits = 8,
    int value_group_size = 32,
    int value_bits = 4,
    Stream s = Stream(0, Device::gpu)) {
  auto q = zeros({1, query_heads, qsl, head_dim}, float16);
  auto k = zeros({1, kv_heads, key_sequence_length, head_dim * key_bits / 32},
                 uint32);
  auto k_scales =
      zeros({1, kv_heads, key_sequence_length, head_dim / key_group_size},
            float16);
  auto k_biases = zeros(k_scales.shape(), float16);
  auto v =
      zeros({1, kv_heads, key_sequence_length, head_dim * value_bits / 32},
            uint32);
  auto v_scales =
      zeros({1, kv_heads, key_sequence_length, head_dim / value_group_size},
            float16);
  auto v_biases = zeros(v_scales.shape(), float16);

  return fast::mixed_quantized_scaled_dot_product_attention(
      q,
      k,
      k_scales,
      k_biases,
      v,
      v_scales,
      v_biases,
      1.0f,
      std::nullopt,
      std::nullopt,
      key_group_size,
      key_bits,
      value_group_size,
      value_bits,
      false,
      s);
}

} // namespace

TEST_CASE("quantized sdpa supports verifier batch shapes") {
  for (int qsl : {1, 8, 9, 16, 32}) {
    for (int head_dim : {64, 128, 256, 512}) {
      CHECK_FALSE(quantized_sdpa_uses_fallback(qsl, /* gqa = */ 8, head_dim));
    }
  }
}

TEST_CASE("quantized sdpa supports affine int4 decode gqa4 head256") {
  CHECK_FALSE(quantized_sdpa_uses_fallback(
      /* qsl = */ 1, /* query_heads = */ 16, /* kv_heads = */ 4, 256));
}

TEST_CASE("quantized sdpa fallback rejects unsupported gate shapes") {
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 33, /* gqa = */ 8, 128));
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 9, /* gqa = */ 8, 96));
  CHECK(quantized_sdpa_uses_fallback(/* qsl = */ 9, /* gqa = */ 33, 128));
}

TEST_CASE("mixed K8/V4 native sdpa rejects unsupported native shapes") {
  CHECK_THROWS_AS(
      make_mixed_k8v4_sdpa(
          /* qsl = */ 33,
          /* query_heads = */ 1,
          /* kv_heads = */ 1,
          /* key_sequence_length = */ 64,
          /* head_dim = */ 64),
      std::invalid_argument);
  CHECK_THROWS_AS(
      make_mixed_k8v4_sdpa(
          /* qsl = */ 1,
          /* query_heads = */ 33,
          /* kv_heads = */ 1,
          /* key_sequence_length = */ 64,
          /* head_dim = */ 64),
      std::invalid_argument);
  CHECK_THROWS_AS(
      make_mixed_k8v4_sdpa(
          /* qsl = */ 1,
          /* query_heads = */ 1,
          /* kv_heads = */ 1,
          /* key_sequence_length = */ 64,
          /* head_dim = */ 96),
      std::invalid_argument);
}

TEST_CASE("mixed K8/V4 native sdpa rejects non-K8V4 bit widths") {
  CHECK_THROWS_AS(
      make_mixed_k8v4_sdpa(
          /* qsl = */ 1,
          /* query_heads = */ 1,
          /* kv_heads = */ 1,
          /* key_sequence_length = */ 64,
          /* head_dim = */ 64,
          /* key_group_size = */ 64,
          /* key_bits = */ 4,
          /* value_group_size = */ 64,
          /* value_bits = */ 4),
      std::invalid_argument);
}
