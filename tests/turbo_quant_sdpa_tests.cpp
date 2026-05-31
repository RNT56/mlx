// Copyright © 2026 RNT56.

#include "doctest/doctest.h"

#include "mlx/fast.h"
#include "mlx/ops.h"

using namespace mlx::core;

namespace {

struct TurboQuantTestInputs {
  array queries = zeros({1, 4, 1, 64}, float32);
  array key_packed = zeros({1, 1, 1, 1, 8}, uint32);
  array key_signs = zeros({1, 1, 1, 1, 2}, uint32);
  array key_high_mask = zeros({1, 1, 1, 1, 2}, uint32);
  array compact = zeros({1}, uint32);
  array key_scales = zeros({1, 1, 1, 1, 3}, float32);
  array value_packed = zeros({1, 1, 1, 1, 8}, uint32);
  array value_scales = zeros({1, 1, 1, 1, 2}, float32);
  fast::TurboQuantAttentionLayoutDescriptor layout{
      6, 1, 1, 1, 1, 0, 0, 64, 1, 8, 2};
  fast::TurboQuantPrecisionPolicyDescriptor precision{
      35, 64, 3, 4, 500, 1000, 4, 3, 2, 8, 1, 2};
  fast::TurboQuantAttentionOptions options{0.125f, true, 0, 0.0f, false, 3};
};

array call_turbo_quant_attention(TurboQuantTestInputs& inputs) {
  return fast::turbo_quant_segmented_attention(
      inputs.queries,
      inputs.key_packed,
      inputs.key_signs,
      inputs.key_high_mask,
      inputs.compact,
      inputs.key_scales,
      inputs.value_packed,
      inputs.compact,
      inputs.compact,
      inputs.compact,
      inputs.value_scales,
      inputs.layout,
      inputs.precision,
      inputs.options,
      Device::cpu);
}

} // namespace

TEST_CASE("turbo quant segmented attention reports no production native backend") {
  CHECK_EQ(
      fast::turbo_quant_segmented_attention_backend(false, Device::cpu),
      fast::TurboQuantSegmentedAttentionBackend::Unavailable);
  CHECK_FALSE(
      fast::turbo_quant_segmented_attention_is_available(false, Device::cpu));
}

TEST_CASE("turbo quant native attention validates query rank and q length") {
  auto inputs = TurboQuantTestInputs{};
  inputs.queries = zeros({1, 4, 9, 64}, float32);
  CHECK_THROWS_AS(call_turbo_quant_attention(inputs), std::invalid_argument);

  inputs = TurboQuantTestInputs{};
  inputs.queries = zeros({1, 4, 64}, float32);
  CHECK_THROWS_AS(call_turbo_quant_attention(inputs), std::invalid_argument);
}

TEST_CASE("turbo quant native attention validates compressed plane shapes") {
  auto inputs = TurboQuantTestInputs{};
  inputs.key_packed = zeros({1, 1, 1, 1, 7}, uint32);
  CHECK_THROWS_AS(call_turbo_quant_attention(inputs), std::invalid_argument);
}

TEST_CASE("turbo quant native attention fails typed when native backend is unavailable") {
  auto inputs = TurboQuantTestInputs{};
  CHECK_THROWS_AS(
      call_turbo_quant_attention(inputs),
      fast::TurboQuantNativeAttentionUnavailable);
}
