// Copyright © 2026 RNT56.
// Generated from mlx-swift/Source/MLX/TurboQuant.swift.
// Keep this file mechanically synchronized when the Swift TurboQuant Metal kernels change.

#pragma once

#include <string_view>

namespace mlx::core::fast::turbo_quant_detail {

inline constexpr std::string_view turbo_quant_attention_header = R"TQMLX(        inline ulong tq_mix(ulong seed, uint index) {
            ulong mixed = seed + ulong(index) * 0x9E3779B97F4A7C15ul;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9ul;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBul;
            mixed ^= mixed >> 31;
            return mixed;
        }

        inline bool tq_random_sign(ulong seed, uint index) {
            return (tq_mix(seed, index) & 1ul) != 0ul;
        }

        inline ulong tq_mix_index(ulong seed, ulong index) {
            ulong mixed = seed + index * 0x9E3779B97F4A7C15ul;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9ul;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBul;
            mixed ^= mixed >> 31;
            return mixed;
        }

        inline bool tq_random_sign_index(ulong seed, ulong index) {
            return (tq_mix_index(seed, index) & 1ul) != 0ul;
        }

        inline ulong tq_make_seed(uint word3, uint word2, uint word1, uint word0) {
            return (ulong(word3) << 48)
                | (ulong(word2) << 32)
                | (ulong(word1) << 16)
                | ulong(word0);
        }

        inline ulong tq_product_channel_rank(ulong seed, uint group_index, uint local_index) {
            ulong state = seed;
            state ^= ulong(group_index) * 0x9E3779B97F4A7C15ul;
            state += ulong(local_index) * 0xD1B54A32D192ED03ul;
            state ^= state >> 30;
            state *= 0xBF58476D1CE4E5B9ul;
            state ^= state >> 27;
            state *= 0x94D049BB133111EBul;
            state ^= state >> 31;
            return state;
        }

        inline bool tq_product_high_precision(
            ulong seed,
            uint group_index,
            uint local,
            uint count,
            uint high_count
        ) {
            if (high_count == 0u) {
                return false;
            }
            if (high_count >= count) {
                return true;
            }
            ulong local_rank = tq_product_channel_rank(seed, group_index, local);
            uint rank = 0u;
            for (uint other = 0u; other < count; other++) {
                ulong other_rank = tq_product_channel_rank(seed, group_index, other);
                if (other_rank < local_rank || (other_rank == local_rank && other < local)) {
                    rank += 1u;
                }
            }
            return rank < high_count;
        }

        inline bool tq_split_high_precision(uint local, uint high_count) {
            return local < high_count;
        }

        inline uint tq_high_precision_count(uint count, uint numerator, uint denominator) {
            if (denominator == 0u) {
                return 0u;
            }
            return uint(round(float(count * numerator) / float(denominator)));
        }

        inline float tq_codebook_unit(uint bits, uint code) {
            if (bits <= 1u) {
                return code == 0u ? -0.797884561f : 0.797884561f;
            }
            if (bits == 2u) {
                switch (min(code, 3u)) {
                case 0u: return -1.510499245f;
                case 1u: return -0.452819573f;
                case 2u: return 0.452819573f;
                default: return 1.510499245f;
                }
            }
            if (bits == 3u) {
                switch (min(code, 7u)) {
                case 0u: return -2.175028018f;
                case 1u: return -1.367204388f;
                case 2u: return -0.773020220f;
                case 3u: return -0.251312159f;
                case 4u: return 0.251312159f;
                case 5u: return 0.773020220f;
                case 6u: return 1.367204388f;
                default: return 2.175028018f;
                }
            }
            if (bits == 5u) {
                uint clamped = min(code, 31u);
                uint magnitude_index = clamped < 16u ? clamped : 31u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.167510584f; break;
                case 1u: magnitude = 2.601080629f; break;
                case 2u: magnitude = 2.248054067f; break;
                case 3u: magnitude = 1.990376987f; break;
                case 4u: magnitude = 1.784481424f; break;
                case 5u: magnitude = 1.607119170f; break;
                case 6u: magnitude = 1.444524024f; break;
                case 7u: magnitude = 1.288831640f; break;
                case 8u: magnitude = 1.135990256f; break;
                case 9u: magnitude = 0.984174410f; break;
                case 10u: magnitude = 0.832676140f; break;
                case 11u: magnitude = 0.681261776f; break;
                case 12u: magnitude = 0.529866428f; break;
                case 13u: magnitude = 0.378475081f; break;
                case 14u: magnitude = 0.227084777f; break;
                default: magnitude = 0.075694884f; break;
                }
                return clamped < 16u ? -magnitude : magnitude;
            }
            if (bits == 6u) {
                uint clamped = min(code, 63u);
                uint magnitude_index = clamped < 32u ? clamped : 63u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.370567258f; break;
                case 1u: magnitude = 2.846634435f; break;
                case 2u: magnitude = 2.539498403f; break;
                case 3u: magnitude = 2.334801410f; break;
                case 4u: magnitude = 2.189068534f; break;
                case 5u: magnitude = 2.077692738f; break;
                case 6u: magnitude = 1.985038395f; break;
                case 7u: magnitude = 1.901543224f; break;
                case 8u: magnitude = 1.821977755f; break;
                case 9u: magnitude = 1.743867835f; break;
                case 10u: magnitude = 1.666217206f; break;
                case 11u: magnitude = 1.588688278f; break;
                case 12u: magnitude = 1.511185949f; break;
                case 13u: magnitude = 1.433688315f; break;
                case 14u: magnitude = 1.356191352f; break;
                case 15u: magnitude = 1.278694490f; break;
                case 16u: magnitude = 1.201197668f; break;
                case 17u: magnitude = 1.123700882f; break;
                case 18u: magnitude = 1.046204128f; break;
                case 19u: magnitude = 0.968707404f; break;
                case 20u: magnitude = 0.891210709f; break;
                case 21u: magnitude = 0.813714039f; break;
                case 22u: magnitude = 0.736217393f; break;
                case 23u: magnitude = 0.658720768f; break;
                case 24u: magnitude = 0.581224162f; break;
                case 25u: magnitude = 0.503727573f; break;
                case 26u: magnitude = 0.426230999f; break;
                case 27u: magnitude = 0.348734437f; break;
                case 28u: magnitude = 0.271237885f; break;
                case 29u: magnitude = 0.193741341f; break;
                case 30u: magnitude = 0.116244802f; break;
                default: magnitude = 0.038748267f; break;
                }
                return clamped < 32u ? -magnitude : magnitude;
            }
            if (bits >= 7u) {
                uint clamped = min(code, 127u);
                uint magnitude_index = clamped < 64u ? clamped : 127u - clamped;
                float magnitude = 0.0f;
                switch (magnitude_index) {
                case 0u: magnitude = 3.471692079f; break;
                case 1u: magnitude = 2.967922351f; break;
                case 2u: magnitude = 2.682760472f; break;
                case 3u: magnitude = 2.503778860f; break;
                case 4u: magnitude = 2.387735667f; break;
                case 5u: magnitude = 2.309487569f; break;
                case 6u: magnitude = 2.252475440f; break;
                case 7u: magnitude = 2.206174364f; break;
                case 8u: magnitude = 2.164605881f; break;
                case 9u: magnitude = 2.124839178f; break;
                case 10u: magnitude = 2.085655475f; break;
                case 11u: magnitude = 2.046629763f; break;
                case 12u: magnitude = 2.007639247f; break;
                case 13u: magnitude = 1.968655011f; break;
                case 14u: magnitude = 1.929671639f; break;
                case 15u: magnitude = 1.890688353f; break;
                case 16u: magnitude = 1.851705074f; break;
                case 17u: magnitude = 1.812721796f; break;
                case 18u: magnitude = 1.773738519f; break;
                case 19u: magnitude = 1.734755243f; break;
                case 20u: magnitude = 1.695771967f; break;
                case 21u: magnitude = 1.656788693f; break;
                case 22u: magnitude = 1.617805419f; break;
                case 23u: magnitude = 1.578822145f; break;
                case 24u: magnitude = 1.539838873f; break;
                case 25u: magnitude = 1.500855601f; break;
                case 26u: magnitude = 1.461872330f; break;
                case 27u: magnitude = 1.422889060f; break;
                case 28u: magnitude = 1.383905790f; break;
                case 29u: magnitude = 1.344922521f; break;
                case 30u: magnitude = 1.305939253f; break;
                case 31u: magnitude = 1.266955985f; break;
                case 32u: magnitude = 1.227972718f; break;
                case 33u: magnitude = 1.188989451f; break;
                case 34u: magnitude = 1.150006185f; break;
                case 35u: magnitude = 1.111022919f; break;
                case 36u: magnitude = 1.072039654f; break;
                case 37u: magnitude = 1.033056390f; break;
                case 38u: magnitude = 0.994073126f; break;
                case 39u: magnitude = 0.955089862f; break;
                case 40u: magnitude = 0.916106599f; break;
                case 41u: magnitude = 0.877123336f; break;
                case 42u: magnitude = 0.838140074f; break;
                case 43u: magnitude = 0.799156812f; break;
                case 44u: magnitude = 0.760173551f; break;
                case 45u: magnitude = 0.721190290f; break;
                case 46u: magnitude = 0.682207029f; break;
                case 47u: magnitude = 0.643223768f; break;
                case 48u: magnitude = 0.604240508f; break;
                case 49u: magnitude = 0.565257248f; break;
                case 50u: magnitude = 0.526273989f; break;
                case 51u: magnitude = 0.487290729f; break;
                case 52u: magnitude = 0.448307470f; break;
                case 53u: magnitude = 0.409324211f; break;
                case 54u: magnitude = 0.370340952f; break;
                case 55u: magnitude = 0.331357694f; break;
                case 56u: magnitude = 0.292374435f; break;
                case 57u: magnitude = 0.253391177f; break;
                case 58u: magnitude = 0.214407919f; break;
                case 59u: magnitude = 0.175424661f; break;
                case 60u: magnitude = 0.136441403f; break;
                case 61u: magnitude = 0.097458145f; break;
                case 62u: magnitude = 0.058474887f; break;
                default: magnitude = 0.019491629f; break;
                }
                return clamped < 64u ? -magnitude : magnitude;
            }
            switch (min(code, 15u)) {
            case 0u: return -2.778927695f;
            case 1u: return -2.124836923f;
            case 2u: return -1.680512470f;
            case 3u: return -1.321175453f;
            case 4u: return -1.003692455f;
            case 5u: return -0.707453186f;
            case 6u: return -0.421537889f;
            case 7u: return -0.140103661f;
            case 8u: return 0.140103661f;
            case 9u: return 0.421537889f;
            case 10u: return 0.707453186f;
            case 11u: return 1.003692455f;
            case 12u: return 1.321175453f;
            case 13u: return 1.680512470f;
            case 14u: return 2.124836923f;
            default: return 2.778927695f;
            }
        }

        inline float tq_codebook_level(uint bits, uint code, uint count) {
            return tq_codebook_unit(bits, code) * rsqrt(float(max(count, 1u)));
        }

        inline uint tq_nearest_codebook_index(float value, uint bits, uint count) {
            uint level_count = 1u << bits;
            uint low = 0u;
            uint high = level_count - 1u;
            while (low < high) {
                uint mid = (low + high) >> 1u;
                float boundary =
                    0.5f * (tq_codebook_level(bits, mid, count)
                        + tq_codebook_level(bits, mid + 1u, count));
                if (value <= boundary) {
                    high = mid;
                } else {
                    low = mid + 1u;
                }
            }
            return low;
        }

        inline void tq_fast_hadamard(thread float* values, uint count) {
            for (uint width = 1u; width < count; width <<= 1u) {
                for (uint start = 0u; start < count; start += width << 1u) {
                    for (uint offset = 0u; offset < width; offset++) {
                        float lhs = values[start + offset];
                        float rhs = values[start + offset + width];
                        values[start + offset] = lhs + rhs;
                        values[start + offset + width] = lhs - rhs;
                    }
                }
            }
        }

        inline void tq_apply_rotation_signs(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index
        ) {
            for (uint local = 0u; local < count; local++) {
                ulong sign_index = ulong(group_index) * 4099ul + ulong(local);
                if (tq_random_sign_index(seed, sign_index)) {
                    values[local] = -values[local];
                }
            }
        }

        inline void tq_apply_givens_pass(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            uint pass,
            float direction
        ) {
            uint offset = pass & 1u;
            for (uint index = offset; index + 1u < count; index += 2u) {
                ulong angle_rank = tq_product_channel_rank(
                    seed ^ (ulong(pass) * 0xA24BAED4963EE407ul),
                    group_index,
                    index >> 1u);
                float unit = float(uint(angle_rank)) / 4294967295.0f;
                float angle = (unit - 0.5f) * 3.14159265358979323846f * direction;
                float c = cos(angle);
                float s = sin(angle);
                float lhs = values[index];
                float rhs = values[index + 1u];
                values[index] = c * lhs - s * rhs;
                values[index + 1u] = s * lhs + c * rhs;
            }
        }

        inline void tq_apply_product_rotation(
            thread float* values,
            uint count,
            ulong seed,
            uint group_index,
            bool inverse
        ) {
            if (count <= 1u) {
                tq_apply_rotation_signs(values, count, seed, group_index);
                return;
            }
            if ((count & (count - 1u)) == 0u) {
                if (inverse) {
                    tq_fast_hadamard(values, count);
                    tq_apply_rotation_signs(values, count, seed, group_index);
                } else {
                    tq_apply_rotation_signs(values, count, seed, group_index);
                    tq_fast_hadamard(values, count);
                }
                float scale = rsqrt(float(count));
                for (uint local = 0u; local < count; local++) {
                    values[local] *= scale;
                }
                return;
            }
            if (inverse) {
                for (uint pass_index = 0u; pass_index < 4u; pass_index++) {
                    tq_apply_givens_pass(values, count, seed, group_index, 3u - pass_index, -1.0f);
                }
            } else {
                for (uint pass = 0u; pass < 4u; pass++) {
                    tq_apply_givens_pass(values, count, seed, group_index, pass, 1.0f);
                }
            }
        }

        inline uint tq_bitset_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group
        ) {
            return (((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * bitset_words_per_group + word;
        }

        inline uint tq_packed_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint word,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            return (((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * mag_words_per_group + word;
        }

        template <typename PackedPtr>
        inline uint tq_read_packed_unsigned(
            PackedPtr packed,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint first = packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] >> packed_bit;
            if (packed_bit + bits > 32u) {
                uint next = packed[tq_packed_offset(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)];
                first |= next << (32u - packed_bit);
            }
            return first & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline uint tq_read_aligned_affine_unsigned(
            PackedPtr packed,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint bit_offset = local * bits;
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint word = packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)];
            return (word >> packed_bit) & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline void tq_write_packed_unsigned(
            PackedPtr packed,
            uint quantized,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint bit_offset,
            uint bits,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group
        ) {
            uint packed_word = bit_offset >> 5;
            uint packed_bit = bit_offset & 31u;
            uint mask = ((1u << bits) - 1u);
            uint value = quantized & mask;
            packed[tq_packed_offset(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                value << packed_bit;
            if (packed_bit + bits > 32u) {
                packed[tq_packed_offset(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                    value >> (32u - packed_bit);
            }
        }

        inline uint tq_scale_offset(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint scale_index,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            return ((((batch * kv_heads + head) * capacity + token)
                * groups_per_vector + group) * 3u) + scale_index;
        }

        inline uint tq_physical_token(
            uint logical_token,
            uint capacity,
            uint ring_offset,
            uint pinned_prefix_length
        ) {
            uint pinned = pinned_prefix_length;
            if (logical_token < pinned) {
                return logical_token;
            }
            uint ring_capacity = capacity - pinned;
            if (ring_capacity == 0u) {
                return min(logical_token, capacity - 1u);
            }
            uint ring_logical = logical_token - pinned;
            return pinned + ((ring_offset + ring_logical) % ring_capacity);
        }

        template <typename HighMaskPtr>
        inline uint tq_attention_high_count_before(
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group
        ) {
            uint full_words = local >> 5;
            uint count = 0u;
            for (uint word = 0u; word < full_words; word++) {
                count += popcount(high_mask[tq_bitset_offset(
                    batch, head, token, group, word,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group)]);
            }
            uint remainder = local & 31u;
            if (remainder > 0u && full_words < bitset_words_per_group) {
                uint mask = (1u << remainder) - 1u;
                count += popcount(high_mask[tq_bitset_offset(
                    batch, head, token, group, full_words,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group)] & mask);
            }
            return count;
        }

        template <typename HighMaskPtr>
        inline uint tq_attention_magnitude_bit_offset(
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits,
            thread uint* bits_out
        ) {
            uint bits = base_bits;
            uint bit_offset = local * base_bits;
            if (high_bits > base_bits) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                bool high_precision =
                    (high_mask[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)]
                        & (1u << bitset_bit)) != 0u;
                bits = high_precision ? high_bits : base_bits;

                uint high_before = tq_attention_high_count_before(
                    high_mask, batch, head, token, group, local,
                    kv_heads, capacity, groups_per_vector, bitset_words_per_group);
                bit_offset += high_before * (high_bits - base_bits);
            }
            *bits_out = bits;
            return bit_offset;
        }

        template <typename PackedPtr, typename HighMaskPtr>
        inline uint tq_read_magnitude(
            PackedPtr packed,
            HighMaskPtr high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits
        ) {
            uint bits = base_bits;
            uint bit_offset = tq_attention_magnitude_bit_offset(
                high_mask, batch, head, token, group, local,
                kv_heads, capacity, groups_per_vector, bitset_words_per_group,
                base_bits, high_bits, &bits);
            return tq_read_packed_unsigned(
                packed, batch, head, token, group, bit_offset, bits,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
        }

        inline uint tq_storage_group_index(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            // TurboQuant (arXiv:2504.19874) uses a SINGLE shared random rotation, not a
            // per-token one: the data-oblivious rotation only needs to be random, not unique
            // per vector, to hit the distortion bound. Keying the rotation / high-precision
            // seed by (batch, head, group) — and NOT by token/capacity — makes the rotation
            // identical for every key token in a head, so the query can be rotated ONCE per
            // group and reused across all keys instead of being re-rotated per key (the prior
            // behaviour, which cost O(N) query rotations per attention step for no quality
            // gain). Encode and decode both route through this function, so the codec stays
            // self-consistent. `token`/`capacity` are intentionally unused.
            (void)token;
            (void)capacity;
            return (batch * kv_heads + head) * groups_per_vector + group;
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline float tq_decode_attention_value(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            uint batch,
            uint head,
            uint token,
            uint dimension,
            ulong seed,
            uint role,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint base_bits,
            uint high_bits,
            uint value_bits,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            thread float* rotated
        ) {
            uint group = dimension / group_size;
            uint local = dimension - group * group_size;
            if (role == 1u) {
                uint quantized = value_bits == 4u || value_bits == 8u
                    ? tq_read_aligned_affine_unsigned(
                        packed, batch, head, token, group, local, value_bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group)
                    : tq_read_packed_unsigned(
                        packed, batch, head, token, group, local * value_bits, value_bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                uint scale_base = ((((batch * kv_heads + head) * capacity + token)
                    * groups_per_vector + group) * 2u);
                return scales[scale_base + 1u] + float(quantized) * scales[scale_base];
            }

            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint bit_offset = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));
            for (uint decode_local = 0u; decode_local < count; decode_local++) {
                uint bitset_word = decode_local >> 5;
                uint bitset_bit = decode_local & 31u;
                uint bit_mask = 1u << bitset_bit;
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(decode_local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, decode_local * key_base_bits,
                        key_base_bits, kv_heads, capacity, groups_per_vector,
                        mag_words_per_group);
                    if (high_precision) {
                        uint extra_code = tq_read_packed_unsigned(
                            packed, batch, head, token, group,
                            group_size * key_base_bits + decode_local,
                            key_high_bits - key_base_bits,
                            kv_heads, capacity, groups_per_vector, mag_words_per_group);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                rotated[decode_local] = tq_codebook_unit(bits, code) * inv_sqrt_count;
            }
            tq_apply_product_rotation(rotated, count, seed, storage_group, true);
            return rotated[local] * scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline float tq_product_attention_inner_product_group(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            tq_apply_product_rotation(query_values, count, seed, storage_group, false);

            float quantized_dot = 0.0f;
            float sign_dot = 0.0f;
            // TQPROF_OPT3 packed-word caches (base-stream + extra-bit slot), mirroring the pair/quad
            // estimators so the split-magnitude decode avoids per-element packed reloads.
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));
            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                quantized_dot += query_values[local] * tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                sign_dot += qjl_sign * query_values[local];
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count))) * sign_dot;
            return norm * quantized_dot + residual;
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_pair(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint pair_repeats,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);
            uint repeats = min(pair_repeats, 2u);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[2];
            float sign_dot[2];
            quantized_dot[0] = 0.0f;
            quantized_dot[1] = 0.0f;
            sign_dot[0] = 0.0f;
            sign_dot[1] = 0.0f;
            // TQPROF_OPT/OPT2/OPT3 packed-word caches (hoisted base + base-stream slot + extra-bit
            // slot), mirroring the quad estimator so all three decode branches avoid per-element
            // packed reloads.
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < repeats; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < repeats; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_quad(
            PackedPtr packed,
            SignsPtr signs,
            HighMaskPtr high_mask,
            ResidualSignsPtr residual_signs,
            ScalesPtr scales,
            thread float* query_values,
            thread float* scores,
            uint batch,
            uint head,
            uint token,
            uint group,
            ulong seed,
            uint group_size,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector,
            uint mag_words_per_group,
            uint bitset_words_per_group,
            uint key_base_bits,
            uint key_high_bits,
            uint layout_version,
            uint head_dim,
            uint high_count,
            bool query_prerotated
        ) {
            uint group_start = group * group_size;
            uint count = min(group_size, head_dim - group_start);
            uint storage_group = tq_storage_group_index(
                batch, head, token, group, kv_heads, capacity, groups_per_vector);

            if (!query_prerotated) {
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    tq_apply_product_rotation(
                        query_values + repeat * group_size, count, seed, storage_group, false);
                }
            }

            float quantized_dot[4];
            float sign_dot[4];
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                quantized_dot[repeat] = 0.0f;
                sign_dot[repeat] = 0.0f;
            }
            // TQPROF_OPT hoist invariant packed base + cache packed word across uniform codes
            uint tqopt_packed_base = tq_packed_offset(
                batch, head, token, group, 0u,
                kv_heads, capacity, groups_per_vector, mag_words_per_group);
            uint tqopt_cached_idx = 0xffffffffu;
            uint tqopt_cached_val = 0u;
            // Second cache slot for the split-magnitude high-precision (extra-bit) stream, which
            // lives in a separate region of the packed buffer from the base-bits stream.
            uint tqopt_extra_idx = 0xffffffffu;
            uint tqopt_extra_val = 0u;
            uint cached_bitset_word = 0xffffffffu;
            uint cached_sign_bits = 0u;
            uint cached_high_word = 0xffffffffu;
            uint cached_high_bits = 0u;
            uint bit_offset = 0u;
            bool split_magnitude =
                layout_version >= 6u
                && key_high_bits == key_base_bits + 1u
                && key_high_bits > key_base_bits;
            float inv_sqrt_count = rsqrt(float(max(count, 1u)));

            for (uint local = 0u; local < count; local++) {
                uint bitset_word = local >> 5;
                uint bitset_bit = local & 31u;
                uint bit_mask = 1u << bitset_bit;
                if (bitset_word != cached_bitset_word) {
                    cached_bitset_word = bitset_word;
                    cached_sign_bits = signs[tq_bitset_offset(
                        batch, head, token, group, bitset_word,
                        kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                }
                uint bits = key_base_bits;
                uint code = 0u;
                if (split_magnitude) {
                    bool high_precision = tq_split_high_precision(local, high_count);
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT3 split-magnitude fast path: cache the base-bits stream (offset
                    // local*key_base_bits, uniform stride) and the high-precision extra-bit stream
                    // (offset group_size*key_base_bits+local, stride 1) in two slots, instead of the
                    // two per-element tq_read_packed_unsigned reloads. Bit-exact. This is the live
                    // turbo3_5 path (verified by negation litmus).
                    uint base_bo = local * key_base_bits;
                    uint base_pw = base_bo >> 5;
                    uint base_pbit = base_bo & 31u;
                    if (base_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = base_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + base_pw];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + base_pw + 1u] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + extra_pw];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + extra_pw + 1u] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    // TQPROF_OPT2 variable-bit fast path: reuse the same packed-word cache as the
                    // uniform branch. Valid because bit_offset advances monotonically over a single
                    // contiguous magnitude bitstream even though `bits` varies per element, so
                    // consecutive variable-width codes still share 32-bit words. Removes the
                    // per-element packed reload that tq_read_packed_unsigned did (turbo3_5 path).
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + tqopt_pw];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + tqopt_pw + 1u] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                } else {
                    // TQPROF_OPT cached uniform-width packed read (1 load per 32/bits codes)
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + tqopt_pw];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + tqopt_pw + 1u] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                }
                float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                float qjl_sign = (cached_sign_bits & bit_mask) != 0u ? -1.0f : 1.0f;
                for (uint repeat = 0u; repeat < 4u; repeat++) {
                    float query_value = query_values[repeat * group_size + local];
                    quantized_dot[repeat] += query_value * level;
                    sign_dot[repeat] += qjl_sign * query_value;
                }
            }

            float norm = scales[tq_scale_offset(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset(
                batch, head, token, group, 1u, kv_heads, capacity, groups_per_vector)];
            float residual_scale = residual_norm * sqrt(3.14159265358979323846f / (2.0f * float(count)));
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                scores[repeat] += norm * quantized_dot[repeat] + residual_scale * sign_dot[repeat];
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_fused_attention_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            uint physical_token = 0u;
            if (active) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                float score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint group_start = group * uint(GROUP_SIZE);
                    uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                    thread float query_values[GROUP_SIZE];
                    for (uint local = 0u; local < count; local++) {
                        query_values[local] = query_cache[group_start + local];
                    }
                    score += tq_product_attention_inner_product_group(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                }
                scaled_score = score * attention_scale;
            }
            tile_scores[lane] = scaled_score;
            tile_physical_tokens[lane] = physical_token;
            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            if (lane < uint(HEAD_DIM)) {
                output_accum[lane] *= old_scale;
            }

            float weight = active ? exp(tile_scores[lane] - new_row_max) : 0.0f;
            tile_scores[lane] = weight;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float next_row_sum = row_sum * old_scale + partial[0];
            if (lane < uint(HEAD_DIM)) {
                thread float decode_scratch[GROUP_SIZE];
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        float value = tq_decode_attention_value(
                            v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                            batch, kv_head, tile_physical_tokens[tile_lane], lane,
                            value_seed, 1u,
                            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                            uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                            uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                            decode_scratch);
                        dimension_accum += tile_weight * value;
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            row_max = new_row_max;
            row_sum = next_row_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < uint(HEAD_DIM)) {
            float inv_sum = 1.0f / max(row_sum, 1.17549435e-38f);
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
            out[out_index] = static_cast<OUTPUT_DTYPE>(output_accum[lane] * inv_sum);
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_fused_attention_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[256];
        threadgroup uint count_partial[256];
        threadgroup float tile_scores[256];
        threadgroup uint tile_physical_tokens[256];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        float sparse_v_threshold = float(runtime_sparse_v_threshold);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
            output_accum[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            if (active) {
                uint physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                float score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint group_start = group * uint(GROUP_SIZE);
                    uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                    thread float query_values[GROUP_SIZE];
                    for (uint local = 0u; local < count; local++) {
                        query_values[local] = query_cache[group_start + local];
                    }
                    score += tq_product_attention_inner_product_group(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                }
                scaled_score = score * attention_scale;
            }

            partial[lane] = scaled_score;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] = max(partial[lane], partial[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float tile_max = partial[0];
            float new_row_max = max(row_max, tile_max);
            float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
            float weight = active ? exp(scaled_score - new_row_max) : 0.0f;
            partial[lane] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            row_sum = row_sum * old_scale + partial[0];
            row_max = new_row_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint skipped_count = 0u;
        uint total_count = 0u;
        float inv_row_sum = 1.0f / max(row_sum, 1.17549435e-38f);
        for (uint tile_start = 0u; tile_start < logical_length; tile_start += threads_per_row) {
            uint logical_token = tile_start + lane;
            bool active = logical_token < logical_length
                && (!DO_CAUSAL || logical_token <= causal_limit);
            float scaled_score = -INFINITY;
            uint physical_token = 0u;
            if (active) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                float score = 0.0f;
                for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                    uint group_start = group * uint(GROUP_SIZE);
                    uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                    thread float query_values[GROUP_SIZE];
                    for (uint local = 0u; local < count; local++) {
                        query_values[local] = query_cache[group_start + local];
                    }
                    score += tq_product_attention_inner_product_group(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                }
                scaled_score = score * attention_scale;
            }

            float final_weight = active ? exp(scaled_score - row_max) * inv_row_sum : 0.0f;
            bool skipped = active && final_weight < sparse_v_threshold;
            tile_scores[lane] = skipped ? 0.0f : final_weight;
            tile_physical_tokens[lane] = physical_token;
            count_partial[lane] = skipped ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    count_partial[lane] += count_partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                skipped_count += count_partial[0];
            }

            count_partial[lane] = active ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    count_partial[lane] += count_partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                total_count += count_partial[0];
            }

            if (lane < uint(HEAD_DIM)) {
                thread float decode_scratch[GROUP_SIZE];
                float dimension_accum = output_accum[lane];
                for (uint tile_lane = 0u; tile_lane < threads_per_row; tile_lane++) {
                    float tile_weight = tile_scores[tile_lane];
                    if (tile_weight > 0.0f) {
                        float value = tq_decode_attention_value(
                            v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                            batch, kv_head, tile_physical_tokens[tile_lane], lane,
                            value_seed, 1u,
                            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                            uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                            uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                            decode_scratch);
                        dimension_accum += tile_weight * value;
                    }
                }
                output_accum[lane] = dimension_accum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < uint(HEAD_DIM)) {
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
            out[out_index] = static_cast<OUTPUT_DTYPE>(output_accum[lane]);
        }
        if (lane == 0u) {
            sparse_stats[row * 2u] = skipped_count;
            sparse_stats[row * 2u + 1u] = total_count;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_diagnostics_source = R"TQMLX(        constexpr uint threads_per_block = 256u;
        uint lane = thread_position_in_threadgroup.x;
        threadgroup uint skipped_partial[256];
        threadgroup uint total_partial[256];

        uint skipped = 0u;
        uint total = 0u;
        for (uint row = lane; row < uint(ROW_COUNT); row += threads_per_block) {
            skipped += sparse_stats[row * 2u];
            total += sparse_stats[row * 2u + 1u];
        }
        skipped_partial[lane] = skipped;
        total_partial[lane] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                skipped_partial[lane] += skipped_partial[lane + stride];
                total_partial[lane] += total_partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            diagnostics[0] = int(BACKEND_VERSION);
            diagnostics[1] = int(KERNEL_KIND);
            diagnostics[2] = int(ACTIVE_BLOCKS);
            diagnostics[3] = int(BLOCK_TOKENS);
            diagnostics[4] = int(skipped_partial[0]);
            diagnostics[5] = int(total_partial[0]);
            diagnostics[6] = int(FALLBACK_CODE);
            diagnostics[7] = int(FLAGS);
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_block_partials_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup float tile_scores[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < uint(HEAD_DIM)) {
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = 0.0f;
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        uint physical_token = 0u;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            float score = 0.0f;
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                thread float query_values[GROUP_SIZE];
                for (uint local = 0u; local < count; local++) {
                    query_values[local] = query_cache[group_start + local];
                }
                score += tq_product_attention_inner_product_group(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                    batch, kv_head, physical_token, group, key_seed,
                    uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                    uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                    uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                    uint(HEAD_DIM),
                    tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
            }
            scaled_score = score * attention_scale;
        }
        tile_scores[lane] = scaled_score;
        tile_physical_tokens[lane] = physical_token;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(tile_scores[lane] - tile_max) : 0.0f;
        tile_scores[lane] = tile_weight;
        partial[lane] = tile_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
            partial_stats[stat_index] = tile_max;
            partial_stats[stat_index + 1u] = partial[0];
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                float weight = tile_scores[tile_lane];
                if (weight > 0.0f) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    dimension_accum += weight * value;
                }
            }
            uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
            partial_out[out_index] = dimension_accum;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_gqa_block_partials_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        // Sized to the actual threadgroup width (was a fixed 4*512 / 512) so threadgroup
        // memory tracks the real block size — at blocks < 512 this frees enough threadgroup
        // memory for multiple threadgroups to be resident per core, raising occupancy.
        threadgroup float partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        constexpr uint repeat_count = uint(GQA_REPEATS) < 4u ? uint(GQA_REPEATS) : 4u;
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = 0.0f;
                }
            }
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                long q_index =
                    long(batch) * q_strides[0]
                    + long(q_head) * q_strides[1]
                    + long(q_token) * q_strides[2]
                    + long(lane) * q_strides[3];
                query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Part B: rotate the query ONCE per (repeat, group) and reuse it across every key
        // token. Because the rotation seed is token-independent (see tq_storage_group_index),
        // the rotation is identical for all keys, so hoisting it here turns the prior O(N)
        // per-key query rotations into O(repeat_count * groups_per_vector) per attention step.
        {
            uint rg_total = repeat_count * uint(GROUPS_PER_VECTOR);
            if (lane < rg_total) {
                uint r = lane / uint(GROUPS_PER_VECTOR);
                uint g = lane % uint(GROUPS_PER_VECTOR);
                uint gs = g * uint(GROUP_SIZE);
                uint cnt = min(uint(GROUP_SIZE), uint(HEAD_DIM) - gs);
                uint rot_seed_index = tq_storage_group_index(
                    batch, kv_head, 0u, g, uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR));
                thread float tmp[GROUP_SIZE];
                for (uint i = 0u; i < cnt; i++) {
                    tmp[i] = query_cache[r * uint(HEAD_DIM) + gs + i];
                }
                tq_apply_product_rotation(tmp, cnt, key_seed, rot_seed_index, false);
                for (uint i = 0u; i < cnt; i++) {
                    query_cache[r * uint(HEAD_DIM) + gs + i] = tmp[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint lanes_per_token = uint(LANES_PER_TOKEN);
        uint physical_token = 0u;
        bool active = false;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

        if (lanes_per_token == 4u) {
            // TQCOOP cooperative quad-per-key coalesced decode (turbo8 uniform path).
            // 4 lanes cooperate on one key, each decoding a contiguous HEAD_DIM/4 chunk
            // (so the quad's 4 chunks tile one cache line -> coalesced, ~4x less L1
            // pressure than the strided 1-thread-per-token mapping). Each quad walks 4
            // tokens in 4 passes; lane j keeps the score of pass j so the existing
            // per-lane back-half (tile_scores[lane], reductions, AV) is unchanged.
            uint lane_in_quad = lane & 3u;
            uint quad_id = lane >> 2u;
            uint num_quads = uint(THREADS_PER_BLOCK) >> 2u;
            uint dims_per_lane = uint(HEAD_DIM) / 4u;
            uint dim_start = lane_in_quad * dims_per_lane;
            uint g = dim_start / uint(GROUP_SIZE);
            uint local_start = dim_start - g * uint(GROUP_SIZE);
            uint count_g = min(uint(GROUP_SIZE), uint(HEAD_DIM) - g * uint(GROUP_SIZE));
            float inv_sqrt_count = rsqrt(float(max(count_g, 1u)));
            float residual_scale_factor =
                sqrt(3.14159265358979323846f / (2.0f * float(count_g)));
            constexpr uint coop_base_bits = uint(KEY_BASE_BITS);
            constexpr uint coop_high_bits = uint(KEY_HIGH_BITS);
            // Coop handles uniform (turbo8/turbo4v2: base==high) AND split-magnitude
            // (turbo3_5: high==base+1 at layout v6 — a base-bits stream + a 1-bit high stream,
            // both per-group contiguous, so each lane's chunk still coalesces). Branch-2
            // variable-bit is dead at v6 and gated out, so these two cases are exhaustive.
            constexpr bool coop_split =
                uint(LAYOUT_VERSION) >= 6u && coop_high_bits == coop_base_bits + 1u;
            uint coop_high_count = coop_split
                ? tq_high_precision_count(
                    count_g, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR))
                : 0u;
            uint my_token = 0u;
            for (uint j = 0u; j < 4u; j++) {
                uint logical_token = block_start + quad_id + j * num_quads;
                bool tok_active = logical_token < logical_length
                    && (!DO_CAUSAL || logical_token <= causal_limit);
                thread float ts[4];
                ts[0] = 0.0f; ts[1] = 0.0f; ts[2] = 0.0f; ts[3] = 0.0f;
                uint phys = 0u;
                if (tok_active) {
                    phys = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    uint base = tq_packed_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP));
                    uint cached_idx = 0xffffffffu;
                    uint cached_val = 0u;
                    uint extra_idx = 0xffffffffu;
                    uint extra_val = 0u;
                    uint cached_bw = 0xffffffffu;
                    uint cached_sign = 0u;
                    thread float qd[4];
                    thread float sd[4];
                    qd[0] = 0.0f; qd[1] = 0.0f; qd[2] = 0.0f; qd[3] = 0.0f;
                    sd[0] = 0.0f; sd[1] = 0.0f; sd[2] = 0.0f; sd[3] = 0.0f;
                    for (uint i = 0u; i < dims_per_lane; i++) {
                        uint local = local_start + i;
                        uint dim = dim_start + i;
                        uint bits;
                        uint code;
                        if (coop_split) {
                            bool hp = local < coop_high_count;
                            bits = hp ? coop_high_bits : coop_base_bits;
                            uint base_bo = local * coop_base_bits;
                            uint base_pw = base_bo >> 5;
                            uint base_pb = base_bo & 31u;
                            if (base_pw != cached_idx) {
                                cached_idx = base_pw;
                                cached_val = k_packed[base + base_pw];
                            }
                            uint base_asm = cached_val >> base_pb;
                            if (base_pb + coop_base_bits > 32u) {
                                base_asm |= k_packed[base + base_pw + 1u] << (32u - base_pb);
                            }
                            code = base_asm & ((1u << coop_base_bits) - 1u);
                            if (hp) {
                                uint extra_bits = coop_high_bits - coop_base_bits;
                                uint extra_bo = uint(GROUP_SIZE) * coop_base_bits + local;
                                uint extra_pw = extra_bo >> 5;
                                uint extra_pb = extra_bo & 31u;
                                if (extra_pw != extra_idx) {
                                    extra_idx = extra_pw;
                                    extra_val = k_packed[base + extra_pw];
                                }
                                uint extra_asm = extra_val >> extra_pb;
                                if (extra_pb + extra_bits > 32u) {
                                    extra_asm |=
                                        k_packed[base + extra_pw + 1u] << (32u - extra_pb);
                                }
                                code |= (extra_asm & ((1u << extra_bits) - 1u)) << coop_base_bits;
                            }
                        } else {
                            bits = coop_base_bits;
                            uint bo = local * coop_base_bits;
                            uint pw = bo >> 5;
                            uint pb = bo & 31u;
                            if (pw != cached_idx) {
                                cached_idx = pw;
                                cached_val = k_packed[base + pw];
                            }
                            uint aw = cached_val >> pb;
                            if (pb + coop_base_bits > 32u) {
                                aw |= k_packed[base + pw + 1u] << (32u - pb);
                            }
                            code = aw & ((1u << coop_base_bits) - 1u);
                        }
                        uint bw = local >> 5;
                        if (bw != cached_bw) {
                            cached_bw = bw;
                            cached_sign = k_signs[tq_bitset_offset(
                                batch, kv_head, phys, g, bw,
                                uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                                uint(BITSET_WORDS_PER_GROUP))];
                        }
                        float level = tq_codebook_unit(bits, code) * inv_sqrt_count;
                        float sgn = (cached_sign & (1u << (local & 31u))) != 0u ? -1.0f : 1.0f;
                        for (uint r = 0u; r < 4u; r++) {
                            float qv = query_cache[r * uint(HEAD_DIM) + dim];
                            qd[r] += qv * level;
                            sd[r] += sgn * qv;
                        }
                    }
                    float norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 0u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_norm = k_scales[tq_scale_offset(
                        batch, kv_head, phys, g, 1u,
                        uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))];
                    float residual_scale = residual_norm * residual_scale_factor;
                    for (uint r = 0u; r < 4u; r++) {
                        ts[r] = norm * qd[r] + residual_scale * sd[r];
                    }
                }
                for (uint r = 0u; r < 4u; r++) {
                    ts[r] += simd_shuffle_xor(ts[r], 1u);
                    ts[r] += simd_shuffle_xor(ts[r], 2u);
                }
                if (lane_in_quad == j) {
                    active = tok_active;
                    my_token = tok_active ? phys : 0u;
                    for (uint r = 0u; r < 4u; r++) {
                        scaled_scores[r] = tok_active ? ts[r] * attention_scale : -INFINITY;
                    }
                }
            }
            physical_token = my_token;
        } else {
        uint logical_token = block_start + lane;
        active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);

        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] = 0.0f;
            }
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                uint group_start = group * uint(GROUP_SIZE);
                uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                if (repeat_count == 4u) {
                    thread float query_values[4 * GROUP_SIZE];
                    thread float quad_scores[4];
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        quad_scores[repeat] = 0.0f;
                        for (uint local = 0u; local < count; local++) {
                            query_values[repeat * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_quad(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        quad_scores,
                        batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint repeat = 0u; repeat < 4u; repeat++) {
                        scaled_scores[repeat] += quad_scores[repeat];
                    }
                    continue;
                }
                for (uint pair_start = 0u; pair_start < repeat_count; pair_start += 2u) {
                    uint pair_repeats = min(2u, repeat_count - pair_start);
                    thread float query_values[2 * GROUP_SIZE];
                    thread float pair_scores[2];
                    pair_scores[0] = 0.0f;
                    pair_scores[1] = 0.0f;
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        uint repeat = pair_start + pair;
                        for (uint local = 0u; local < count; local++) {
                            query_values[pair * uint(GROUP_SIZE) + local] =
                                query_cache[repeat * uint(HEAD_DIM) + group_start + local];
                        }
                    }
                    tq_product_attention_inner_product_group_pair(
                        k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                        pair_scores,
                        pair_repeats, batch, kv_head, physical_token, group, key_seed,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                        uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                        uint(HEAD_DIM),
                        tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)),
                        true);
                    for (uint pair = 0u; pair < pair_repeats; pair++) {
                        scaled_scores[pair_start + pair] += pair_scores[pair];
                    }
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                scaled_scores[repeat] *= attention_scale;
            }
        }
        }
        tile_physical_tokens[lane] = physical_token;

        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_scores[score_base + lane] = scaled_scores[repeat];
            partial[score_base + lane] = scaled_scores[repeat];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] =
                        max(partial[score_base + lane], partial[score_base + lane + stride]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_maxes[repeat] = partial[score_base];
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = tile_weight;
            partial[score_base + lane] = tile_weight;
            if (tile_weight > 0.0f) {
                has_weight = 1u;
            }
        }
        tile_has_weight[lane] = has_weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    partial[score_base + lane] += partial[score_base + lane + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = partial[score_base];
            }
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            thread float dimension_accum[4];
            dimension_accum[0] = 0.0f;
            dimension_accum[1] = 0.0f;
            dimension_accum[2] = 0.0f;
            dimension_accum[3] = 0.0f;

            for (uint tile_lane = 0u; tile_lane < threads_per_block; tile_lane++) {
                if (tile_has_weight[tile_lane] != 0u) {
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, tile_physical_tokens[tile_lane], lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        dimension_accum[repeat] +=
                            tile_scores[repeat * threads_per_block + tile_lane] * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = dimension_accum[repeat];
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_block_reduce_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scales[512];

        if (lane < uint(BLOCK_COUNT)) {
            partial[lane] = partial_stats[(row * uint(BLOCK_COUNT) + lane) * 2u];
        } else {
            partial[lane] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_max = partial[0];
        if (lane < uint(BLOCK_COUNT)) {
            uint stat_index = (row * uint(BLOCK_COUNT) + lane) * 2u;
            float tile_sum = partial_stats[stat_index + 1u];
            float tile_scale = tile_sum > 0.0f ? exp(partial_stats[stat_index] - row_max) : 0.0f;
            tile_scales[lane] = tile_scale;
            partial[lane] = tile_scale * tile_sum;
        } else {
            tile_scales[lane] = 0.0f;
            partial[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float row_sum = partial[0];
        if (lane < uint(HEAD_DIM)) {
            float accum = 0.0f;
            for (uint block = 0u; block < uint(BLOCK_COUNT); block++) {
                float tile_scale = tile_scales[block];
                if (tile_scale > 0.0f) {
                    uint partial_index = ((row * uint(BLOCK_COUNT) + block) * uint(HEAD_DIM)) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(
                accum / max(row_sum, 1.17549435e-38f));
        }
)TQMLX";

} // namespace mlx::core::fast::turbo_quant_detail
