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
                * groups_per_vector + group) * 2u) + scale_index;
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

inline constexpr std::string_view turbo_quant_attention_header_v7_ext = R"TQMLX(        inline uint tq_packed_offset_v7(
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
            uint words_per_token = groups_per_vector * mag_words_per_group;
            uint plane_base = (batch * kv_heads + head) * capacity * words_per_token;
            uint tile_base = plane_base + (token & ~31u) * words_per_token;
            return tile_base + ((group * mag_words_per_group + word) << 5) + (token & 31u);
        }

        inline uint tq_bitset_offset_v7(
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
            uint words_per_token = groups_per_vector * bitset_words_per_group;
            uint plane_base = (batch * kv_heads + head) * capacity * words_per_token;
            uint tile_base = plane_base + (token & ~31u) * words_per_token;
            return tile_base + ((group * bitset_words_per_group + word) << 5) + (token & 31u);
        }

        inline uint tq_scale_offset_v7(
            uint batch,
            uint head,
            uint token,
            uint group,
            uint scale_index,
            uint kv_heads,
            uint capacity,
            uint groups_per_vector
        ) {
            uint scales_per_token = groups_per_vector * 2u;
            uint plane_base = (batch * kv_heads + head) * capacity * scales_per_token;
            uint tile_base = plane_base + (token & ~31u) * scales_per_token;
            return tile_base + ((group * 2u + scale_index) << 5) + (token & 31u);
        }

        template <typename PackedPtr>
        inline uint tq_read_packed_unsigned_v7(
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
            uint first = packed[tq_packed_offset_v7(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] >> packed_bit;
            if (packed_bit + bits > 32u) {
                uint next = packed[tq_packed_offset_v7(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)];
                first |= next << (32u - packed_bit);
            }
            return first & ((1u << bits) - 1u);
        }

        template <typename PackedPtr>
        inline void tq_write_packed_unsigned_v7(
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
            packed[tq_packed_offset_v7(
                batch, head, token, group, packed_word,
                kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                value << packed_bit;
            if (packed_bit + bits > 32u) {
                packed[tq_packed_offset_v7(
                    batch, head, token, group, packed_word + 1u,
                    kv_heads, capacity, groups_per_vector, mag_words_per_group)] |=
                    value >> (32u - packed_bit);
            }
        }

        template <
            typename PackedPtr,
            typename SignsPtr,
            typename HighMaskPtr,
            typename ResidualSignsPtr,
            typename ScalesPtr
        >
        inline void tq_product_attention_inner_product_group_pair_v7(
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
            uint tqopt_packed_base = tq_packed_offset_v7(
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
                    cached_sign_bits = signs[tq_bitset_offset_v7(
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
                        tqopt_cached_val = packed[tqopt_packed_base + (base_pw << 5)];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + ((base_pw + 1u) << 5)] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + (extra_pw << 5)];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + ((extra_pw + 1u) << 5)] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset_v7(
                            batch, head, token, group, bitset_word,
                            kv_heads, capacity, groups_per_vector, bitset_words_per_group)];
                    }
                    bool high_precision = (cached_high_bits & bit_mask) != 0u;
                    bits = high_precision ? key_high_bits : key_base_bits;
                    code = tq_read_packed_unsigned_v7(
                        packed, batch, head, token, group, bit_offset, bits,
                        kv_heads, capacity, groups_per_vector, mag_words_per_group);
                    bit_offset += bits;
                } else {
                    code = tq_read_packed_unsigned_v7(
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

            float norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset_v7(
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
        inline void tq_product_attention_inner_product_group_quad_v7(
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
            uint tqopt_packed_base = tq_packed_offset_v7(
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
                    cached_sign_bits = signs[tq_bitset_offset_v7(
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
                        tqopt_cached_val = packed[tqopt_packed_base + (base_pw << 5)];
                    }
                    uint base_asm = tqopt_cached_val >> base_pbit;
                    if (base_pbit + key_base_bits > 32u) {
                        base_asm |= packed[tqopt_packed_base + ((base_pw + 1u) << 5)] << (32u - base_pbit);
                    }
                    code = base_asm & ((1u << key_base_bits) - 1u);
                    if (high_precision) {
                        uint extra_bits = key_high_bits - key_base_bits;
                        uint extra_bo = group_size * key_base_bits + local;
                        uint extra_pw = extra_bo >> 5;
                        uint extra_pbit = extra_bo & 31u;
                        if (extra_pw != tqopt_extra_idx) {
                            tqopt_extra_idx = extra_pw;
                            tqopt_extra_val = packed[tqopt_packed_base + (extra_pw << 5)];
                        }
                        uint extra_asm = tqopt_extra_val >> extra_pbit;
                        if (extra_pbit + extra_bits > 32u) {
                            extra_asm |= packed[tqopt_packed_base + ((extra_pw + 1u) << 5)] << (32u - extra_pbit);
                        }
                        uint extra_code = extra_asm & ((1u << extra_bits) - 1u);
                        code |= extra_code << key_base_bits;
                    }
                } else if (key_high_bits > key_base_bits) {
                    if (bitset_word != cached_high_word) {
                        cached_high_word = bitset_word;
                        cached_high_bits = high_mask[tq_bitset_offset_v7(
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
                        tqopt_cached_val = packed[tqopt_packed_base + (tqopt_pw << 5)];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + ((tqopt_pw + 1u) << 5)] << (32u - tqopt_pbit);
                    }
                    code = tqopt_asm & ((1u << bits) - 1u);
                    bit_offset += bits;
                } else {
                    // TQPROF_OPT cached uniform-width packed read (1 load per 32/bits codes)
                    uint tqopt_pw = bit_offset >> 5;
                    uint tqopt_pbit = bit_offset & 31u;
                    if (tqopt_pw != tqopt_cached_idx) {
                        tqopt_cached_idx = tqopt_pw;
                        tqopt_cached_val = packed[tqopt_packed_base + (tqopt_pw << 5)];
                    }
                    uint tqopt_asm = tqopt_cached_val >> tqopt_pbit;
                    if (tqopt_pbit + bits > 32u) {
                        tqopt_asm |= packed[tqopt_packed_base + ((tqopt_pw + 1u) << 5)] << (32u - tqopt_pbit);
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

            float norm = scales[tq_scale_offset_v7(
                batch, head, token, group, 0u, kv_heads, capacity, groups_per_vector)];
            float residual_norm = scales[tq_scale_offset_v7(
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
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_ROW];
        threadgroup uint count_partial[THREADS_PER_ROW];
        threadgroup float tile_scores[THREADS_PER_ROW];
        threadgroup uint tile_physical_tokens[THREADS_PER_ROW];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        float sparse_v_threshold = float(runtime_sparse_v_threshold);
        uint selection_mode = uint(runtime_sparse_v_selection_mode);
        int top_k_value = int(runtime_sparse_v_top_k);
        uint sparse_top_k = top_k_value > 0 ? uint(top_k_value) : 0u;
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

            if (selection_mode == 2u && logical_length <= threads_per_row) {
                tile_scores[lane] = scaled_score;
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

        float topk_score_cutoff = -INFINITY;
        uint topk_token_cutoff = 0xffffffffu;
        if (selection_mode == 2u && logical_length <= threads_per_row) {
            uint active_count = min(logical_length, DO_CAUSAL ? causal_limit + 1u : logical_length);
            uint limit = min(sparse_top_k, active_count);
            if (limit == 0u) {
                topk_score_cutoff = INFINITY;
                topk_token_cutoff = 0u;
            } else if (limit >= active_count) {
                topk_score_cutoff = -INFINITY;
                topk_token_cutoff = 0xffffffffu;
            } else {
                partial[lane] = lane < active_count ? tile_scores[lane] : -INFINITY;
                count_partial[lane] = lane < active_count ? lane : 0xffffffffu;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint width = 2u; width <= threads_per_row; width <<= 1) {
                    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                        uint pair_lane = lane ^ stride;
                        float self_score = partial[lane];
                        uint self_token = count_partial[lane];
                        float pair_score = partial[pair_lane];
                        uint pair_token = count_partial[pair_lane];
                        bool self_before_pair = self_score > pair_score
                            || (self_score == pair_score && self_token < pair_token);
                        bool descending = (lane & width) == 0u;
                        bool swap = descending ? !self_before_pair : self_before_pair;
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (pair_lane > lane && swap) {
                            partial[lane] = pair_score;
                            count_partial[lane] = pair_token;
                            partial[pair_lane] = self_score;
                            count_partial[pair_lane] = self_token;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
                topk_score_cutoff = partial[limit - 1u];
                topk_token_cutoff = count_partial[limit - 1u];
            }
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
                if (selection_mode == 2u && logical_length <= threads_per_row) {
                    scaled_score = tile_scores[lane];
                } else {
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
            }

            float final_weight = active ? exp(scaled_score - row_max) * inv_row_sum : 0.0f;
            bool topk_retained = scaled_score > topk_score_cutoff
                || (scaled_score == topk_score_cutoff && logical_token <= topk_token_cutoff);
            bool skipped = active && (selection_mode == 2u
                ? !topk_retained
                : final_weight < sparse_v_threshold);
            tile_scores[lane] = skipped ? 0.0f : final_weight;
            tile_physical_tokens[lane] = physical_token;
            if (output_sparse_stats) {
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
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
        if (output_sparse_stats && lane == 0u) {
            sparse_stats[row * 2u] = skipped_count;
            sparse_stats[row * 2u + 1u] = total_count;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_page_scores_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint block_tokens = uint(BLOCK_TOKENS);
        constexpr uint page_score_samples = uint(PAGE_SCORE_SAMPLES);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float q_group_abs[GROUPS_PER_VECTOR];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = max(uint(QUERY_HEADS) / uint(KV_HEADS), 1u);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint visible_length = DO_CAUSAL ? min(logical_length, causal_limit + 1u) : logical_length;

        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
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

        uint sample_count = min(page_score_samples, block_tokens);
        uint token_offset = sample_count <= 1u
            ? 0u
            : (lane * (block_tokens - 1u)) / (sample_count - 1u);
        uint logical_token = block_index * block_tokens + token_offset;
        float token_bound = -INFINITY;
        if (lane < sample_count && logical_token < visible_length) {
            uint physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            token_bound = 0.0f;
            for (uint group = 0u; group < uint(GROUPS_PER_VECTOR); group++) {
                float norm = fabs(k_scales[tq_scale_offset(
                    batch, kv_head, physical_token, group, 0u,
                    uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))]);
                float residual_norm = fabs(k_scales[tq_scale_offset(
                    batch, kv_head, physical_token, group, 1u,
                    uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR))]);
                token_bound += q_group_abs[group] * (norm + residual_norm);
            }
        }
        partial[lane] = token_bound;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            page_score_tiles[(row * uint(BLOCK_COUNT)) + block_index] = partial[0];
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_page_summary_scores_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float q_group_abs[GROUPS_PER_VECTOR];

        uint logical_length = uint(runtime_logical_length);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = max(uint(QUERY_HEADS) / uint(KV_HEADS), 1u);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint visible_length = DO_CAUSAL ? min(logical_length, causal_limit + 1u) : logical_length;

        if (lane < uint(HEAD_DIM)) {
            long q_index =
                long(batch) * q_strides[0]
                + long(q_head) * q_strides[1]
                + long(q_token) * q_strides[2]
                + long(lane) * q_strides[3];
            query_cache[lane] = float(q[q_index]);
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

        float contribution = 0.0f;
        if (block_index * uint(BLOCK_TOKENS) < visible_length &&
            lane < uint(GROUPS_PER_VECTOR) &&
            block_index < uint(PAGE_CAPACITY)) {
            uint offset = (((batch * uint(KV_HEADS) + kv_head) * uint(PAGE_CAPACITY)
                + block_index) * uint(GROUPS_PER_VECTOR)) + lane;
            contribution = q_group_abs[lane] * float(page_summary[offset]);
        }
        partial[lane] = contribution;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            page_score_tiles[(row * uint(BLOCK_COUNT)) + block_index] = partial[0];
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_page_topk_attention_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint block_tokens = uint(BLOCK_TOKENS);
        constexpr uint recent_tokens = uint(PAGE_RECENT_TOKENS);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_ROW];
        threadgroup float tile_scores[THREADS_PER_ROW];
        threadgroup uint tile_physical_tokens[THREADS_PER_ROW];
        threadgroup float page_scores[THREADS_PER_ROW];
        threadgroup uint page_tokens[THREADS_PER_ROW];
        threadgroup uint retained_pages[8];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float output_accum[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        int page_top_k_value = int(runtime_sparse_v_top_k);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = max(uint(QUERY_HEADS) / uint(KV_HEADS), 1u);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint visible_length = DO_CAUSAL ? min(logical_length, causal_limit + 1u) : logical_length;
        uint page_count = (visible_length + block_tokens - 1u) / block_tokens;
        uint page_top_k = page_top_k_value > 0 ? uint(page_top_k_value) : 0u;
        page_top_k = min(page_top_k, page_count);
        uint recent_count = min(recent_tokens, visible_length);
        uint recent_start = visible_length - recent_count;
        uint recent_first_page = recent_count > 0u ? recent_start / block_tokens : page_count;
        uint recent_page_count = recent_count > 0u
            ? ((visible_length - 1u) / block_tokens) - recent_first_page + 1u
            : 0u;
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
        if (lane < page_count) {
            page_scores[lane] = page_score_tiles[(row * uint(BLOCK_COUNT)) + lane];
            page_tokens[lane] = lane;
        } else {
            page_scores[lane] = -INFINITY;
            page_tokens[lane] = 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < page_count && recent_count > 0u) {
            uint page = page_tokens[lane];
            uint page_start = page * block_tokens;
            if (page_start >= recent_start) {
                page_scores[lane] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (page_top_k <= 8u) {
            if (lane == 0u) {
                for (uint rank = 0u; rank < page_top_k; rank++) {
                    float best_score = -INFINITY;
                    uint best_page = 0xffffffffu;
                    for (uint page_index = 0u; page_index < page_count; page_index++) {
                        uint page = page_tokens[page_index];
                        bool already_selected = false;
                        for (uint prior = 0u; prior < rank; prior++) {
                            already_selected = already_selected || retained_pages[prior] == page;
                        }
                        float score = page_scores[page_index];
                        bool better = score > best_score
                            || (score == best_score && page < best_page);
                        if (!already_selected && better) {
                            best_score = score;
                            best_page = page;
                        }
                    }
                    retained_pages[rank] = best_page;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < page_top_k) {
                page_tokens[lane] = retained_pages[lane];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            for (uint width = 2u; width <= threads_per_row; width <<= 1) {
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    uint pair_lane = lane ^ stride;
                    float self_score = page_scores[lane];
                    uint self_page = page_tokens[lane];
                    float pair_score = page_scores[pair_lane];
                    uint pair_page = page_tokens[pair_lane];
                    bool self_before_pair = self_score > pair_score
                        || (self_score == pair_score && self_page < pair_page);
                    bool descending = (lane & width) == 0u;
                    bool swap = descending ? !self_before_pair : self_before_pair;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (pair_lane > lane && swap) {
                        page_scores[lane] = pair_score;
                        page_tokens[lane] = pair_page;
                        page_scores[pair_lane] = self_score;
                        page_tokens[pair_lane] = self_page;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        uint retained_iteration_count = page_top_k + recent_page_count;
        for (uint retained_index = 0u; retained_index < retained_iteration_count; retained_index++) {
            bool recent_iteration = retained_index >= page_top_k;
            uint page = recent_iteration
                ? recent_first_page + (retained_index - page_top_k)
                : page_tokens[retained_index];
            uint logical_token = page * block_tokens + lane;
            bool in_recent_window = recent_count > 0u
                && logical_token >= recent_start
                && logical_token < visible_length;
            bool active = lane < block_tokens
                && logical_token < visible_length
                && (recent_iteration ? in_recent_window : !in_recent_window);
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
                for (uint tile_lane = 0u; tile_lane < block_tokens; tile_lane++) {
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
            float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
            out[out_index] = static_cast<OUTPUT_DTYPE>(output_accum[lane] * inv_sum);
        }
        if (output_sparse_stats && lane == 0u) {
            uint retained = recent_count;
            for (uint rank = 0u; rank < page_top_k; rank++) {
                uint page = page_tokens[rank];
                uint start = page * block_tokens;
                if (start < visible_length) {
                    uint end = min(start + block_tokens, visible_length);
                    if (start < recent_start) {
                        retained += min(end, recent_start) - start;
                    }
                }
            }
            sparse_stats[row * 2u] = visible_length - min(visible_length, retained);
            sparse_stats[row * 2u + 1u] = visible_length;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_candidate_sparse_attention_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint block_tokens = 512u;
        constexpr uint sketch_dim = 32u;
        constexpr uint candidate_page_limit = uint(CANDIDATE_PAGE_LIMIT);
        constexpr uint candidate_token_limit = uint(CANDIDATE_TOKEN_LIMIT);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_ROW];
        threadgroup uint token_partial[THREADS_PER_ROW];
        threadgroup float page_scores[THREADS_PER_ROW];
        threadgroup uint page_tokens[THREADS_PER_ROW];
        threadgroup uint retained_pages[CANDIDATE_PAGE_LIMIT];
        threadgroup float query_cache[HEAD_DIM];
        threadgroup float query_projection[32];
        threadgroup float candidate_scores[CANDIDATE_TOKEN_LIMIT];
        threadgroup uint candidate_tokens[CANDIDATE_TOKEN_LIMIT];
        threadgroup float selected_scores[TOPK_LIMIT];
        threadgroup uint selected_tokens_tg[TOPK_LIMIT];
        threadgroup float tile_scores[THREADS_PER_ROW];
        threadgroup uint tile_physical_tokens[THREADS_PER_ROW];
        threadgroup float output_accum[HEAD_DIM];
        threadgroup uint selected_pages_tg;
        threadgroup uint candidate_tokens_considered_tg;
        threadgroup uint selected_older_tg;

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        int older_top_k_value = int(runtime_sparse_v_top_k);
        int recent_tokens_value = int(runtime_sparse_v_recent_tokens);
        int candidate_pages_value = int(runtime_sparse_v_candidate_pages);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = max(uint(QUERY_HEADS) / uint(KV_HEADS), 1u);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint visible_length = DO_CAUSAL ? min(logical_length, causal_limit + 1u) : logical_length;
        uint recent_budget = recent_tokens_value > 0 ? uint(recent_tokens_value) : 1024u;
        uint recent_count = min(recent_budget, visible_length);
        uint recent_start = visible_length - recent_count;
        uint older_page_count = (recent_start + block_tokens - 1u) / block_tokens;
        uint page_count = (visible_length + block_tokens - 1u) / block_tokens;
        uint requested_pages = candidate_pages_value > 0 ? uint(candidate_pages_value) : candidate_page_limit;
        uint page_select_count = min(min(requested_pages, candidate_page_limit), older_page_count);
        uint older_top_k = older_top_k_value > 0 ? uint(older_top_k_value) : 0u;
        older_top_k = min(older_top_k, topk_limit);
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
        if (lane < sketch_dim) {
            query_projection[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < sketch_dim) {
            float projection = 0.0f;
            for (uint dim = 0u; dim < uint(HEAD_DIM); dim++) {
                uint hash = (lane + 1u) * 747796405u ^ (dim + 1u) * 2891336453u;
                float sign = (hash & 1u) == 0u ? 1.0f : -1.0f;
                projection += query_cache[dim] * sign;
            }
            query_projection[lane] = projection;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < page_count && lane < threads_per_row) {
            float score = -INFINITY;
            if (lane < older_page_count && lane < uint(PAGE_CAPACITY)) {
                uint sketch_base =
                    (((batch * uint(KV_HEADS) + kv_head) * uint(PAGE_CAPACITY) + lane)
                        * 64u);
                score = 0.0f;
                for (uint proj = 0u; proj < sketch_dim; proj++) {
                    float q_proj = query_projection[proj];
                    float lo = key_candidate_sketch[sketch_base + proj];
                    float hi = key_candidate_sketch[sketch_base + sketch_dim + proj];
                    score += q_proj >= 0.0f ? q_proj * hi : q_proj * lo;
                }
            }
            page_scores[lane] = score;
            page_tokens[lane] = lane;
        } else if (lane < threads_per_row) {
            page_scores[lane] = -INFINITY;
            page_tokens[lane] = 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            uint tokens_considered = 0u;
            for (uint rank = 0u; rank < candidate_page_limit; rank++) {
                float best_score = -INFINITY;
                uint best_page = 0xffffffffu;
                if (rank < page_select_count) {
                    for (uint page_index = 0u; page_index < older_page_count; page_index++) {
                        uint page = page_tokens[page_index];
                        bool already_selected = false;
                        for (uint prior = 0u; prior < rank; prior++) {
                            already_selected = already_selected || retained_pages[prior] == page;
                        }
                        float score = page_scores[page_index];
                        bool better = score > best_score
                            || (score == best_score && page < best_page);
                        if (!already_selected && better) {
                            best_score = score;
                            best_page = page;
                        }
                    }
                }
                retained_pages[rank] = best_page;
                if (rank < page_select_count && best_page != 0xffffffffu) {
                    uint start = best_page * block_tokens;
                    if (start < recent_start) {
                        tokens_considered += min(block_tokens, recent_start - start);
                    }
                }
            }
            selected_pages_tg = page_select_count;
            candidate_tokens_considered_tg = tokens_considered;
            selected_older_tg = min(older_top_k, tokens_considered);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint slot = lane; slot < candidate_token_limit; slot += threads_per_row) {
            uint page_rank = slot / block_tokens;
            uint page_offset = slot - page_rank * block_tokens;
            uint page = page_rank < candidate_page_limit ? retained_pages[page_rank] : 0xffffffffu;
            uint logical_token = page * block_tokens + page_offset;
            bool active = page_rank < selected_pages_tg
                && page != 0xffffffffu
                && logical_token < recent_start
                && slot < candidate_token_limit;
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
            candidate_scores[slot] = scaled_score;
            candidate_tokens[slot] = active ? logical_token : 0xffffffffu;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_older_tg < candidate_tokens_considered_tg) {
            float previous_score = INFINITY;
            uint previous_token = 0u;
            for (uint rank = 0u; rank < topk_limit; rank++) {
                float best_score = -INFINITY;
                uint best_token = 0xffffffffu;
                if (rank < selected_older_tg) {
                    for (uint slot = lane; slot < candidate_token_limit; slot += threads_per_row) {
                        float score = candidate_scores[slot];
                        uint token = candidate_tokens[slot];
                        bool active = token != 0xffffffffu && score > -INFINITY;
                        bool after_previous = rank == 0u
                            || score < previous_score
                            || (score == previous_score && token > previous_token);
                        bool better = score > best_score
                            || (score == best_score && token < best_token);
                        if (active && after_previous && better) {
                            best_score = score;
                            best_token = token;
                        }
                    }
                }
                partial[lane] = best_score;
                token_partial[lane] = best_token;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        float other_score = partial[lane + stride];
                        uint other_token = token_partial[lane + stride];
                        bool other_better = other_score > partial[lane]
                            || (other_score == partial[lane] && other_token < token_partial[lane]);
                        if (other_better) {
                            partial[lane] = other_score;
                            token_partial[lane] = other_token;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0u) {
                    selected_scores[rank] = partial[0];
                    selected_tokens_tg[rank] =
                        rank < selected_older_tg ? token_partial[0] : 0xffffffffu;
                }
                previous_score = partial[0];
                previous_token = token_partial[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (uint tile_start = recent_start; tile_start < visible_length; tile_start += block_tokens) {
            uint logical_token = tile_start + lane;
            bool active = lane < block_tokens && logical_token < visible_length;
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
                for (uint tile_lane = 0u; tile_lane < block_tokens; tile_lane++) {
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

        if (selected_older_tg == candidate_tokens_considered_tg) {
            for (uint candidate_start = 0u; candidate_start < candidate_token_limit; candidate_start += block_tokens) {
                uint slot = candidate_start + lane;
                bool active = lane < block_tokens
                    && slot < candidate_token_limit
                    && candidate_tokens[slot] != 0xffffffffu;
                float scaled_score = active ? candidate_scores[slot] : -INFINITY;
                uint physical_token = active
                    ? tq_physical_token(
                        candidate_tokens[slot], uint(CAPACITY), ring_offset, pinned_prefix_length)
                    : 0u;
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
                    for (uint tile_lane = 0u; tile_lane < block_tokens; tile_lane++) {
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
        } else {
            for (uint rank = 0u; rank < topk_limit; rank++) {
                if (rank >= selected_older_tg) {
                    break;
                }
                float scaled_score = selected_scores[rank];
                uint logical_token = selected_tokens_tg[rank];
                float new_row_max = max(row_max, scaled_score);
                float old_scale = row_sum > 0.0f ? exp(row_max - new_row_max) : 0.0f;
                float weight = exp(scaled_score - new_row_max);
                if (lane < uint(HEAD_DIM)) {
                    output_accum[lane] *= old_scale;
                    uint physical_token = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    thread float decode_scratch[GROUP_SIZE];
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, physical_token, lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    output_accum[lane] += weight * value;
                }
                row_sum = row_sum * old_scale + weight;
                row_max = new_row_max;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (lane < uint(HEAD_DIM)) {
            float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + lane;
            out[out_index] = static_cast<OUTPUT_DTYPE>(output_accum[lane] * inv_sum);
        }
        if (output_sparse_stats && lane == 0u) {
            uint retained = min(visible_length, recent_count + selected_older_tg);
            uint stat_index = row * 8u;
            sparse_stats[stat_index] = visible_length > retained ? visible_length - retained : 0u;
            sparse_stats[stat_index + 1u] = visible_length;
            sparse_stats[stat_index + 2u] = recent_count;
            sparse_stats[stat_index + 3u] = selected_older_tg;
            sparse_stats[stat_index + 4u] = selected_pages_tg;
            sparse_stats[stat_index + 5u] = older_page_count;
            sparse_stats[stat_index + 6u] = candidate_tokens_considered_tg;
            sparse_stats[stat_index + 7u] = retained;
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

inline constexpr std::string_view turbo_quant_candidate_sparse_gqa_attention_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint block_tokens = 512u;
        constexpr uint sketch_dim = 32u;
        constexpr uint candidate_page_limit = uint(CANDIDATE_PAGE_LIMIT);
        constexpr uint candidate_token_limit = uint(CANDIDATE_TOKEN_LIMIT);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        constexpr uint repeat_group_count = uint(REPEAT_GROUP_COUNT);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group = threadgroup_position_in_grid.x;
        uint repeat_group = group % repeat_group_count;
        uint gqa_row = group / repeat_group_count;
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        thread float output_accum[4];
        output_accum[0] = 0.0f;
        output_accum[1] = 0.0f;
        output_accum[2] = 0.0f;
        output_accum[3] = 0.0f;
        threadgroup float partial[THREADS_PER_ROW];
        threadgroup uint token_partial[THREADS_PER_ROW];
        threadgroup uint retained_pages[CANDIDATE_PAGE_LIMIT];
        threadgroup float query_cache[4 * HEAD_DIM];
        threadgroup float query_projection[4 * 32];
        threadgroup float candidate_scores[CANDIDATE_TOKEN_LIMIT];
        threadgroup uint candidate_tokens[CANDIDATE_TOKEN_LIMIT];
        threadgroup uint selected_tokens_tg[TOPK_LIMIT];
        threadgroup float tile_scores[4 * THREADS_PER_ROW];
        threadgroup uint tile_physical_tokens[THREADS_PER_ROW];
        threadgroup uint selected_pages_tg;
        threadgroup uint candidate_tokens_considered_tg;
        threadgroup uint selected_older_tg;

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        int older_top_k_value = int(runtime_sparse_v_top_k);
        int recent_tokens_value = int(runtime_sparse_v_recent_tokens);
        int candidate_pages_value = int(runtime_sparse_v_candidate_pages);
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint repeat_base = repeat_group * 4u;
        uint repeat_count = min(4u, gqa_repeats - repeat_base);
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint visible_length = DO_CAUSAL ? min(logical_length, causal_limit + 1u) : logical_length;
        uint recent_budget = recent_tokens_value > 0 ? uint(recent_tokens_value) : 1024u;
        uint recent_count = min(recent_budget, visible_length);
        uint recent_start = visible_length - recent_count;
        uint older_page_count = (recent_start + block_tokens - 1u) / block_tokens;
        uint requested_pages = candidate_pages_value > 0 ? uint(candidate_pages_value) : candidate_page_limit;
        uint page_select_count = min(min(requested_pages, candidate_page_limit), older_page_count);
        uint older_top_k = older_top_k_value > 0 ? uint(older_top_k_value) : 0u;
        older_top_k = min(older_top_k, topk_limit);
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                if (repeat < repeat_count) {
                    uint q_head = kv_head * gqa_repeats + repeat_base + repeat;
                    long q_index =
                        long(batch) * q_strides[0]
                        + long(q_head) * q_strides[1]
                        + long(q_token) * q_strides[2]
                        + long(lane) * q_strides[3];
                    query_cache[repeat * uint(HEAD_DIM) + lane] = float(q[q_index]);
                } else {
                    query_cache[repeat * uint(HEAD_DIM) + lane] = 0.0f;
                }
            }
        }
        if (lane < 4u * sketch_dim) {
            query_projection[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < repeat_count * sketch_dim) {
            uint repeat = lane / sketch_dim;
            uint proj = lane - repeat * sketch_dim;
            float projection = 0.0f;
            for (uint dim = 0u; dim < uint(HEAD_DIM); dim++) {
                uint hash = (proj + 1u) * 747796405u ^ (dim + 1u) * 2891336453u;
                float sign = (hash & 1u) == 0u ? 1.0f : -1.0f;
                projection += query_cache[repeat * uint(HEAD_DIM) + dim] * sign;
            }
            query_projection[repeat * sketch_dim + proj] = projection;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            uint tokens_considered = 0u;
            for (uint rank = 0u; rank < candidate_page_limit; rank++) {
                float best_score = -INFINITY;
                uint best_page = 0xffffffffu;
                if (rank < page_select_count) {
                    for (uint page_index = 0u; page_index < older_page_count; page_index++) {
                        uint page = page_index;
                        bool already_selected = false;
                        for (uint prior = 0u; prior < rank; prior++) {
                            already_selected = already_selected || retained_pages[prior] == page;
                        }
                        float group_score = -INFINITY;
                        if (page < uint(PAGE_CAPACITY)) {
                            uint sketch_base =
                                (((batch * uint(KV_HEADS) + kv_head) * uint(PAGE_CAPACITY) + page)
                                    * 64u);
                            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                                float score = 0.0f;
                                for (uint proj = 0u; proj < sketch_dim; proj++) {
                                    float q_proj = query_projection[repeat * sketch_dim + proj];
                                    float lo = key_candidate_sketch[sketch_base + proj];
                                    float hi = key_candidate_sketch[sketch_base + sketch_dim + proj];
                                    score += q_proj >= 0.0f ? q_proj * hi : q_proj * lo;
                                }
                                group_score = max(group_score, score);
                            }
                        }
                        bool better = group_score > best_score
                            || (group_score == best_score && page < best_page);
                        if (!already_selected && better) {
                            best_score = group_score;
                            best_page = page;
                        }
                    }
                }
                retained_pages[rank] = best_page;
                if (rank < page_select_count && best_page != 0xffffffffu) {
                    uint start = best_page * block_tokens;
                    if (start < recent_start) {
                        tokens_considered += min(block_tokens, recent_start - start);
                    }
                }
            }
            selected_pages_tg = page_select_count;
            candidate_tokens_considered_tg = tokens_considered;
            selected_older_tg = min(older_top_k, tokens_considered);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint slot = lane; slot < candidate_token_limit; slot += threads_per_row) {
            uint page_rank = slot / block_tokens;
            uint page_offset = slot - page_rank * block_tokens;
            uint page = page_rank < candidate_page_limit ? retained_pages[page_rank] : 0xffffffffu;
            uint logical_token = page * block_tokens + page_offset;
            bool active = page_rank < selected_pages_tg
                && page != 0xffffffffu
                && logical_token < recent_start
                && slot < candidate_token_limit;
            candidate_tokens[slot] = active ? logical_token : 0xffffffffu;
            thread float scaled_scores[4];
            scaled_scores[0] = -INFINITY;
            scaled_scores[1] = -INFINITY;
            scaled_scores[2] = -INFINITY;
            scaled_scores[3] = -INFINITY;
            float max_scaled_score = -INFINITY;
            if (active) {
                uint physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    scaled_scores[repeat] = 0.0f;
                }
                for (uint group_index = 0u; group_index < uint(GROUPS_PER_VECTOR); group_index++) {
                    uint group_start = group_index * uint(GROUP_SIZE);
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
                            batch, kv_head, physical_token, group_index, key_seed,
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
                            pair_repeats, batch, kv_head, physical_token, group_index, key_seed,
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
                    max_scaled_score = max(max_scaled_score, scaled_scores[repeat]);
                }
            }
            candidate_scores[slot] = max_scaled_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float previous_score = INFINITY;
        uint previous_token = 0u;
        for (uint rank = 0u; rank < topk_limit; rank++) {
            float best_score = -INFINITY;
            uint best_token = 0xffffffffu;
            if (rank < selected_older_tg) {
                for (uint slot = lane; slot < candidate_token_limit; slot += threads_per_row) {
                    uint token = candidate_tokens[slot];
                    bool active = token != 0xffffffffu;
                    float score = active ? candidate_scores[slot] : -INFINITY;
                    bool after_previous = rank == 0u
                        || score < previous_score
                        || (score == previous_score && token > previous_token);
                    bool better = score > best_score
                        || (score == best_score && token < best_token);
                    if (active && after_previous && better) {
                        best_score = score;
                        best_token = token;
                    }
                }
            }
            partial[lane] = best_score;
            token_partial[lane] = best_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    float other_score = partial[lane + stride];
                    uint other_token = token_partial[lane + stride];
                    bool other_better = other_score > partial[lane]
                        || (other_score == partial[lane] && other_token < token_partial[lane]);
                    if (other_better) {
                        partial[lane] = other_score;
                        token_partial[lane] = other_token;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                uint selected_token = rank < selected_older_tg ? token_partial[0] : 0xffffffffu;
                selected_tokens_tg[rank] = selected_token;
            }
            previous_score = partial[0];
            previous_token = token_partial[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float row_maxes[4];
        thread float row_sums[4];
        row_maxes[0] = -INFINITY;
        row_maxes[1] = -INFINITY;
        row_maxes[2] = -INFINITY;
        row_maxes[3] = -INFINITY;
        row_sums[0] = 0.0f;
        row_sums[1] = 0.0f;
        row_sums[2] = 0.0f;
        row_sums[3] = 0.0f;

        for (uint tile_start = recent_start; tile_start < visible_length; tile_start += block_tokens) {
            uint logical_token = tile_start + lane;
            bool active_token = lane < block_tokens && logical_token < visible_length;
            thread float scaled_scores[4];
            scaled_scores[0] = -INFINITY;
            scaled_scores[1] = -INFINITY;
            scaled_scores[2] = -INFINITY;
            scaled_scores[3] = -INFINITY;
            uint physical_token = 0u;
            if (active_token) {
                physical_token = tq_physical_token(
                    logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    scaled_scores[repeat] = 0.0f;
                }
                for (uint group_index = 0u; group_index < uint(GROUPS_PER_VECTOR); group_index++) {
                    uint group_start = group_index * uint(GROUP_SIZE);
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
                            batch, kv_head, physical_token, group_index, key_seed,
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
                            pair_repeats, batch, kv_head, physical_token, group_index, key_seed,
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
            tile_physical_tokens[lane] = physical_token;
            for (uint repeat = 0u; repeat < 4u; repeat++) {
                uint score_base = repeat * threads_per_row;
                tile_scores[score_base + lane] =
                    repeat < repeat_count ? scaled_scores[repeat] : -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            thread float old_scales[4];
            thread float tile_sums[4];
            old_scales[0] = 0.0f;
            old_scales[1] = 0.0f;
            old_scales[2] = 0.0f;
            old_scales[3] = 0.0f;
            tile_sums[0] = 0.0f;
            tile_sums[1] = 0.0f;
            tile_sums[2] = 0.0f;
            tile_sums[3] = 0.0f;
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint score_base = repeat * threads_per_row;
                partial[lane] = tile_scores[score_base + lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] = max(partial[lane], partial[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float tile_max = partial[0];
                float new_row_max = max(row_maxes[repeat], tile_max);
                float old_scale = row_sums[repeat] > 0.0f
                    ? exp(row_maxes[repeat] - new_row_max)
                    : 0.0f;
                old_scales[repeat] = old_scale;
                if (lane < uint(HEAD_DIM)) {
                    output_accum[repeat] *= old_scale;
                }
                float weight = active_token
                    ? exp(tile_scores[score_base + lane] - new_row_max)
                    : 0.0f;
                tile_scores[score_base + lane] = weight;
                partial[lane] = weight;
                row_maxes[repeat] = new_row_max;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] += partial[lane + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                tile_sums[repeat] = partial[0];
            }
            if (lane < uint(HEAD_DIM)) {
                thread float decode_scratch[GROUP_SIZE];
                for (uint tile_lane = 0u; tile_lane < block_tokens; tile_lane++) {
                    bool has_weight = false;
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        has_weight = has_weight
                            || tile_scores[repeat * threads_per_row + tile_lane] > 0.0f;
                    }
                    if (has_weight) {
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
                            output_accum[repeat] +=
                                tile_scores[repeat * threads_per_row + tile_lane] * value;
                        }
                    }
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                row_sums[repeat] =
                    row_sums[repeat] * old_scales[repeat]
                    + tile_sums[repeat];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint rank = 0u; rank < topk_limit; rank++) {
            if (rank >= selected_older_tg) {
                break;
            }
            uint logical_token = selected_tokens_tg[rank];
            bool active = logical_token != 0xffffffffu;
            uint physical_token = active
                ? tq_physical_token(logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length)
                : 0u;
            if (lane < repeat_count) {
                float scaled_score = -INFINITY;
                if (active) {
                    float score = 0.0f;
                    for (uint group_index = 0u; group_index < uint(GROUPS_PER_VECTOR); group_index++) {
                        uint group_start = group_index * uint(GROUP_SIZE);
                        uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
                        thread float query_values[GROUP_SIZE];
                        for (uint local = 0u; local < count; local++) {
                            query_values[local] =
                                query_cache[lane * uint(HEAD_DIM) + group_start + local];
                        }
                        score += tq_product_attention_inner_product_group(
                            k_packed, k_signs, k_high_mask, k_residual_signs, k_scales, query_values,
                            batch, kv_head, physical_token, group_index, key_seed,
                            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                            uint(MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP),
                            uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS), uint(LAYOUT_VERSION),
                            uint(HEAD_DIM),
                            tq_high_precision_count(count, uint(HIGH_NUMERATOR), uint(HIGH_DENOMINATOR)));
                    }
                    scaled_score = score * attention_scale;
                }
                partial[lane] = scaled_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < uint(HEAD_DIM)) {
                thread float decode_scratch[GROUP_SIZE];
                float value = active
                    ? tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, physical_token, lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch)
                    : 0.0f;
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    float scaled_score = partial[repeat];
                    float new_row_max = max(row_maxes[repeat], scaled_score);
                    float old_scale = row_sums[repeat] > 0.0f
                        ? exp(row_maxes[repeat] - new_row_max)
                        : 0.0f;
                    float weight = active ? exp(scaled_score - new_row_max) : 0.0f;
                    output_accum[repeat] = output_accum[repeat] * old_scale + weight * value;
                }
            }
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                float scaled_score = partial[repeat];
                float new_row_max = max(row_maxes[repeat], scaled_score);
                float old_scale = row_sums[repeat] > 0.0f
                    ? exp(row_maxes[repeat] - new_row_max)
                    : 0.0f;
                float weight = active ? exp(scaled_score - new_row_max) : 0.0f;
                row_sums[repeat] = row_sums[repeat] * old_scale + weight;
                row_maxes[repeat] = new_row_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < uint(HEAD_DIM)) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                float inv_sum = row_sums[repeat] > 0.0f ? 1.0f / row_sums[repeat] : 0.0f;
                uint q_head = kv_head * gqa_repeats + repeat_base + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = (row * uint(HEAD_DIM)) + lane;
                out[out_index] =
                    static_cast<OUTPUT_DTYPE>(output_accum[repeat] * inv_sum);
            }
        }
        if (output_sparse_stats && lane == 0u) {
            uint retained = min(visible_length, recent_count + selected_older_tg);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat_base + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = row * 8u;
                sparse_stats[stat_index] = visible_length > retained ? visible_length - retained : 0u;
                sparse_stats[stat_index + 1u] = visible_length;
                sparse_stats[stat_index + 2u] = recent_count;
                sparse_stats[stat_index + 3u] = selected_older_tg;
                sparse_stats[stat_index + 4u] = selected_pages_tg;
                sparse_stats[stat_index + 5u] = older_page_count;
                sparse_stats[stat_index + 6u] = candidate_tokens_considered_tg;
                sparse_stats[stat_index + 7u] = retained;
            }
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_extended_diagnostics_source = R"TQMLX(        constexpr uint threads_per_block = 256u;
        uint lane = thread_position_in_threadgroup.x;
        threadgroup uint partial0[256];
        threadgroup uint partial1[256];
        threadgroup uint partial2[256];
        threadgroup uint partial3[256];
        threadgroup uint partial4[256];
        threadgroup uint partial5[256];
        threadgroup uint partial6[256];
        threadgroup uint partial7[256];

        uint sum0 = 0u;
        uint sum1 = 0u;
        uint sum2 = 0u;
        uint sum3 = 0u;
        uint sum4 = 0u;
        uint sum5 = 0u;
        uint sum6 = 0u;
        uint sum7 = 0u;
        for (uint row = lane; row < uint(ROW_COUNT); row += threads_per_block) {
            uint base = row * 8u;
            sum0 += sparse_stats[base];
            sum1 += sparse_stats[base + 1u];
            sum2 += sparse_stats[base + 2u];
            sum3 += sparse_stats[base + 3u];
            sum4 += sparse_stats[base + 4u];
            sum5 += sparse_stats[base + 5u];
            sum6 += sparse_stats[base + 6u];
            sum7 += sparse_stats[base + 7u];
        }
        partial0[lane] = sum0;
        partial1[lane] = sum1;
        partial2[lane] = sum2;
        partial3[lane] = sum3;
        partial4[lane] = sum4;
        partial5[lane] = sum5;
        partial6[lane] = sum6;
        partial7[lane] = sum7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial0[lane] += partial0[lane + stride];
                partial1[lane] += partial1[lane + stride];
                partial2[lane] += partial2[lane + stride];
                partial3[lane] += partial3[lane + stride];
                partial4[lane] += partial4[lane + stride];
                partial5[lane] += partial5[lane + stride];
                partial6[lane] += partial6[lane + stride];
                partial7[lane] += partial7[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            diagnostics[0] = int(BACKEND_VERSION);
            diagnostics[1] = int(KERNEL_KIND);
            diagnostics[2] = int(ACTIVE_BLOCKS);
            diagnostics[3] = int(BLOCK_TOKENS);
            diagnostics[4] = int(partial0[0]);
            diagnostics[5] = int(partial1[0]);
            diagnostics[6] = int(FALLBACK_CODE);
            diagnostics[7] = int(FLAGS);
            diagnostics[8] = int(partial2[0]);
            diagnostics[9] = int(partial3[0]);
            diagnostics[10] = int(partial4[0]);
            diagnostics[11] = int(partial5[0]);
            diagnostics[12] = int(partial6[0]);
            diagnostics[13] = int(partial7[0]);
            diagnostics[14] = 0;
            diagnostics[15] = 0;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_block_stats_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
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
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));

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
        uint score_index =
            ((row * uint(BLOCK_COUNT) + block_index) * uint(BLOCK_TOKENS)) + lane;
        score_tiles[score_index] = scaled_score;
        partial[lane] = scaled_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float tile_max = partial[0];
        float tile_weight = active ? exp(scaled_score - tile_max) : 0.0f;
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
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_gqa_block_stats_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
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
            return;
        }
        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
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
        uint token_slot = lane;

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
                    token_slot = quad_id + j * num_quads;
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
            uint q_head = kv_head * gqa_repeats + repeat;
            uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
            uint score_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(BLOCK_TOKENS)) + token_slot;
            score_tiles[score_index] = scaled_scores[repeat];
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

)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_gqa_block_stats_topk_candidates_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        threadgroup float partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint candidate_token_scratch[THREADS_PER_BLOCK];
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
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                if (lane == 0u) {
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
                for (uint rank = lane; rank < topk_limit; rank += threads_per_block) {
                    uint candidate_index =
                        ((row * uint(BLOCK_COUNT) + block_index) * topk_limit) + rank;
                    candidate_scores[candidate_index] = -INFINITY;
                    candidate_tokens[candidate_index] = -1;
                }
            }
            return;
        }

        ulong key_seed = tq_make_seed(uint(SEED_3), uint(SEED_2), uint(SEED_1), uint(SEED_0));
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

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        uint physical_token = 0u;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

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

        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            partial[score_base + lane] = tile_scores[score_base + lane];
            candidate_token_scratch[lane] = active ? logical_token : 0xffffffffu;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint width = 2u; width <= threads_per_block; width <<= 1) {
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    uint pair_lane = lane ^ stride;
                    float self_score = partial[score_base + lane];
                    uint self_token = candidate_token_scratch[lane];
                    float pair_score = partial[score_base + pair_lane];
                    uint pair_token = candidate_token_scratch[pair_lane];
                    bool self_before_pair = self_score > pair_score
                        || (self_score == pair_score && self_token < pair_token);
                    bool descending = (lane & width) == 0u;
                    bool swap = descending ? !self_before_pair : self_before_pair;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (pair_lane > lane && swap) {
                        partial[score_base + lane] = pair_score;
                        candidate_token_scratch[lane] = pair_token;
                        partial[score_base + pair_lane] = self_score;
                        candidate_token_scratch[pair_lane] = self_token;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            uint q_head = kv_head * gqa_repeats + repeat;
            uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
            for (uint rank = lane; rank < topk_limit; rank += threads_per_block) {
                uint candidate_index =
                    ((row * uint(BLOCK_COUNT) + block_index) * topk_limit) + rank;
                bool valid = rank < uint(BLOCK_TOKENS)
                    && candidate_token_scratch[rank] != 0xffffffffu;
                candidate_scores[candidate_index] =
                    valid ? partial[score_base + rank] : -INFINITY;
                candidate_tokens[candidate_index] =
                    valid ? int(candidate_token_scratch[rank]) : -1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            tile_maxes[repeat] = partial[score_base];
            float tile_weight = active
                ? exp(tile_scores[score_base + lane] - tile_maxes[repeat])
                : 0.0f;
            partial[score_base + lane] = tile_weight;
        }
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

    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_block_global_stats_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        float block_max = -INFINITY;
        for (uint block = lane; block < uint(BLOCK_COUNT); block += threads_per_block) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block) * 2u);
            block_max = max(block_max, partial_stats[stat_index]);
        }
        partial[lane] = block_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] = max(partial[lane], partial[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float global_max = partial[0];

        float block_sum = 0.0f;
        for (uint block = lane; block < uint(BLOCK_COUNT); block += threads_per_block) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block) * 2u);
            float tile_max = partial_stats[stat_index];
            float tile_sum = partial_stats[stat_index + 1u];
            block_sum += tile_sum > 0.0f ? exp(tile_max - global_max) * tile_sum : 0.0f;
        }
        partial[lane] = block_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            global_stats[row * 2u] = global_max;
            global_stats[row * 2u + 1u] = partial[0];
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_block_selection_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup uint count_partial[THREADS_PER_BLOCK];

        uint logical_length = uint(runtime_logical_length);
        uint selection_mode = uint(runtime_sparse_v_selection_mode);
        int top_k_value = int(runtime_sparse_v_top_k);
        int max_top_k_value = int(runtime_sparse_v_max_top_k);
        uint sparse_top_k = top_k_value > 0 ? uint(top_k_value) : 0u;
        uint sparse_max_top_k = max_top_k_value > 0 ? uint(max_top_k_value) : 0u;
        float cumulative_mass_target =
            clamp(float(runtime_sparse_v_cumulative_mass), 0.0f, 1.0f);
        float threshold = max(float(runtime_sparse_v_threshold), 0.0f);
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint total_slots = uint(BLOCK_COUNT) * uint(BLOCK_TOKENS);
        uint score_base = row * total_slots;

        if (global_sum <= 0.0f || logical_length == 0u) {
            if (lane == 0u) {
                selection_stats[row * 4u] = INFINITY;
                selection_stats[row * 4u + 1u] = 0.0f;
                selection_stats[row * 4u + 2u] = 0.0f;
                selection_stats[row * 4u + 3u] = 0.0f;
            }
            return;
        }

        float cutoff = threshold;
        if (selection_mode == 2u || selection_mode == 4u) {
            uint limit = selection_mode == 4u ? sparse_max_top_k : sparse_top_k;
            limit = min(limit, logical_length);
            if (limit == 0u) {
                cutoff = INFINITY;
            } else {
                float local_max_weight = 0.0f;
                for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                    uint logical_token = slot;
                    float scaled_score = score_tiles[score_base + slot];
                    bool active = logical_token < logical_length && scaled_score > -INFINITY;
                    float weight = active
                        ? exp(scaled_score - global_max) / global_sum
                        : 0.0f;
                    local_max_weight = max(local_max_weight, weight);
                }
                partial[lane] = local_max_weight;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] = max(partial[lane], partial[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                float topk_low = 0.0f;
                float topk_high = partial[0];
                if (topk_high <= 0.0f) {
                    cutoff = INFINITY;
                } else {
                    for (uint iter = 0u; iter < 24u; iter++) {
                        float mid = 0.5f * (topk_low + topk_high);
                        uint local_count = 0u;
                        for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                            uint logical_token = slot;
                            float scaled_score = score_tiles[score_base + slot];
                            bool active = logical_token < logical_length && scaled_score > -INFINITY;
                            float weight = active
                                ? exp(scaled_score - global_max) / global_sum
                                : 0.0f;
                            local_count += weight >= mid ? 1u : 0u;
                        }
                        count_partial[lane] = local_count;
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                            if (lane < stride) {
                                count_partial[lane] += count_partial[lane + stride];
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                        if (count_partial[0] >= limit) {
                            topk_low = mid;
                        } else {
                            topk_high = mid;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    cutoff = topk_low;
                }

                if (selection_mode == 4u) {
                    float mass_cutoff = 0.0f;
                    if (cumulative_mass_target <= 0.0f) {
                        mass_cutoff = INFINITY;
                    } else if (cumulative_mass_target < 1.0f && topk_high > 0.0f) {
                        float low = 0.0f;
                        float high = topk_high;
                        for (uint iter = 0u; iter < 24u; iter++) {
                            float mid = 0.5f * (low + high);
                            float local_mass = 0.0f;
                            for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                                uint logical_token = slot;
                                float scaled_score = score_tiles[score_base + slot];
                                bool active = logical_token < logical_length && scaled_score > -INFINITY;
                                float weight = active
                                    ? exp(scaled_score - global_max) / global_sum
                                    : 0.0f;
                                local_mass += weight >= mid ? weight : 0.0f;
                            }
                            partial[lane] = local_mass;
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                                if (lane < stride) {
                                    partial[lane] += partial[lane + stride];
                                }
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                            }
                            if (partial[0] >= cumulative_mass_target) {
                                low = mid;
                            } else {
                                high = mid;
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                        mass_cutoff = low;
                    }
                    cutoff = max(cutoff, mass_cutoff);
                }
            }
        } else if (selection_mode == 3u) {
            if (cumulative_mass_target <= 0.0f) {
                cutoff = INFINITY;
            } else if (cumulative_mass_target >= 1.0f) {
                cutoff = 0.0f;
            } else {
                float local_max_weight = 0.0f;
                for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                    uint logical_token = slot;
                    float scaled_score = score_tiles[score_base + slot];
                    bool active = logical_token < logical_length && scaled_score > -INFINITY;
                    float weight = active
                        ? exp(scaled_score - global_max) / global_sum
                        : 0.0f;
                    local_max_weight = max(local_max_weight, weight);
                }
                partial[lane] = local_max_weight;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane] = max(partial[lane], partial[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                float low = 0.0f;
                float high = partial[0];
                for (uint iter = 0u; iter < 24u; iter++) {
                    float mid = 0.5f * (low + high);
                    float local_mass = 0.0f;
                    for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                        uint logical_token = slot;
                        float scaled_score = score_tiles[score_base + slot];
                        bool active = logical_token < logical_length && scaled_score > -INFINITY;
                        float weight = active
                            ? exp(scaled_score - global_max) / global_sum
                            : 0.0f;
                        local_mass += weight >= mid ? weight : 0.0f;
                    }
                    partial[lane] = local_mass;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                        if (lane < stride) {
                            partial[lane] += partial[lane + stride];
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (partial[0] >= cumulative_mass_target) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                cutoff = low;
            }
        }

        float local_retained_mass = 0.0f;
        uint local_retained_count = 0u;
        uint local_total_count = 0u;
        for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
            uint logical_token = slot;
            float scaled_score = score_tiles[score_base + slot];
            bool active = logical_token < logical_length && scaled_score > -INFINITY;
            float weight = active ? exp(scaled_score - global_max) / global_sum : 0.0f;
            bool retained = active && weight >= cutoff;
            local_retained_mass += retained ? weight : 0.0f;
            local_retained_count += retained ? 1u : 0u;
            local_total_count += active ? 1u : 0u;
        }
        partial[lane] = local_retained_mass;
        count_partial[lane] = local_retained_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                partial[lane] += partial[lane + stride];
                count_partial[lane] += count_partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float retained_mass = partial[0];
        uint retained_count = count_partial[0];

        count_partial[lane] = local_total_count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                count_partial[lane] += count_partial[lane + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            selection_stats[row * 4u] = cutoff;
            selection_stats[row * 4u + 1u] = retained_mass;
            selection_stats[row * 4u + 2u] = float(retained_count);
            selection_stats[row * 4u + 3u] = float(count_partial[0]);
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_global_selection_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup uint token_partial[THREADS_PER_BLOCK];

        uint logical_length = uint(runtime_logical_length);
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint total_slots = uint(BLOCK_COUNT) * uint(BLOCK_TOKENS);
        uint score_base = row * total_slots;
        uint selected_count = min(topk_limit, logical_length);

        if (global_sum <= 0.0f || logical_length == 0u || topk_limit == 0u) {
            for (uint rank = lane; rank < topk_limit; rank += threads_per_block) {
                selected_tokens[row * topk_limit + rank] = -1;
            }
            if (lane == 0u) {
                selection_stats[row * 4u] = INFINITY;
                selection_stats[row * 4u + 1u] = 0.0f;
                selection_stats[row * 4u + 2u] = 0.0f;
                selection_stats[row * 4u + 3u] = 0.0f;
            }
            return;
        }

        float previous_score = INFINITY;
        uint previous_token = 0u;
        float retained_mass = 0.0f;
        float cutoff_score = -INFINITY;
        uint cutoff_token = 0xffffffffu;

        for (uint rank = 0u; rank < topk_limit; rank++) {
            float best_score = -INFINITY;
            uint best_token = 0xffffffffu;
            if (rank < selected_count) {
                for (uint slot = lane; slot < total_slots; slot += threads_per_block) {
                    uint logical_token = slot;
                    float scaled_score = score_tiles[score_base + slot];
                    bool active = logical_token < logical_length && scaled_score > -INFINITY;
                    bool after_previous = rank == 0u
                        || scaled_score < previous_score
                        || (scaled_score == previous_score && logical_token > previous_token);
                    bool better = scaled_score > best_score
                        || (scaled_score == best_score && logical_token < best_token);
                    if (active && after_previous && better) {
                        best_score = scaled_score;
                        best_token = logical_token;
                    }
                }
            }

            partial[lane] = best_score;
            token_partial[lane] = best_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    float other_score = partial[lane + stride];
                    uint other_token = token_partial[lane + stride];
                    bool other_better = other_score > partial[lane]
                        || (other_score == partial[lane] && other_token < token_partial[lane]);
                    if (other_better) {
                        partial[lane] = other_score;
                        token_partial[lane] = other_token;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float chosen_score = partial[0];
            uint chosen_token = token_partial[0];
            if (lane == 0u) {
                selected_tokens[row * topk_limit + rank] =
                    rank < selected_count ? int(chosen_token) : -1;
                if (rank < selected_count) {
                    retained_mass += exp(chosen_score - global_max) / global_sum;
                    cutoff_score = chosen_score;
                    cutoff_token = chosen_token;
                }
            }
            previous_score = chosen_score;
            previous_token = chosen_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            selection_stats[row * 4u] = cutoff_score;
            selection_stats[row * 4u + 1u] = retained_mass;
            selection_stats[row * 4u + 2u] = float(selected_count);
            selection_stats[row * 4u + 3u] = float(logical_length);
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_local_candidates_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup uint token_partial[THREADS_PER_BLOCK];

        uint logical_length = uint(runtime_logical_length);
        uint block_start = block_index * uint(BLOCK_TOKENS);
        uint logical_token = block_start + lane;
        uint score_index =
            ((row * uint(BLOCK_COUNT) + block_index) * uint(BLOCK_TOKENS)) + lane;
        bool active = lane < uint(BLOCK_TOKENS) && logical_token < logical_length;
        partial[lane] = active ? score_tiles[score_index] : -INFINITY;
        token_partial[lane] = active ? logical_token : 0xffffffffu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint width = 2u; width <= threads_per_block; width <<= 1) {
            for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                uint pair_lane = lane ^ stride;
                float self_score = partial[lane];
                uint self_token = token_partial[lane];
                float pair_score = partial[pair_lane];
                uint pair_token = token_partial[pair_lane];
                bool self_before_pair = self_score > pair_score
                    || (self_score == pair_score && self_token < pair_token);
                bool descending = (lane & width) == 0u;
                bool swap = descending ? !self_before_pair : self_before_pair;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (pair_lane > lane && swap) {
                    partial[lane] = pair_score;
                    token_partial[lane] = pair_token;
                    partial[pair_lane] = self_score;
                    token_partial[pair_lane] = self_token;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint rank = lane; rank < topk_limit; rank += threads_per_block) {
            uint out_index =
                ((row * uint(BLOCK_COUNT) + block_index) * topk_limit) + rank;
            bool valid = rank < uint(BLOCK_TOKENS) && rank < topk_limit
                && token_partial[rank] != 0xffffffffu;
            candidate_scores[out_index] = valid ? partial[rank] : -INFINITY;
            candidate_tokens[out_index] = valid ? int(token_partial[rank]) : -1;
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_compact_selection_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup uint token_partial[THREADS_PER_BLOCK];
        threadgroup uint block_partial[THREADS_PER_BLOCK];
        threadgroup uint block_offsets[BLOCK_COUNT];

        uint logical_length = uint(runtime_logical_length);
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint candidate_base = row * uint(BLOCK_COUNT) * topk_limit;
        uint selected_count = min(topk_limit, logical_length);

        if (global_sum <= 0.0f || logical_length == 0u || topk_limit == 0u) {
            for (uint rank = lane; rank < topk_limit; rank += threads_per_block) {
                selected_tokens[row * topk_limit + rank] = -1;
            }
            if (lane == 0u) {
                selection_stats[row * 4u] = INFINITY;
                selection_stats[row * 4u + 1u] = 0.0f;
                selection_stats[row * 4u + 2u] = 0.0f;
                selection_stats[row * 4u + 3u] = 0.0f;
            }
            return;
        }

        for (uint block = lane; block < uint(BLOCK_COUNT); block += threads_per_block) {
            block_offsets[block] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float retained_mass = 0.0f;
        float cutoff_score = -INFINITY;
        uint cutoff_token = 0xffffffffu;

        for (uint rank = 0u; rank < topk_limit; rank++) {
            float best_score = -INFINITY;
            uint best_token = 0xffffffffu;
            uint best_block = 0xffffffffu;
            if (rank < selected_count) {
                for (uint block = lane; block < uint(BLOCK_COUNT); block += threads_per_block) {
                    uint offset = block_offsets[block];
                    uint candidate = block * topk_limit + offset;
                    int token_value = candidate_tokens[candidate_base + candidate];
                    uint logical_token = token_value >= 0 ? uint(token_value) : 0xffffffffu;
                    float scaled_score = candidate_scores[candidate_base + candidate];
                    bool active = offset < topk_limit
                        && token_value >= 0
                        && logical_token < logical_length
                        && scaled_score > -INFINITY;
                    bool better = scaled_score > best_score
                        || (scaled_score == best_score && logical_token < best_token);
                    if (active && better) {
                        best_score = scaled_score;
                        best_token = logical_token;
                        best_block = block;
                    }
                }
            }

            partial[lane] = best_score;
            token_partial[lane] = best_token;
            block_partial[lane] = best_block;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    float other_score = partial[lane + stride];
                    uint other_token = token_partial[lane + stride];
                    bool other_better = other_score > partial[lane]
                        || (other_score == partial[lane] && other_token < token_partial[lane]);
                    if (other_better) {
                        partial[lane] = other_score;
                        token_partial[lane] = other_token;
                        block_partial[lane] = block_partial[lane + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float chosen_score = partial[0];
            uint chosen_token = token_partial[0];
            uint chosen_block = block_partial[0];
            if (lane == 0u) {
                selected_tokens[row * topk_limit + rank] =
                    rank < selected_count ? int(chosen_token) : -1;
                if (rank < selected_count) {
                    retained_mass += exp(chosen_score - global_max) / global_sum;
                    cutoff_score = chosen_score;
                    cutoff_token = chosen_token;
                    if (chosen_block != 0xffffffffu) {
                        block_offsets[chosen_block] += 1u;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            selection_stats[row * 4u] = cutoff_score;
            selection_stats[row * 4u + 1u] = retained_mass;
            selection_stats[row * 4u + 2u] = float(selected_count);
            selection_stats[row * 4u + 3u] = float(logical_length);
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_global_compact_output_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_ROW];
        threadgroup uint token_partial[THREADS_PER_ROW];
        threadgroup float selected_scores[TOPK_LIMIT];
        threadgroup uint selected_tokens_tg[TOPK_LIMIT];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint total_slots = uint(BLOCK_COUNT) * uint(BLOCK_TOKENS);
        uint score_base = row * total_slots;
        uint selected_count = min(topk_limit, logical_length);
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (global_sum <= 0.0f || logical_length == 0u || topk_limit == 0u) {
            if (lane < uint(HEAD_DIM)) {
                out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(0.0f);
            }
            if (output_sparse_stats && lane == 0u) {
                uint stat_index = row * 2u;
                sparse_stats[stat_index] = 0u;
                sparse_stats[stat_index + 1u] = logical_length;
            }
            return;
        }

        float previous_score = INFINITY;
        uint previous_token = 0u;
        for (uint rank = 0u; rank < topk_limit; rank++) {
            float best_score = -INFINITY;
            uint best_token = 0xffffffffu;
            if (rank < selected_count) {
                for (uint slot = lane; slot < total_slots; slot += threads_per_row) {
                    uint logical_token = slot;
                    float scaled_score = score_tiles[score_base + slot];
                    bool active = logical_token < logical_length && scaled_score > -INFINITY;
                    bool after_previous = rank == 0u
                        || scaled_score < previous_score
                        || (scaled_score == previous_score && logical_token > previous_token);
                    bool better = scaled_score > best_score
                        || (scaled_score == best_score && logical_token < best_token);
                    if (active && after_previous && better) {
                        best_score = scaled_score;
                        best_token = logical_token;
                    }
                }
            }

            partial[lane] = best_score;
            token_partial[lane] = best_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    float other_score = partial[lane + stride];
                    uint other_token = token_partial[lane + stride];
                    bool other_better = other_score > partial[lane]
                        || (other_score == partial[lane] && other_token < token_partial[lane]);
                    if (other_better) {
                        partial[lane] = other_score;
                        token_partial[lane] = other_token;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float chosen_score = partial[0];
            uint chosen_token = token_partial[0];
            if (lane == 0u) {
                selected_scores[rank] = chosen_score;
                selected_tokens_tg[rank] = rank < selected_count ? chosen_token : 0xffffffffu;
            }
            previous_score = chosen_score;
            previous_token = chosen_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            for (uint rank = 0u; rank < topk_limit; rank++) {
                uint logical_token = selected_tokens_tg[rank];
                if (rank < selected_count && logical_token != 0xffffffffu) {
                    uint physical_token = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    float weight = exp(selected_scores[rank] - global_max) / global_sum;
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, physical_token, lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    dimension_accum += weight * value;
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }

        if (output_sparse_stats && lane == 0u) {
            uint stat_index = row * 2u;
            sparse_stats[stat_index] = logical_length > selected_count
                ? logical_length - selected_count
                : 0u;
            sparse_stats[stat_index + 1u] = logical_length;
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_candidate_compact_output_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_ROW];
        threadgroup uint token_partial[THREADS_PER_ROW];
        threadgroup float selected_scores[TOPK_LIMIT];
        threadgroup uint selected_tokens_tg[TOPK_LIMIT];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint total_candidates = uint(BLOCK_COUNT) * topk_limit;
        uint candidate_base = row * total_candidates;
        uint selected_count = min(topk_limit, logical_length);
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (global_sum <= 0.0f || logical_length == 0u || topk_limit == 0u) {
            if (lane < uint(HEAD_DIM)) {
                out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(0.0f);
            }
            if (output_sparse_stats && lane == 0u) {
                uint stat_index = row * 2u;
                sparse_stats[stat_index] = 0u;
                sparse_stats[stat_index + 1u] = logical_length;
            }
            return;
        }

        float previous_score = INFINITY;
        uint previous_token = 0u;
        for (uint rank = 0u; rank < topk_limit; rank++) {
            float best_score = -INFINITY;
            uint best_token = 0xffffffffu;
            if (rank < selected_count) {
                for (uint candidate = lane; candidate < total_candidates; candidate += threads_per_row) {
                    int token_value = candidate_tokens[candidate_base + candidate];
                    uint logical_token = token_value >= 0 ? uint(token_value) : 0xffffffffu;
                    float scaled_score = candidate_scores[candidate_base + candidate];
                    bool active = token_value >= 0
                        && logical_token < logical_length
                        && scaled_score > -INFINITY;
                    bool after_previous = rank == 0u
                        || scaled_score < previous_score
                        || (scaled_score == previous_score && logical_token > previous_token);
                    bool better = scaled_score > best_score
                        || (scaled_score == best_score && logical_token < best_token);
                    if (active && after_previous && better) {
                        best_score = scaled_score;
                        best_token = logical_token;
                    }
                }
            }

            partial[lane] = best_score;
            token_partial[lane] = best_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_row >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    float other_score = partial[lane + stride];
                    uint other_token = token_partial[lane + stride];
                    bool other_better = other_score > partial[lane]
                        || (other_score == partial[lane] && other_token < token_partial[lane]);
                    if (other_better) {
                        partial[lane] = other_score;
                        token_partial[lane] = other_token;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float chosen_score = partial[0];
            uint chosen_token = token_partial[0];
            if (lane == 0u) {
                selected_scores[rank] = chosen_score;
                selected_tokens_tg[rank] = rank < selected_count ? chosen_token : 0xffffffffu;
            }
            previous_score = chosen_score;
            previous_token = chosen_token;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            for (uint rank = 0u; rank < topk_limit; rank++) {
                uint logical_token = selected_tokens_tg[rank];
                if (rank < selected_count && logical_token != 0xffffffffu) {
                    uint physical_token = tq_physical_token(
                        logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                    float weight = exp(selected_scores[rank] - global_max) / global_sum;
                    float value = tq_decode_attention_value(
                        v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                        batch, kv_head, physical_token, lane,
                        value_seed, 1u,
                        uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                        uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                        uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                        uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                        decode_scratch);
                    dimension_accum += weight * value;
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }

        if (output_sparse_stats && lane == 0u) {
            uint stat_index = row * 2u;
            sparse_stats[stat_index] = logical_length > selected_count
                ? logical_length - selected_count
                : 0u;
            sparse_stats[stat_index + 1u] = logical_length;
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_topk_compact_output_source = R"TQMLX(        constexpr uint threads_per_row = uint(THREADS_PER_ROW);
        constexpr uint topk_limit = uint(TOPK_LIMIT);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        uint total_slots = uint(BLOCK_COUNT) * uint(BLOCK_TOKENS);
        uint score_base = row * total_slots;
        uint selected_count = min(topk_limit, logical_length);
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        if (lane < uint(HEAD_DIM)) {
            thread float decode_scratch[GROUP_SIZE];
            float dimension_accum = 0.0f;
            if (global_sum > 0.0f) {
                for (uint rank = 0u; rank < topk_limit; rank++) {
                    int token_value = selected_tokens[row * topk_limit + rank];
                    if (rank < selected_count && token_value >= 0) {
                        uint logical_token = uint(token_value);
                        uint physical_token = tq_physical_token(
                            logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
                        float scaled_score = score_tiles[score_base + logical_token];
                        float weight = exp(scaled_score - global_max) / global_sum;
                        float value = tq_decode_attention_value(
                            v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                            batch, kv_head, physical_token, lane,
                            value_seed, 1u,
                            uint(GROUP_SIZE), uint(KV_HEADS), uint(CAPACITY), uint(GROUPS_PER_VECTOR),
                            uint(VALUE_MAG_WORDS_PER_GROUP), uint(BITSET_WORDS_PER_GROUP), uint(BASE_BITS), uint(HIGH_BITS),
                            uint(VALUE_BITS), uint(KEY_BASE_BITS), uint(KEY_HIGH_BITS),
                            uint(LAYOUT_VERSION), uint(HEAD_DIM), 0u,
                            decode_scratch);
                        dimension_accum += weight * value;
                    }
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }

        if (output_sparse_stats && lane == 0u) {
            uint stat_index = row * 2u;
            sparse_stats[stat_index] = logical_length > selected_count
                ? logical_length - selected_count
                : 0u;
            sparse_stats[stat_index + 1u] = logical_length;
        }
    )TQMLX";

inline constexpr std::string_view turbo_quant_sparse_block_partials_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint row = group_index / uint(BLOCK_COUNT);
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        threadgroup float partial[THREADS_PER_BLOCK];
        threadgroup uint count_partial[THREADS_PER_BLOCK];
        threadgroup float tile_scores[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        float sparse_v_threshold = float(runtime_sparse_v_threshold);
        uint selection_mode = uint(runtime_sparse_v_selection_mode);
        bool block_threshold_mode = selection_mode == 5u;
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        float global_max = global_stats[row * 2u];
        float global_sum = global_stats[row * 2u + 1u];
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane < uint(HEAD_DIM)) {
                uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
            }
            if (output_sparse_stats && lane == 0u) {
                uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                sparse_stats[stat_index] = 0u;
                sparse_stats[stat_index + 1u] = 0u;
            }
            return;
        }
        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        float scaled_score = -INFINITY;
        uint physical_token = 0u;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            uint score_index =
                ((row * uint(BLOCK_COUNT) + block_index) * uint(BLOCK_TOKENS)) + lane;
            scaled_score = score_tiles[score_index];
        }

        float final_weight =
            active && global_sum > 0.0f ? exp(scaled_score - global_max) / global_sum : 0.0f;
        float block_mass = 0.0f;
        if (block_threshold_mode) {
            partial[lane] = final_weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    partial[lane] += partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            block_mass = partial[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        bool skipped = active && (block_threshold_mode
            ? block_mass < sparse_v_threshold
            : final_weight < sparse_v_threshold);
        tile_scores[lane] = skipped ? 0.0f : final_weight;
        tile_physical_tokens[lane] = physical_token;
        uint skipped_count = 0u;
        uint total_count = 0u;
        if (output_sparse_stats) {
            count_partial[lane] = skipped ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    count_partial[lane] += count_partial[lane + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            skipped_count = count_partial[0];
            uint block_end = min(block_start + uint(BLOCK_TOKENS), logical_length);
            uint visible_end = DO_CAUSAL ? min(block_end, causal_limit + 1u) : block_end;
            total_count = visible_end > block_start ? visible_end - block_start : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
            partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }
        if (output_sparse_stats && lane == 0u) {
            uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
            sparse_stats[stat_index] = skipped_count;
            sparse_stats[stat_index + 1u] = total_count;
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_gqa_block_partials_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        constexpr bool output_sparse_stats = bool(OUTPUT_SPARSE_STATS);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        constexpr uint repeat_count = uint(GQA_REPEATS) < 4u ? uint(GQA_REPEATS) : 4u;
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % uint(BLOCK_COUNT);
        uint gqa_row = group_index / uint(BLOCK_COUNT);
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        threadgroup uint count_partial[4 * THREADS_PER_BLOCK];
        threadgroup float mass_partial[4 * THREADS_PER_BLOCK];
        threadgroup float tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight[THREADS_PER_BLOCK];
        threadgroup uint tile_physical_tokens[THREADS_PER_BLOCK];
        threadgroup float query_cache[4 * HEAD_DIM];

        uint logical_length = uint(runtime_logical_length);
        uint ring_offset = uint(runtime_ring_offset);
        uint pinned_prefix_length = uint(runtime_pinned_prefix_length);
        float attention_scale = float(runtime_attention_scale);
        float sparse_v_threshold = float(runtime_sparse_v_threshold);
        uint selection_mode = uint(runtime_sparse_v_selection_mode);
        bool block_threshold_mode = selection_mode == 5u;
        uint q_token = gqa_row % uint(QUERY_LENGTH);
        uint kv_head = (gqa_row / uint(QUERY_LENGTH)) % uint(KV_HEADS);
        uint batch = gqa_row / (uint(QUERY_LENGTH) * uint(KV_HEADS));
        uint causal_limit = logical_length - uint(QUERY_LENGTH) + q_token;
        uint block_start = block_index * uint(BLOCK_TOKENS);
        if (DO_CAUSAL && block_start > causal_limit) {
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * uint(BLOCK_COUNT) + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
                }
            }
            if (output_sparse_stats && lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    sparse_stats[stat_index] = 0u;
                    sparse_stats[stat_index + 1u] = 0u;
                }
            }
            return;
        }

        ulong value_seed = tq_make_seed(
            uint(VALUE_SEED_3), uint(VALUE_SEED_2),
            uint(VALUE_SEED_1), uint(VALUE_SEED_0));

        uint logical_token = block_start + lane;
        bool active = lane < uint(BLOCK_TOKENS)
            && logical_token < logical_length
            && (!DO_CAUSAL || logical_token <= causal_limit);
        uint physical_token = 0u;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;
        if (active) {
            physical_token = tq_physical_token(
                logical_token, uint(CAPACITY), ring_offset, pinned_prefix_length);
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint score_index =
                    ((row * uint(BLOCK_COUNT) + block_index) * uint(BLOCK_TOKENS)) + lane;
                scaled_scores[repeat] = score_tiles[score_index];
            }
        }

        tile_physical_tokens[lane] = physical_token;
        uint has_weight = 0u;
        uint skipped_counts[4];
        skipped_counts[0] = 0u;
        skipped_counts[1] = 0u;
        skipped_counts[2] = 0u;
        skipped_counts[3] = 0u;
        uint block_end = min(block_start + uint(BLOCK_TOKENS), logical_length);
        uint visible_end = DO_CAUSAL ? min(block_end, causal_limit + 1u) : block_end;
        uint total_count = visible_end > block_start ? visible_end - block_start : 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint q_head = kv_head * gqa_repeats + repeat;
            uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
            float global_max = global_stats[row * 2u];
            float global_sum = global_stats[row * 2u + 1u];
            float final_weight =
                active && global_sum > 0.0f
                ? exp(scaled_scores[repeat] - global_max) / global_sum
                : 0.0f;
            uint base = repeat * threads_per_block;
            tile_scores[base + lane] = final_weight;
            if (block_threshold_mode) {
                mass_partial[base + lane] = final_weight;
            }
        }
        if (block_threshold_mode) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        uint base = repeat * threads_per_block;
                        mass_partial[base + lane] += mass_partial[base + lane + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint base = repeat * threads_per_block;
            uint q_head = kv_head * gqa_repeats + repeat;
            uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
            float final_weight = tile_scores[base + lane];
            float block_mass = block_threshold_mode ? mass_partial[base] : 0.0f;
            bool skipped = active && (block_threshold_mode
                ? block_mass < sparse_v_threshold
                : final_weight < sparse_v_threshold);
            float retained_weight = skipped ? 0.0f : final_weight;
            tile_scores[base + lane] = retained_weight;
            if (retained_weight > 0.0f) {
                has_weight = 1u;
            }
            skipped_counts[repeat] = skipped ? 1u : 0u;
        }
        tile_has_weight[lane] = has_weight;

        if (output_sparse_stats) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                count_partial[repeat * threads_per_block + lane] = skipped_counts[repeat];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
                if (lane < stride) {
                    for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                        uint base = repeat * threads_per_block;
                        count_partial[base + lane] += count_partial[base + lane + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint stat_index = ((row * uint(BLOCK_COUNT) + block_index) * 2u);
                    sparse_stats[stat_index] = count_partial[repeat * threads_per_block];
                    sparse_stats[stat_index + 1u] = total_count;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_sparse_block_sum_reduce_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }
        if (lane < uint(HEAD_DIM)) {
            float sum = 0.0f;
            for (uint block = 0u; block < uint(BLOCK_COUNT); block++) {
                uint index = ((row * uint(BLOCK_COUNT) + block) * uint(HEAD_DIM)) + lane;
                sum += float(partial_out[index]);
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(sum);
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_block_partials_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint row = group_index / block_count;
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
                uint stat_index = ((row * block_count + block_index) * 2u);
                partial_stats[stat_index] = -INFINITY;
                partial_stats[stat_index + 1u] = 0.0f;
            }
            if (lane < uint(HEAD_DIM)) {
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
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
            uint stat_index = ((row * block_count + block_index) * 2u);
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
            uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
            partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum);
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_gqa_block_partials_rf1_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
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
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
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
                        // COOPW invariant: rows [repeat_count, 4) of query_cache are
                        // NEVER initialized by the shared prologue (init loop and rotation
                        // are both bounded by repeat_count). Clamping every coop r-loop to
                        // repeat_count is load-bearing: the hardcoded r<4u form read those
                        // uninitialized threadgroup rows (UB, shader-validation trap risk).
                        // At repeat_count==4 the bound is unchanged -> byte-identical.
                        for (uint r = 0u; r < repeat_count; r++) {
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
                    for (uint r = 0u; r < repeat_count; r++) {
                        ts[r] = norm * qd[r] + residual_scale * sd[r];
                    }
                }
                for (uint r = 0u; r < repeat_count; r++) {
                    ts[r] += simd_shuffle_xor(ts[r], 1u);
                    ts[r] += simd_shuffle_xor(ts[r], 2u);
                }
                if (lane_in_quad == j) {
                    active = tok_active;
                    my_token = tok_active ? phys : 0u;
                    for (uint r = 0u; r < repeat_count; r++) {
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
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = partial[repeat * threads_per_block];
        }
        // Every lane must finish reading the reduced maxes above BEFORE any lane
        // overwrites partial[] with exp-weights below. Without this barrier, a lagging
        // simdgroup can read lane 0's already-written weight as the "max" for a later
        // repeat (drift across the barrier-free repeat loop is widest at the last
        // repeat), which intermittently corrupts the whole output row of one GQA
        // repeat at BT>=512 / 131072 (256 active blocks). Root-caused 2026-07-06 as
        // the same class as the v7 quad race; see
        // artifacts/turboquant-w2-regrad-20260706 and
        // artifacts/turboquant-v7-20260703/quad-nondeterminism.md.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
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
                uint stat_index = ((row * block_count + block_index) * 2u);
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
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_gqa_block_partials_h16_rf1_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
        uint total_gqa_rows = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(QUERY_LENGTH);
        if (gqa_row >= total_gqa_rows) {
            return;
        }

        // T2.2 H16 diet: partial/tile_scores staged as half (was float) and
        // tile_has_weight folded into a 32-lane bitset (was uint[THREADS_PER_BLOCK]).
        // This halves the two largest tgmem arrays and shrinks the mask array from
        // THREADS_PER_BLOCK*4B to (THREADS_PER_BLOCK/32)*4B, clearing the < 16384 B
        // static-tgmem boundary needed for 2 threadgroups/core (see G5 probe). All
        // reduction math still runs in fp32 registers; only the threadgroup-memory
        // storage dtype changes. query_cache and tile_physical_tokens are unchanged
        // (rotation precision / full 32-bit token indices at 131K).
        threadgroup half partial[4 * THREADS_PER_BLOCK];
        threadgroup half tile_scores[4 * THREADS_PER_BLOCK];
        threadgroup uint tile_has_weight_bits[(THREADS_PER_BLOCK + 31u) / 32u];
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
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
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
            // H16 overflow guard: the raw pre-softmax logit is unbounded and is
            // stored as half before the max-subtraction. Clamp to a safe fp16
            // sub-max range; -INFINITY (masked/inactive lanes) maps to -65504,
            // which the exp(x - max) step drives to 0 identically to -inf.
            float raw = scaled_scores[repeat];
            float guarded = raw;
            if (!(raw <= 60000.0f)) {
                guarded = (raw == -INFINITY) ? -65504.0f : 60000.0f;
            } else if (raw < -65504.0f) {
                guarded = -65504.0f;
            }
            tile_scores[score_base + lane] = half(guarded);
            partial[score_base + lane] = half(guarded);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    float a = float(partial[score_base + lane]);
                    float b = float(partial[score_base + lane + stride]);
                    partial[score_base + lane] = half(max(a, b));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        thread float tile_maxes[4];
        tile_maxes[0] = -INFINITY;
        tile_maxes[1] = -INFINITY;
        tile_maxes[2] = -INFINITY;
        tile_maxes[3] = -INFINITY;
        // Clear this lane's bitset word once (only the 32 lane-0-of-word threads).
        if ((lane & 31u) == 0u) { tile_has_weight_bits[lane >> 5] = 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = float(partial[repeat * threads_per_block]);
        }
        // Read the reduced maxes above BEFORE any lane overwrites partial[] with
        // exp-weights below (same v6-family RAW race fixed in the fp32 GQA kernel;
        // see artifacts/turboquant-w2-regrad-20260706).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
            float tile_weight = active
                ? exp(float(tile_scores[score_base + lane]) - tile_maxes[repeat])
                : 0.0f;
            tile_scores[score_base + lane] = half(tile_weight);
            partial[score_base + lane] = half(tile_weight);
            if (tile_weight > 0.0f) {
                has_weight = 1u;
            }
        }
        if (has_weight != 0u) {
            atomic_fetch_or_explicit(
                (threadgroup atomic_uint*)&tile_has_weight_bits[lane >> 5],
                1u << (lane & 31u), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_block >> 1; stride > 0u; stride >>= 1) {
            if (lane < stride) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint score_base = repeat * threads_per_block;
                    float s = float(partial[score_base + lane])
                            + float(partial[score_base + lane + stride]);
                    partial[score_base + lane] = half(s);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint stat_index = ((row * block_count + block_index) * 2u);
                uint score_base = repeat * threads_per_block;
                partial_stats[stat_index] = tile_maxes[repeat];
                partial_stats[stat_index + 1u] = float(partial[score_base]);
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
                if ((tile_has_weight_bits[tile_lane >> 5] & (1u << (tile_lane & 31u))) != 0u) {
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
                            float(tile_scores[repeat * threads_per_block + tile_lane]) * value;
                    }
                }
            }

            for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                uint q_head = kv_head * gqa_repeats + repeat;
                uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_gqa_block_partials_v7_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        constexpr uint gqa_repeats = uint(GQA_REPEATS);
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint block_index = group_index % block_count;
        uint gqa_row = group_index / block_count;
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
                    uint stat_index = ((row * block_count + block_index) * 2u);
                    partial_stats[stat_index] = -INFINITY;
                    partial_stats[stat_index + 1u] = 0.0f;
                }
            }
            if (lane < uint(HEAD_DIM)) {
                for (uint repeat = 0u; repeat < repeat_count; repeat++) {
                    uint q_head = kv_head * gqa_repeats + repeat;
                    uint row = ((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH)) + q_token;
                    uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                    partial_out[out_index] = static_cast<OUTPUT_DTYPE>(0.0f);
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

        uint physical_token = 0u;
        bool active = false;
        thread float scaled_scores[4];
        scaled_scores[0] = -INFINITY;
        scaled_scores[1] = -INFINITY;
        scaled_scores[2] = -INFINITY;
        scaled_scores[3] = -INFINITY;

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
                    tq_product_attention_inner_product_group_quad_v7(
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
                    tq_product_attention_inner_product_group_pair_v7(
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
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            tile_maxes[repeat] = partial[repeat * threads_per_block];
        }
        // Every lane must finish reading the reduced maxes above BEFORE any lane
        // overwrites partial[] with exp-weights below. Without this barrier, a lagging
        // simdgroup can read lane 0's already-written weight as the "max" for a later
        // repeat (the drift across the barrier-free repeat loop is widest at the last
        // repeat), which intermittently corrupted the whole output row of the last GQA
        // repeat (q_head = kv_head*4+3) at a 5-25% per-dispatch rate. Root-caused
        // 2026-07-03; see artifacts/turboquant-v7-20260703/quad-nondeterminism.md.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint has_weight = 0u;
        for (uint repeat = 0u; repeat < repeat_count; repeat++) {
            uint score_base = repeat * threads_per_block;
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
                uint stat_index = ((row * block_count + block_index) * 2u);
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
                uint out_index = ((row * block_count + block_index) * uint(HEAD_DIM)) + lane;
                partial_out[out_index] = static_cast<OUTPUT_DTYPE>(dimension_accum[repeat]);
            }
        }
)TQMLX";

inline constexpr std::string_view turbo_quant_block_reduce_source = R"TQMLX(        constexpr uint threads_per_block = uint(THREADS_PER_BLOCK);
        uint block_count = uint(runtime_block_count);
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        if (row >= uint(ROW_COUNT)) {
            return;
        }

        threadgroup float partial[512];
        threadgroup float tile_scales[512];

        if (lane < block_count) {
            partial[lane] = partial_stats[(row * block_count + lane) * 2u];
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
        if (lane < block_count) {
            uint stat_index = (row * block_count + lane) * 2u;
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
            for (uint block = 0u; block < block_count; block++) {
                float tile_scale = tile_scales[block];
                if (tile_scale > 0.0f) {
                    uint partial_index = ((row * block_count + block) * uint(HEAD_DIM)) + lane;
                    accum += tile_scale * partial_out[partial_index];
                }
            }
            out[row * uint(HEAD_DIM) + lane] = static_cast<OUTPUT_DTYPE>(
                accum / max(row_sum, 1.17549435e-38f));
        }
)TQMLX";

} // namespace mlx::core::fast::turbo_quant_detail
