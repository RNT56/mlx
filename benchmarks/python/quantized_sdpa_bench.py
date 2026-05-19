# Copyright © 2026 Apple Inc.

import argparse
import math
import time

import mlx.core as mx


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        mx.eval(fn())

    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return 1e3 * (time.perf_counter() - start) / iters


def prepare_inputs(B, Hq, Hkv, Lq, Lk, D, dtype, mode):
    q = (0.1 * mx.random.normal(shape=(B, Hq, Lq, D))).astype(dtype)
    k = (0.1 * mx.random.normal(shape=(B, Hkv, Lk, D))).astype(dtype)
    v = (0.1 * mx.random.normal(shape=(B, Hkv, Lk, D))).astype(dtype)
    k_q, k_scales = mx.quantize(k, mode=mode)
    v_q, v_scales = mx.quantize(v, mode=mode)
    mx.eval(q, k, v, k_q, k_scales, v_q, v_scales)
    return q, k, v, k_q, k_scales, v_q, v_scales


def dense_sdpa(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def quantized_sdpa(q, k_q, k_scales, v_q, v_scales, scale, mode):
    bits = 8 if mode == "mxfp8" else 4
    return mx.fast.quantized_scaled_dot_product_attention(
        q,
        k_q,
        k_scales,
        v_q,
        v_scales,
        scale=scale,
        mode=mode,
        bits=bits,
    )


def run_case(shape, dtype, mode, iters):
    B, Hq, Hkv, Lq, Lk, D = shape
    scale = 1.0 / math.sqrt(D)
    q, k, v, k_q, k_scales, v_q, v_scales = prepare_inputs(
        B, Hq, Hkv, Lq, Lk, D, dtype, mode
    )

    dense_ms = bench(lambda: dense_sdpa(q, k, v, scale), iters=iters)
    quant_ms = bench(
        lambda: quantized_sdpa(q, k_q, k_scales, v_q, v_scales, scale, mode),
        iters=iters,
    )

    print(
        f"{B:2d}, {Hq:2d}, {Hkv:2d}, {Lq:2d}, {Lk:5d}, {D:3d}, "
        f"{str(dtype):>9s}, {mode:>5s}, {dense_ms:8.4f}, {quant_ms:8.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantized SDPA benchmarks")
    parser.add_argument("--mode", choices=["mxfp4", "mxfp8"], default="mxfp4")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    dtype = getattr(mx, args.dtype)
    mx.random.seed(0)

    shapes = [
        # B, Hq, Hkv, Lq, Lk, D
        (1, 8, 1, 1, 4096, 128),
        (1, 8, 1, 8, 4096, 128),
        (1, 8, 1, 9, 4096, 128),
        (1, 8, 1, 32, 4096, 128),
        (1, 8, 1, 32, 4096, 512),
    ]
    if args.quick:
        shapes = [(1, 8, 1, 9, 512, 128)]

    print(" B, Hq, Hk, Lq,    Lk,   D,     dtype,  mode, dense_ms, quant_ms")
    for shape in shapes:
        run_case(shape, dtype, args.mode, args.iters)
