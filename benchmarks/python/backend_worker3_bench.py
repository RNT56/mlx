# Copyright (c) 2026 Apple Inc.

import argparse
import json
import platform
import time

import mlx.core as mx


def eval_result(result):
    if isinstance(result, tuple):
        mx.eval(*result)
    else:
        mx.eval(result)
    mx.synchronize()


def bench(fn, warmup, iters):
    for _ in range(warmup):
        eval_result(fn())
    start = time.perf_counter()
    for _ in range(iters):
        eval_result(fn())
    return (time.perf_counter() - start) * 1000.0 / iters


def device_record(backend):
    info = {}
    try:
        info = dict(mx.device_info())
    except Exception:
        pass
    return {
        "backend": backend,
        "device": str(mx.default_device()),
        "platform": platform.platform(),
        "device_info": info,
    }


def record(name, backend, dtype, shape, baseline_fn, optimized_fn, args):
    try:
        baseline_ms = bench(baseline_fn, args.warmup, args.iters)
        optimized_ms = bench(optimized_fn, args.warmup, args.iters)
        return {
            "name": name,
            **device_record(backend),
            "dtype": dtype,
            "shape": shape,
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup": baseline_ms / optimized_ms if optimized_ms else None,
        }
    except Exception as exc:
        return {
            "name": name,
            **device_record(backend),
            "dtype": dtype,
            "shape": shape,
            "skipped": True,
            "reason": str(exc),
        }


def einsum_case(args):
    a = mx.random.normal((64, 8), dtype=mx.float32)
    b = mx.random.normal((8, 1024), dtype=mx.float32)
    c = mx.random.normal((1024, 64), dtype=mx.float32)
    mx.eval(a, b, c)
    return record(
        "einsum_contraction_order",
        "any",
        "float32",
        {"a": a.shape, "b": b.shape, "c": c.shape, "equation": "ij,jk,kl->il"},
        lambda: (a @ b) @ c,
        lambda: mx.einsum("ij,jk,kl->il", a, b, c),
        args,
    )


def cpu_scan_case(args):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x = mx.random.normal((128, 512), dtype=mx.float32)
        tri = mx.tri(512, dtype=mx.float32)
        mx.eval(x, tri)
        return record(
            "cpu_scan_cumsum",
            "cpu",
            "float32",
            {"x": x.shape, "axis": -1},
            lambda: x @ tri.T,
            lambda: mx.cumsum(x, axis=-1),
            args,
        )
    finally:
        mx.set_default_device(old)


def cpu_cholesky_case(args):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        x = mx.random.normal((32, 32), dtype=mx.float32)
        a = x @ x.T + mx.eye(32) * 1e-3
        mx.eval(a)
        return record(
            "cpu_cholesky_psd",
            "cpu",
            "float32",
            {"a": a.shape},
            lambda: mx.linalg.cholesky(mx.contiguous(a)),
            lambda: mx.linalg.cholesky(a),
            args,
        )
    finally:
        mx.set_default_device(old)


def reduce_case(args):
    x = mx.random.normal((256, 512), dtype=mx.float32)[:, ::2]
    mx.eval(x)
    return record(
        "gpu_reduce_strided",
        "gpu",
        "float32",
        {"x": x.shape, "axis": 0, "strides": x.strides},
        lambda: mx.sum(mx.contiguous(x), axis=0),
        lambda: mx.sum(x, axis=0),
        args,
    )


def grouped_conv_case(args):
    x = mx.random.normal((8, 32, 32, 32), dtype=mx.float32)
    w = mx.random.normal((64, 3, 3, 8), dtype=mx.float32)
    groups = 4
    xs = mx.split(x, groups, axis=-1)
    ws = mx.split(w, groups, axis=0)
    mx.eval(x, w)
    return record(
        "cuda_grouped_conv2d",
        "cuda",
        "float32",
        {"x": x.shape, "w": w.shape, "groups": groups},
        lambda: mx.concatenate(
            [mx.conv2d(xg, wg, padding=1) for xg, wg in zip(xs, ws)], axis=-1
        ),
        lambda: mx.conv2d(x, w, padding=1, groups=groups),
        args,
    )


def qmm_case(args):
    x = mx.random.normal((128, 128), dtype=mx.float16)
    w = mx.random.normal((256, 128), dtype=mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=128, bits=4)
    mx.eval(x, wq, scales, biases)

    def baseline():
        scale_setup = mx.contiguous(mx.swapaxes(scales, -1, -2))
        bias_setup = mx.contiguous(mx.swapaxes(biases, -1, -2))
        y = mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=128, bits=4
        )
        return y, scale_setup, bias_setup

    return record(
        "cuda_qmm_setup_copy",
        "cuda",
        "float16",
        {"x": x.shape, "wq": wq.shape, "scales": scales.shape},
        baseline,
        lambda: mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=128, bits=4
        ),
        args,
    )


def fft_case(args):
    x = mx.random.normal((64, 4096), dtype=mx.float32).astype(mx.complex64)
    mx.eval(x)
    return record(
        "metal_fft_donation",
        "metal",
        "complex64",
        {"x": x.shape, "axis": -1},
        lambda: mx.fft.fft(x + mx.array(0, dtype=mx.complex64), axis=-1),
        lambda: mx.fft.fft(mx.negative(mx.negative(x)), axis=-1),
        args,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Emit Worker 3 backend before/after benchmark JSON."
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--case", choices=[
        "all",
        "einsum",
        "cpu-scan",
        "cpu-cholesky",
        "reduce",
        "grouped-conv",
        "qmm",
        "fft",
    ], default="all")
    args = parser.parse_args()

    cases = {
        "einsum": einsum_case,
        "cpu-scan": cpu_scan_case,
        "cpu-cholesky": cpu_cholesky_case,
        "reduce": reduce_case,
        "grouped-conv": grouped_conv_case,
        "qmm": qmm_case,
        "fft": fft_case,
    }
    selected = cases.keys() if args.case == "all" else [args.case]
    results = [cases[name](args) for name in selected]
    print(json.dumps({"schema": "mlx.worker3.backend_bench.v1", "results": results}, indent=2))


if __name__ == "__main__":
    main()
