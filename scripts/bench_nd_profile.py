#!/usr/bin/env python3
"""
Benchmark ND vs specialized paths with configurable autotune/thread-cap settings
and capture detailed ND profiling data.

Usage examples:
  ./scripts/bench_nd_profile.py --parallels 4,8,16 --output benchmarks/nd_profile_runs.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import edt  # noqa: E402


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def default_shapes(dims: Sequence[int]) -> List[Tuple[int, ...]]:
    shapes: List[Tuple[int, ...]] = []
    if 2 in dims:
        shapes.extend([(128, 128), (256, 256), (512, 512), (1024, 1024)])
    if 3 in dims:
        shapes.extend([
            (48, 48, 48),
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128),
            (256, 256, 256),
            (384, 384, 384),
            (512, 512, 512),
        ])
    return shapes


def make_array(rng: np.random.Generator, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    arr = rng.integers(0, 3, size=shape, dtype=dtype)
    if arr.ndim == 2:
        y, x = shape
        if y > 20 and x > 20:
            arr[y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    elif arr.ndim == 3:
        z, y, x = shape
        if z > 10 and y > 20 and x > 20:
            arr[z // 4 : z // 3, y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * z // 5 : 4 * z // 5, 3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    return arr


def bench_pair(arr: np.ndarray, parallel: int, reps: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # warmup
    edt.edtsq(arr, parallel=parallel)
    edt.edtsq_nd(arr, parallel=parallel)

    spec_best = float('inf')
    nd_best = float('inf')
    spec_out: np.ndarray | None = None
    nd_out: np.ndarray | None = None
    for _ in range(reps):
        t0 = time.perf_counter(); spec_tmp = edt.edtsq(arr, parallel=parallel); t1 = time.perf_counter()
        t2 = time.perf_counter(); nd_tmp = edt.edtsq_nd(arr, parallel=parallel); t3 = time.perf_counter()
        spec_best = min(spec_best, t1 - t0)
        nd_best = min(nd_best, t3 - t2)
        spec_out = spec_tmp
        nd_out = nd_tmp
    assert spec_out is not None and nd_out is not None
    return spec_best, nd_best, spec_out, nd_out


def adaptive_repeat(time_s: float, min_samples: int, min_time: float, max_time: float) -> int:
    if time_s <= 0:
        return min_samples
    reps = max(min_samples, int(min_time / time_s))
    reps = min(reps, int(max_time / max(time_s, 1e-9)))
    if reps < 1:
        reps = 1
    return reps


def extract_axes(profile: dict) -> str:
    axes = profile.get('axes', [])
    parts = []
    for entry in axes:
        parts.append(f"{entry.get('kind')}@{entry.get('axis')}:{float(entry.get('time', 0.0)):.6f}")
    return ';'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ND profiling scenarios")
    parser.add_argument('--parallels', default='4,8,16', help='Comma separated list of parallel values')
    parser.add_argument('--dims', default='2,3', help='Comma separated dimensions to test (e.g. "2,3")')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions per timing')
    parser.add_argument('--dtype', default='uint8', help='NumPy dtype for test arrays')
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'nd_profile_runs.csv'), help='Output CSV path')
    parser.add_argument('--no-header', action='store_true', help='Skip CSV header when appending to existing file')
    args = parser.parse_args()

    parallels = parse_int_list(args.parallels)
    dims_requested = parse_int_list(args.dims)
    dtype = np.dtype(args.dtype)
    shapes = default_shapes(dims_requested)
    min_time = float(os.environ.get('EDT_BENCH_MIN_TIME', '0.5'))
    max_time = float(os.environ.get('EDT_BENCH_MAX_TIME', '10.0'))
    min_samples = int(os.environ.get('EDT_BENCH_MIN_REPEAT', '1'))
    rng = np.random.default_rng(0)

    os.environ.pop('EDT_ND_AUTOTUNE', None)
    os.environ.pop('EDT_ND_THREAD_CAP', None)
    os.environ['EDT_ND_PROFILE'] = '1'

    rows = []
    for parallel in parallels:
        for shape in shapes:
            arr = make_array(rng, shape, dtype)
            spec_time, nd_time, spec_out, nd_out = bench_pair(arr, parallel, args.reps)
            diff = float(np.max(np.abs(spec_out - nd_out)))
            del spec_out, nd_out

            repeat_count = adaptive_repeat(nd_time, min_samples=min_samples, min_time=min_time, max_time=max_time)
            if repeat_count > 1:
                spec_times = [spec_time]
                nd_times = [nd_time]
                max_diff = diff
                for _ in range(repeat_count - 1):
                    spec_time_r, nd_time_r, spec_out, nd_out = bench_pair(arr, parallel, args.reps)
                    spec_times.append(spec_time_r)
                    nd_times.append(nd_time_r)
                    max_diff = max(max_diff, float(np.max(np.abs(spec_out - nd_out))))
                    del spec_out, nd_out
                spec_time = float(np.mean(spec_times))
                nd_time = float(np.mean(nd_times))
                diff = max_diff
            profile = edt.edtsq_nd_last_profile() or {}
            sections = profile.get('sections', {})
            row = {
                'shape': 'x'.join(str(s) for s in shape),
                'dims': len(shape),
                'parallel_request': parallel,
                'spec_ms': spec_time * 1000.0,
                'nd_ms': nd_time * 1000.0,
                'ratio': nd_time / spec_time if spec_time else float('inf'),
                'parallel_used': profile.get('parallel_used'),
                'max_abs_diff': diff,
                'total_ms': float(sections.get('total', 0.0)) * 1000.0,
                'prep_ms': float(sections.get('prep', 0.0)) * 1000.0,
                'multi_pass_ms': float(sections.get('multi_pass', 0.0)) * 1000.0,
                'parabolic_pass_ms': float(sections.get('parabolic_pass', 0.0)) * 1000.0,
                'multi_fix_ms': float(sections.get('multi_fix', 0.0)) * 1000.0,
                'post_fix_ms': float(sections.get('post_fix', 0.0)) * 1000.0,
                'axes_detail': extract_axes(profile),
            }
            rows.append(row)
            print(
                f"shape={row['shape']:<12} p={parallel:<3d} spec={row['spec_ms']:.3f}ms "
                f"nd={row['nd_ms']:.3f}ms ratio={row['ratio']:.3f} diff={row['max_abs_diff']:.3e} "
                f"multi_ms={row['multi_pass_ms']:.3f} parab_ms={row['parabolic_pass_ms']:.3f}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not args.no_header and (not output_path.exists() or output_path.stat().st_size == 0)
    with output_path.open('a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == '__main__':
    main()
