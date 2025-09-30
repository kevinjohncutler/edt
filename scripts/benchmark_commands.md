# Benchmark Commands

Use these commands to rebuild the extensions and capture ND vs. legacy timings on each platform.

## macOS (local machine)

```bash
# rebuild in-place (optional, keeps .so in src/)
python setup.py build_ext --inplace

# run benchmark sweep and write CSV in milliseconds
EDT_BENCH_MIN_TIME=0.05 EDT_BENCH_MAX_TIME=1.0 \
python scripts/bench_nd_profile.py \
  --parallels 1,2,4,8,16 --dims 2,3 --reps 5 \
  --output benchmarks/nd_profile_mac_20250929.csv
```

To inspect the ND profile for a specific shape:

```bash
python - <<'PY'
import numpy as np, edt, os
os.environ['EDT_ND_PROFILE'] = '1'
arr = np.zeros((384, 384, 384), dtype=np.uint8)
arr[192, 192, 192] = 1
edt.edtsq_nd(arr, parallel=4)
print(edt.edtsq_nd_last_profile())
PY
```

## Threadripper (remote)

```bash
ssh kcutler@threadripper.local \
  'cd DataDrive/edt && PYTHONPATH=. ~/.pyenv/versions/3.12.11/bin/python \
     scripts/bench_nd_profile.py \
     --parallels 1,2,4,8,16 --dims 2,3 --reps 5 \
     --output benchmarks/nd_profile_threadripper_20250929.csv'
```

Make sure the editable install is up to date first:

```bash
ssh kcutler@threadripper.local \
  'cd DataDrive/edt && PYTHONPATH=. ~/.pyenv/versions/3.12.11/bin/pip install -e .'
```
