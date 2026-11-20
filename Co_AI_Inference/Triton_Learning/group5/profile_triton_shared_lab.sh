#!/usr/bin/env bash
# Group5 runner; distinct from the competition benchmark.
# MODE=smoke to test correctness/compilation; RUN_GLUON=1 to include optional S04.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=${OUT:-"results/run_${STAMP}"}
mkdir -p "$OUT"
MODE=${MODE:-full}
RUN_GLUON=${RUN_GLUON:-0}

python3 test_smem_lab_math.py

if [[ "$MODE" == "smoke" ]]; then
  echo '== Group5 GPU smoke tests =='
  python3 triton_shared_memory_lab.py --lab S01 --m 65 --n 129 \
    --transpose-tile 32 --single-run --strict --out "$OUT/S01_smoke.csv"
  python3 triton_shared_memory_lab.py --lab S02 --m 64 --n 64 --k 64 \
    --gemm-tiles 32x32,64x64 --single-run --strict --out "$OUT/S02_smoke.csv"
  python3 triton_shared_memory_lab.py --lab S03 --elements 4096 --barrier-steps 4 \
    --single-run --strict --out "$OUT/S03_smoke.csv"
elif [[ "$MODE" == "full" ]]; then
  echo '== Group5 A/B performance tests =='
  for lab in S01 S02 S03; do
    python3 triton_shared_memory_lab.py --lab "$lab" \
      --dump-asm "$OUT/asm" --strict --out "$OUT/${lab}.csv" \
      2>&1 | tee "$OUT/${lab}.log"
  done
else
  echo "Unknown MODE=$MODE (use smoke or full)" >&2
  exit 2
fi

if [[ "$RUN_GLUON" == "1" ]]; then
  echo '== Optional S04: explicit Gluon shared-memory swizzle =='
  if [[ "$MODE" == "smoke" ]]; then
    python3 triton_shared_memory_lab.py --lab S04 --m 64 --n 64 \
      --single-run --strict --out "$OUT/S04_smoke.csv"
  else
    python3 triton_shared_memory_lab.py --lab S04 --dump-asm "$OUT/asm" \
      --strict --out "$OUT/S04.csv" 2>&1 | tee "$OUT/S04.log"
  fi
else
  echo 'S04 not attempted; set RUN_GLUON=1 to test explicit shared-memory layouts.'
fi

echo "Results: $OUT"
echo 'Inspect compiler shared_bytes / assembly before attributing effects to shared memory.'
