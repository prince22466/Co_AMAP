#!/usr/bin/env bash
# Group11 standalone GPU A/B tests. Does not change MiniCPM, vLLM or official benchmarks.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${MODE:-full}
PROFILE=${PROFILE:-0}
RUN_GLUON=${RUN_GLUON:-0}
case "$MODE" in full|smoke) ;; *) echo "MODE must be smoke or full" >&2; exit 2 ;; esac
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
OUT=${OUT:-results/run_${RUN_ID}}
mkdir -p "$OUT"
python3 test_compiler_math.py 2>&1 | tee "$OUT/cpu_tests.log"
for lab in K01 K02 K03 K04; do
    echo "== Group11 $lab ($MODE) =="
    opts=(--lab "$lab" --strict --out "$OUT/${lab}.csv" --dump-asm "$OUT/asm/$lab")
    if [[ "$MODE" == smoke ]]; then
        opts+=(--single-run --n 4096 --rows 129 --cols 193 --iters 16 --live-iters 16 --licm-iters 16)
        case "$lab" in
            K01) opts+=(--unrolls 1,4) ;;
            K02) opts+=(--tiles 16x64,32x32) ;;
        esac
    fi
    python3 triton_compiler_lab.py "${opts[@]}" 2>&1 | tee "$OUT/${lab}.log"
done
if [[ "$RUN_GLUON" == 1 ]]; then
    echo '== Optional explicit Gluon layouts (K02G) =='
    opts=(--lab K02G --strict --out "$OUT/K02G.csv" --dump-asm "$OUT/asm/K02G")
    if [[ "$MODE" == smoke ]]; then
        opts+=(--single-run --n 4096 --vector-widths 1,2)
    fi
    # Unsupported Gluon will be reported and can fail; never silently substitute standard Triton.
    python3 triton_compiler_lab.py "${opts[@]}" 2>&1 | tee "$OUT/K02G.log"
fi
if [[ "$PROFILE" == 1 ]]; then
    command -v ixkn-cli >/dev/null || { echo 'PROFILE=1 requires ixkn-cli' >&2; exit 2; }
    ixkn-cli --version > "$OUT/profiler_version.txt" 2>&1
    ixkn-cli --query-metrics > "$OUT/available_metrics.txt" 2>&1
    # Only profile named, single-run variants. All capture data is separate from ordinary timing.
    metrics=${METRICS:-time_duration}
    for target in 'K01 unroll1 unroll_kernel' 'K01 unroll4 unroll_kernel' \
                  'K04 licm_enabled licm_kernel' 'K04 licm_disabled licm_kernel'; do
        read -r lab variant kernel <<< "$target"
        echo "== Profiler capture $lab/$variant =="
        ixkn-cli --profile-from-start off --page details --csv --cache-control none \
            --kernel-name "$kernel" --metrics "$metrics" \
            python3 triton_compiler_lab.py --lab "$lab" --variant "$variant" --single-run \
                --n 4096 --iters 16 --licm-iters 16 --dump-asm "$OUT/profile_asm" \
                --out "$OUT/prof_${lab}_${variant}.csv" \
                > "$OUT/prof_${lab}_${variant}.log" 2>&1 || {
                    cat "$OUT/prof_${lab}_${variant}.log"; exit 1;
                }
    done
    echo 'Profiler raw records require manual numeric-counter verification.'
fi
echo "Group11 results: $OUT"
