#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

SRC=../../tests_BI-V150/Optimization/v8
[[ -d "$SRC/FlagGems" && -d "$SRC/vllm-plugin-FL" && -s readme.md && -s report.pdf ]]

# Stage the files under the same top-level directory as the existing archive.
output="$(pwd)/prince22466.zip"
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
mkdir -- "$stage/prince22466"
cp -a -- "$SRC/FlagGems" "$SRC/vllm-plugin-FL" readme.md report.pdf "$stage/prince22466/"
rm -f -- "$output"
(
  cd -- "$stage"
  zip -qr "$output" prince22466 \
    -x '*/.git' '*/.git/*' '*/__pycache__/*' '*/.pytest_cache/*' '*/.mypy_cache/*' '*/.ruff_cache/*' '*.pyc'
)
echo "Created: $output"
