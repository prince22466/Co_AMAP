#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

SRC=../../tests_BI-V150/Optimization/v8
[[ -d "$SRC/FlagGems" && -d "$SRC/vllm-plugin-FL" && -s readme.md && -s report.pdf ]]

# Refresh the generated source copies and archive.
rm -rf -- FlagGems vllm-plugin-FL
cp -a -- "$SRC/FlagGems" "$SRC/vllm-plugin-FL" .
rm -f -- prince22466.zip
zip -qr prince22466.zip FlagGems vllm-plugin-FL readme.md report.pdf \
  -x '*/.git' '*/.git/*' '*/__pycache__/*' '*/.pytest_cache/*' '*/.mypy_cache/*' '*/.ruff_cache/*' '*.pyc'
echo "Created: $(pwd)/prince22466.zip"
