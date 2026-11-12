#!/usr/bin/env bash
# Build a FlagOS S2 / Track 2 draft submission from the checked-in v8 code.
# This script does NOT run benchmarks, change kernels, or edit competition code.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$HERE/../../tests_BI-V150" && pwd)"
SRC="$PROJECT/Optimization/v8"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/dist}"
REPORT_PDF="${REPORT_PDF:-$HERE/report.pdf}"

: "${TEAM_NAME:?Set TEAM_NAME to the exact registered team name}"

if [[ "$TEAM_NAME" == */* || "$TEAM_NAME" == *\\* || "$TEAM_NAME" == "." || "$TEAM_NAME" == ".." ]]; then
  echo "Invalid TEAM_NAME: cannot contain slash, backslash or relative path components" >&2
  exit 2
fi

for tool in mktemp tar zip unzip sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Required packaging utility missing: $tool" >&2
    exit 2
  }
done

for src_dir in "$SRC/vllm-plugin-FL" "$SRC/FlagGems"; do
  [[ -d "$src_dir" ]] || {
    echo "Missing v8 source tree: $src_dir" >&2
    exit 2
  }
done

[[ -s "$HERE/readme.md" ]] || {
  echo "Missing competition readme.md" >&2
  exit 2
}
[[ -s "$REPORT_PDF" ]] || {
  echo "Missing reviewed report PDF: $REPORT_PDF" >&2
  echo "Render report.md to report.pdf first; see README_v8.md" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
root="$tmp/$TEAM_NAME"
mkdir -p "$root/vllm-plugin-FL" "$root/FlagGems"

# Copy the actual v8 sources; never copy only a benchmark script or a different version.
# Only VCS metadata and transient local caches are excluded.
for dir in vllm-plugin-FL FlagGems; do
  tar -C "$SRC/$dir" \
    --exclude='./.git' \
    --exclude='*/.git' \
    --exclude='*/__pycache__' \
    --exclude='*/.pytest_cache' \
    --exclude='*/.mypy_cache' \
    --exclude='*/.ruff_cache' \
    --exclude='*.pyc' \
    -cf - . | tar -C "$root/$dir" -xf -
done

cp -- "$HERE/readme.md" "$root/readme.md"
cp -- "$REPORT_PDF" "$root/report.pdf"

archive="$OUTPUT_DIR/$TEAM_NAME.zip"
rm -f -- "$archive"
(
  cd "$tmp"
  zip -q -r "$archive" "$TEAM_NAME"
)
unzip -tq "$archive" >/dev/null

echo "Created: $archive"
echo "Expected archive root: $TEAM_NAME/"
echo "SHA-256:"
sha256sum "$archive" "$REPORT_PDF"

# Validate the four mandatory ZIP entries without a pipe-to-head SIGPIPE.
unzip -Z -1 "$archive" > "$tmp/contents.txt"
for entry in \
  "$TEAM_NAME/FlagGems/" \
  "$TEAM_NAME/vllm-plugin-FL/" \
  "$TEAM_NAME/readme.md" \
  "$TEAM_NAME/report.pdf"; do
  if ! grep -Fxq -- "$entry" "$tmp/contents.txt"; then
    echo "Missing required ZIP entry: $entry" >&2
    exit 1
  fi
  printf '%s\n' "$entry"
done

echo "Packaging completed. Reproduce deployment, accuracy and both benchmark cases"
echo "from this ZIP before uploading it to the competition platform."
