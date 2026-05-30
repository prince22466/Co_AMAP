# Train CSV Compression Results

Source: `E:/Projects/Co_AMAP/Co_Kaggle/g1/competition_material/train.csv`

The source has 9,500 rows and no duplicate prompts or duplicate prompt-answer pairs, so the useful compression path is representative subset selection rather than exact deduplication.

Detected task families:

| family | rows |
| --- | ---: |
| bit_manipulation | 1,602 |
| gravity_formula | 1,597 |
| unit_conversion | 1,594 |
| text_decryption | 1,576 |
| numeral_system | 1,576 |
| equation_transform | 1,555 |

## Recommended 500-row subset

Use `keep_0500_stratified_feature_coverage.json` for the row indices, or `keep_0500_stratified_feature_coverage_train.csv` as the ready-to-train compressed CSV.

Rationale: this keeps the six task families balanced and samples across family-specific features:

- bit manipulation: number of examples, input/output bit density, presence of all-zero/all-one edge cases
- text decryption: example count, target/answer word counts, lexical starts
- gravity and unit conversion: numeric target and answer ranges
- numeral system: numeric target ranges and answer length
- equation transforms: target/answer length and symbol density

Rows kept: 500 of 9,500. Approximate prompt+answer character volume: 5.25% of the original.

Family distribution:

| family | rows |
| --- | ---: |
| bit_manipulation | 84 |
| gravity_formula | 84 |
| unit_conversion | 84 |
| text_decryption | 83 |
| numeral_system | 83 |
| equation_transform | 82 |

## Alternative subsets

| subset | rows | approx char volume | use case |
| --- | ---: | ---: | --- |
| `keep_0500_stratified_feature_coverage` | 500 | 5.25% | Recommended default; balanced task and feature coverage |
| `keep_0500_tfidf_centroid_coverage` | 500 | 5.67% | Text-space prototypes from TF-IDF/SVD clusters; useful as a diversity check |
| `keep_0500_length_coverage` | 500 | 4.95% | Preserves prompt/answer length spread; less balanced across task families |
| `keep_0500_random_seed_20260529` | 500 | 5.21% | Deterministic random baseline |
| `keep_0250_stratified_feature_coverage` | 250 | 2.62% | Faster smoke-test subset before training on 500 rows |

## Files

Each subset has three files:

- `*.json`: full zero-based row-index list to keep from the original CSV
- `*.csv`: row index, id, family, and length metadata
- `*_train.csv`: compressed training file with `id,prompt,answer`

The analysis script is `../analyze_train_compression.py`; rerun it to regenerate these files.
