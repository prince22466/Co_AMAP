# V6 promotion evidence

Candidate source SHA-256:
`b335347ea08533102058d8314cbb030cdd7176dd26d7ed613b46c3cdb29c6db1`

The final notebook was frozen before the proof run. Seeds 5101-5120 were not
used in the preceding mix/labor tuning.

## Fresh-seed tournament

- Engine: audited public baseline `kaggle-environments==1.32.7`
- Environment: `kaggriculture`
- Episode length: 720
- Seeds: 5101-5120
- Seats: both orientations per seed
- Completion errors: 0

Because every game completed, the post-audit rule that counts agent failures as
forfeits does not change any result below.

| Pairing | V6 record | Mean V6 reward | Mean opponent reward | Mean V6 margin | Minimum V6 game margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| v6 vs v5 | 40-0-0 | 53,669.0 | 46,384.9 | +7,284.2 | +4,491 |
| v6 vs v4 | 40-0-0 | 50,252.0 | 31,600.5 | +18,651.5 | +12,368 |

The complete immutable rows and metadata are in
[`results_v6_fresh_20/results.json`](results_v6_fresh_20/results.json), with CSV
views in the same directory. The stored raw matches were also re-scored with the
post-audit arena code: v6 remains 80-0-0 overall with zero errors.

## Loader and self-play validation

Seeds 6101-6102 ran through the file loader with `--self-play --fail-fast`.
Every v6 game finished `DONE`; self-play is excluded from ranking. Results are
in [`results_v6_validation/results.json`](results_v6_validation/results.json).
After hardening the arena, seed 6201 was rerun with isolated root-level
`main.py` paths, resolved-seed assertions, both seats, self-play, and
`--fail-fast`; all four games completed. That smoke evidence is in
[`results_arena_audit_smoke/results.json`](results_arena_audit_smoke/results.json).

The generated archive probe contained exactly one root member, `main.py`; its
uncompressed source was 21,431 bytes and the gzip archive was about 5.8 KB
(exact compressed size varies with tar metadata).

See [`AUDIT.md`](AUDIT.md) for the distinction between mechanics-faithful local
evaluation and Kaggle's non-reproducible live matchmaking/rating layer.
