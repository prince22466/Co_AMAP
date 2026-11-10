# v11 latest research handoff

Updated UTC: 2026-11-10T09:54:01+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 55675bc55c1a67fa7dfc9e50de134f3c871b7dcb

Remaining GPU time: unknown

Finding: User ended session: provisional shape-specific grouped MM candidate is saved but not applied; original v11 inference code remains restored

Evidence/provenance: Draft PR #236; fresh control 4K 2211.04/2526.52 and 16K 1824.16/1894.52 total tok/s with all four rows successful. Independent kernel screens 1.40223 to 1.16975 ms and 1.41153 to 1.17371 ms on BF16 2048x2048 by 2048x12288. Saved .autoresearch-work/mm-candidate.patch and mm-candidate.py; logs .autoresearch-work/remote-handoff/mm-screen.log and mm-screen-2.log. No production candidate proof or serving A/B.

Small CSV evidence committed when --push works: none (external backup required)

Next action: In next confirmed GPU allocation restore candidate from saved patch into remote v11, verify hashes, add real-shape numerical and fallback tests, prove production invocation, run matched 4K/16K short diagnostics; only if justified and admitted run guarded unchanged half-runs, preserve and independently replicate. Human runs unchanged MATH-500 for chosen candidate.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
