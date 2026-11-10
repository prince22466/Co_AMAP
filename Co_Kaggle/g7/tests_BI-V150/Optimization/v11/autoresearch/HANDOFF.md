# v11 latest research handoff

Updated UTC: 2026-11-10T16:47:02+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: eae8b766d691667596e8c8c915d2f5735bc20cc4

Remaining GPU time: unknown

Finding: REJECTED second-projection grouped MM extension: 2048x6144 by 6144x2048 median 0.72554 to 0.76789 ms, 5.84 percent slower

Evidence/provenance: Nine BF16/FP16, stride and fallback GPU tests passed; 10 alternating same-input event samples, max output difference zero. First projection 1.40098 to 1.17001 ms. Remote .autoresearch-work/mm-dual-screen.log and mm-tests-dual.log; extension reverted to prior single-shape candidate. No serving run for extension.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up logs, inspect production-path and kernel resource evidence to select a stronger next hypothesis; single-shape candidate still has no-go serving screen

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
