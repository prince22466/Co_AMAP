# v11 latest research handoff

Updated UTC: 2026-11-10T05:32:08+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: e1a7b59a3e43ff90eb227f29fa8155960d14e2e9

Remaining GPU time: unknown

Finding: Short prefill profiles: 4K sampled GPU-sum mm 46.43 percent, attention 45.33 percent; 16K attention 79.25 percent, mm 17.81 percent

Evidence/provenance: Off-instance .autoresearch-work/remote-handoff/profiles holds both compressed torch traces and summaries; 2048x2048 BF16 projection shapes and attention Q shape 2048x16x128 observed; 16K diagnostic cache hit rate rose to 16.7 percent; profiles include initial warmup and graph replay limits, not wall-time shares

Small CSV evidence committed when --push works: none (external backup required)

Next action: Capture mixed/decode phases, assess Amdahl bound and select one real operator change

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
