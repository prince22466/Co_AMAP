# v11 latest research handoff

Updated UTC: 2026-11-10T05:28:44+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: ef21fd93d45cbe18c2dcab9b8493877693cb60fc

Remaining GPU time: unknown

Finding: Initial 4K short profile reports sampled summed GPU time mm 46.43 percent and unified attention 45.33 percent; diagnostic capture is not serving wall time

Evidence/provenance: Remote .autoresearch-work/profiles/profiler_out_0.txt and rank0.1794288498430610818.pt.trace.json.gz; first 20 profiled engine iterations with warmup and graph visibility limitations; short probe in progress

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up traces, profile 16K and mixed phases, then rank kernel hypotheses with an Amdahl bound

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
