# v11 latest research handoff

Updated UTC: 2026-11-10T09:15:00+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 206bfb25f117da8519b2953f3fda9650dc3f8015

Remaining GPU time: unknown

Finding: Baseline server-log sampling shows sustained 16K KV pressure and queues; this does not identify a permissible or causal optimization by itself

Evidence/provenance: Approximate 16K time window: 238 active 10-second server samples, 201 with KV usage >=90 percent and 225 with requests waiting; 4K window: 110 active samples, zero with KV usage >=90 percent and 14 with requests waiting. Local .autoresearch-work/analyze-server-log.py and remote-handoff/v11-control.log. Windows are approximate and neither GPU kernel shares nor attribution of wall delay

Small CSV evidence committed when --push works: none (external backup required)

Next action: Next session measure phase-resolved critical path and preemption cost; investigate only allowed equivalent-semantics operator/data-path changes, without altering cache capacity or scheduler settings

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
