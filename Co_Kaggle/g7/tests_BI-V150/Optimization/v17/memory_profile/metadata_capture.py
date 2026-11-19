"""Diagnostic-only host metadata; never reads a GPU tensor or synchronizes."""
import json
import math
import os
import time

_step = 0
_highwater = {}
_replay_until = {}


def capture(runner, scheduler_output, num_scheduled_tokens):
    global _step
    path = os.environ.get("V17_METADATA")
    if not path:
        return
    _step += 1
    n = runner.input_batch.num_reqs
    q = [int(x) for x in num_scheduled_tokens[:n]]
    old = [int(x) for x in runner.input_batch.num_computed_tokens_cpu[:n]]
    prompt = [int(x) for x in runner.input_batch.num_prompt_tokens_cpu_tensor[:n]]
    ids = runner.input_batch.req_ids[:n]
    s = [a+b for a,b in zip(q,old)]
    if not q or min(q) < 1:
        return

    # Track scheduled positions, not completed GPU work. IDs remain in memory
    # only. Resumption with an older position indicates planned replay.
    for req in scheduler_output.finished_req_ids:
        _highwater.pop(req,None)
        _replay_until.pop(req,None)
    resumed = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    replay = repeated = resumed_count = 0
    for req, lo, hi in zip(ids,old,s):
        previous = _highwater.get(req,lo)
        if req in resumed:
            resumed_count += 1
            _replay_until[req] = previous
        repeated += max(0,min(hi,previous)-lo)
        replay += max(0,min(hi,_replay_until.get(req,lo))-lo)
        if hi>=_replay_until.get(req,lo):
            _replay_until.pop(req,None)
        _highwater[req] = max(previous,hi)

    # Emit every Nth batch and every observed resumed/repeated batch.
    if (_step-1) % int(os.environ.get("V17_METADATA_EVERY","10")) and not (resumed_count or repeated):
        return
    qkind = "decode" if max(q)==1 else ("mixed" if min(q)==1 else "prefill")
    prefill_reqs = sum(a<b for a,b in zip(old,prompt))
    phase = "decode" if prefill_reqs==0 else ("prefill" if prefill_reqs==n else "mixed")
    bq = 2 if qkind=="decode" else 8
    # Proxies for the target BF16 GQA path, not actual allocation or HBM bytes.
    row = dict(timestamp=time.time(), pid=os.getpid(), step=_step,
               batch_kind_by_query_length=qkind, batch_kind_by_prompt_progress=phase,
               prefill_requests=prefill_reqs, requests=n, scheduled=sum(q),
               single_query_requests=q.count(1), query_lengths=q, seq_lengths=s,
               prior_computed_tokens=old, prompt_lengths=prompt,
               resumed_requests=resumed_count, repeated_scheduled_tokens=repeated,
               resumed_replay_scheduled_tokens=replay,
               unique_kv_payload_bytes_per_layer=1024*sum(s),
               padded_query_positions_proxy=sum(math.ceil(x/bq)*bq-x for x in q),
               active_tail_slack_tokens_proxy=sum((-x)%16 for x in s))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row)+"\n")
