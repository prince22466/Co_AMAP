"""Run the frozen v12_wsp notebook evaluation cells on Colab L4 (supports custom disjoint discovery cohorts)."""
import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"
CHAMPION = "8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815"
DISCOVERY02 = ["rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130"]
DISCOVERY03 = [
    "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
    "fastapi_14786", "fastapi_14479", "requests_7328",
]
HOLDOUT = ["rich_3043", "rich_3064", "rich_2725"]


def capture_requests(path):
    """Observe local inference inputs on development runs. Persist only model-input fields."""
    import httpx
    original_send = httpx.AsyncClient.send
    original_sync = httpx.Client.send

    def record(request):
        if request.url.host not in {"127.0.0.1", "localhost"} or not request.url.path.endswith("/chat/completions"):
            return
        data = json.loads(request.content)
        fields = ("model", "messages", "tools", "chat_template_kwargs", "tool_choice", "max_tokens", "temperature")
        with path.open("a", encoding="utf8") as f:
            print(json.dumps({k: data[k] for k in fields if k in data}), file=f)

    async def async_send(self, request, *args, **kwargs):
        record(request)
        return await original_send(self, request, *args, **kwargs)

    def sync_send(self, request, *args, **kwargs):
        record(request)
        return original_sync(self, request, *args, **kwargs)

    httpx.AsyncClient.send = async_send
    httpx.Client.send = sync_send


def save_replay_tokens(run, base_url):
    """Ask the serving tokenizer to render captured requests after the task timers stop."""
    import requests
    path = run / "requests.jsonl"
    url = base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
    with (run / "replay_tokens.jsonl").open("w") as output:
        for row in map(json.loads, path.read_text().splitlines()):
            payload = {k: row[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in row}
            payload["add_generation_prompt"] = True
            headers = {"Authorization": "Bearer " + os.environ.get("LOCAL_API_KEY", "EMPTY")}
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            tokens = response.json()["tokens"]
            digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            print(json.dumps(dict(request_sha=digest, tokens=tokens)), file=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("discovery", "discovery03", "holdout", "custom_dev", "15", "30"), required=True)
    parser.add_argument("--tasks", nargs="+", help="Explicit disjoint dev task IDs when --split=custom_dev")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--sha", default=CHAMPION)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    run = ROOT / "experiments" / args.name
    run.mkdir(parents=True, exist_ok=False)
    assets = json.loads((ROOT / "assets.json").read_text())
    model = ROOT / "input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.exists():
        model.symlink_to(assets["model"], target_is_directory=True)

    bundle = run / "input"
    with zipfile.ZipFile(ROOT / "champion.zip") as z:
        z.extractall(bundle)
    if args.adapter:
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(args.adapter / filename, bundle / "adapters/wsp_r1" / filename)
    actual = hashlib.sha256((bundle / "adapters/wsp_r1/adapter_model.safetensors").read_bytes()).hexdigest()
    if actual != args.sha:
        raise ValueError(f"Requested adapter digest {args.sha} differs from staged adapter {actual}")

    split_num = 30 if args.split == "30" else 15
    (bundle / "autoresearch_run.json").write_text(
        json.dumps(dict(iter_id=args.name, split=split_num, expected_adapter_sha256=actual))
    )
    rolling = ROOT / "input/datasets/prince22466/g6-autoresearch-sub-v2"
    rolling.parent.mkdir(parents=True, exist_ok=True)
    if rolling.is_symlink():
        rolling.unlink()
    rolling.symlink_to(bundle, target_is_directory=True)

    cells = json.loads((ROOT / "frozen.ipynb").read_text())["cells"]
    from IPython.display import display
    scope = dict(__name__="__main__", display=display)

    def cell(index, patch_gpu_mem=False):
        source = "".join(cells[index]["source"]).replace("/kaggle/input/", str(ROOT / "input") + "/")
        source = source.replace("/kaggle/working", str(ROOT / "working"))
        if patch_gpu_mem and args.gpu_mem != 0.90:
            # Avoid single-L4 22 GiB OOM during _dummy_sampler_run when max_lora_rank >= 8
            source = source.replace("gpu_memory_utilization=0.90,", f"gpu_memory_utilization={args.gpu_mem},")
        exec(compile(source, f"<frozen-cell-{index}>", "exec"), scope)

    setup = "".join(cells[2]["source"]).split("WHEELHOUSE_DIR =")[0]
    exec(compile(setup, "<frozen-environment>", "exec"), scope)
    for index in (4, 5, 6):
        cell(index)
    scope.update(GRAPH_DIR=str(DATA / "graphs"), EMBEDDINGS_DIR=str(DATA / "embeddings"))
    try:
        cell(10, patch_gpu_mem=True)
        cell(13)
        cell(14)
        if args.split in ("discovery", "discovery03", "holdout", "custom_dev"):
            if args.split == "discovery":
                chosen = DISCOVERY02
            elif args.split == "discovery03":
                chosen = DISCOVERY03
            elif args.split == "holdout":
                chosen = HOLDOUT
            else:
                if not args.tasks:
                    raise ValueError("--tasks required for --split custom_dev")
                chosen = args.tasks
            tasks = scope["tasks"]
            reserved = {(tasks[i].repo, tasks[i].base_commit) for i in scope["QUICK_30"]}
            if args.split != "holdout":
                reserved |= {(t.repo, t.base_commit) for t in tasks if t.instance_id in HOLDOUT}
            cohort = [t for t in tasks if t.instance_id in chosen]
            assert len(cohort) == len(chosen), f"Unknown task IDs: {set(chosen) - {t.instance_id for t in cohort}}"
            snapshots = {(t.repo, t.base_commit) for t in cohort}
            assert len(snapshots) == len(chosen) and not (snapshots & reserved), "Snapshot overlap with QUICK_30/HOLDOUT!"
            scope["TASK_IDS"] = chosen
            capture_requests(run / "requests.jsonl")
        cell(15)
        cell(19)
        for name in ("neon_metrics.json", "neon_failures.jsonl"):
            shutil.copy2(scope["WORKING_DIR"] / name, run / name)
        shutil.copy2(scope["DIAGNOSTICS_ZIP"], run / "diagnostics.zip")
        if (run / "requests.jsonl").is_file():
            save_replay_tokens(run, scope["server_instance"].base_url)
    finally:
        if scope.get("server_instance") is not None:
            scope["server_instance"].stop()


if __name__ == "__main__":
    main()
