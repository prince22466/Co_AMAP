#!/usr/bin/env python3
"""Sequential v17 diagnostic orchestrator; requires exclusive GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from analyze import summarize
from probe import CASES, ATTN

HERE = Path(__file__).resolve().parent
V17 = HERE.parent


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_for(plugin=V17/"vllm-plugin-FL"):
    env = os.environ.copy()
    for key in ("VLLM_FL_USE_FLAGGEMS_ATTN","VLLM_ATTENTION_BACKEND","VLLM_BATCH_INVARIANT",
                "V17_METADATA","V17_METADATA_EVERY"):
        env.pop(key,None)
    env["PYTHONPATH"] = os.pathsep.join([str(V17/"FlagGems/src"),str(plugin),
                                        str(HERE),env.get("PYTHONPATH","")])
    env["VLLM_PLUGINS"] = "fl"
    return env


def logged(cmd, path, env, timeout):
    with path.open("w") as f:
        f.write("COMMAND "+json.dumps(cmd)+"\n")
        f.flush()
        try:
            p = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,
                                 start_new_session=True)
        except OSError as e:
            f.write(str(e)+"\n")
            return 127
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write("\nTIMEOUT\n")
            return 124
        finally:
            stop(p)


def stop(p):
    if p.poll() is None:
        try:
            os.killpg(p.pid,signal.SIGTERM)
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()
        except ProcessLookupError:
            pass


def assert_port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1",port))
        except OSError as e:
            raise RuntimeError("Port already in use: stop your own server first") from e


def native(a, out):
    d=out/"native"
    d.mkdir()
    env=env_for()
    cli=shutil.which("ixkn-cli")
    if cli:
        for flag in ("--help","--version","--list-sections","--query-metrics","--list-metrics"):
            logged([cli,flag],d/(flag[2:]+".txt"),env,60)
    else:
        append(out/"status.jsonl",dict(stage="native_counters",status="unavailable",reason="ixkn-cli not on PATH"))
    for case in a.cases:
        layouts=["shuffled","contiguous"] if case in ATTN and a.compare_layouts else ["shuffled"]
        for layout in layouts:
            stem=case+"_"+layout if case in ATTN else case
            cmd=[sys.executable,str(HERE/"probe.py"),case,"--layout",layout]
            rc=logged(cmd,d/(stem+".timing.log"),env,a.probe_timeout)
            append(out/"status.jsonl",dict(stage="probe_timing",case=stem,returncode=rc))
            if rc or not cli:
                continue
            # Keep raw vendor values/units. Never substitute NVIDIA metrics.
            section=(["--metrics",a.native_metrics] if a.native_metrics else
                     ["--section",a.native_sections])
            kernel=("kernel_unified_attention,reduce_segments" if case in ATTN else
                    ("linear_kernel" if case.startswith("linear") else "mm_kernel"))
            args=[cli,"--profile-from-start","off","--csv","--page","raw",
                  "--kernel-name",kernel,*section,*cmd,"--native"]
            rc=logged(args,d/(stem+".counters.csv"),env,a.probe_timeout)
            append(out/"status.jsonl",dict(stage="native_counters",case=stem,returncode=rc,
                   note="Inspect actual metric rows and kernel names; exit 0 alone does not prove collection."))


def api(port,path,post=False,timeout=10):
    req=urllib.request.Request("http://127.0.0.1:"+str(port)+path,
                               data=b"" if post else None)
    with urllib.request.urlopen(req,timeout=timeout) as f:
        return f.read().decode()


def instrument(out):
    # Copy plugin only. The production tree and benchmark remain untouched.
    plugin=out/"instrumented_plugin"
    shutil.copytree(V17/"vllm-plugin-FL",plugin,ignore=shutil.ignore_patterns(".git","__pycache__"))
    path=plugin/"vllm_fl/worker/model_runner.py"
    source=path.read_text()
    anchor="        self.input_batch.block_table.commit_block_table(num_reqs)\n"
    if source.count(anchor)!=1:
        raise RuntimeError("v17 runner anchor changed; refuse ambiguous instrumentation")
    insert=("        from metadata_capture import capture as _v17_capture\n"
            "        _v17_capture(self, scheduler_output, num_scheduled_tokens)\n")
    path.write_text(source.replace(anchor,anchor+insert,1))
    (out/"instrumentation.json").write_text(json.dumps(dict(
        original_sha256=sha(V17/"vllm-plugin-FL/vllm_fl/worker/model_runner.py"),
        diagnostic_sha256=sha(path), anchor=anchor, insertion=insert),indent=2))
    return plugin


def serve(a,out):
    assert_port_free(a.port)
    benchmark=a.benchmark
    if benchmark is None:
        candidates=[V17.parent.parent/"benchmark_throughput_serve_half.py",
                    Path("/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py")]
        benchmark=next((p for p in candidates if p.is_file()),None)
    if benchmark is None or not benchmark.is_file():
        raise RuntimeError("Supply --benchmark pointing to the existing unmodified diagnostic half-run script")
    benchmark=benchmark.resolve()
    if not Path(a.model).is_dir():
        raise RuntimeError("Model directory not found: "+a.model)
    plugin=instrument(out) if not a.no_metadata else V17/"vllm-plugin-FL"
    env=env_for(plugin)
    if not a.no_metadata:
        env.update(V17_METADATA=str(out/"batch_metadata.jsonl"),V17_METADATA_EVERY=str(a.metadata_every))
    traces=out/"traces"
    traces.mkdir()
    config=dict(profiler="torch",torch_profiler_dir=str(traces),
                torch_profiler_use_gzip=True,torch_profiler_record_shapes=True,
                torch_profiler_with_memory=True,torch_profiler_with_stack=False)
    cmd=["vllm","serve",a.model,"--port",str(a.port),
         "--compilation-config",'{"cudagraph_mode":"FULL_DECODE_ONLY"}',
         "--served-model-name","minicpm","--gpu-memory-utilization","0.85",
         "--max-model-len","131072","--profiler-config",json.dumps(config)]
    (out/"server_config.json").write_text(json.dumps(dict(
        command=cmd,benchmark=str(benchmark),benchmark_sha256=sha(benchmark),
        plugin=str(plugin),diagnostic_only=True),indent=2))
    with (out/"server.log").open("w") as f:
        server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,
                                start_new_session=True)
        try:
            until=time.monotonic()+a.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server exited; inspect server.log")
                try:
                    api(a.port,"/health")
                    break
                except OSError:
                    if time.monotonic()>until:
                        raise RuntimeError("Server readiness timeout")
                    time.sleep(2)
            for label,n,prompts in (("4k",4096,256),("16k",16384,128)):
                if label not in a.workloads:
                    continue
                start=time.time()
                stop_event=threading.Event()
                active=threading.Event()

                def telemetry():
                    while not stop_event.is_set():
                        row=dict(workload=label,timestamp=time.time())
                        try:
                            row["metrics"]=api(a.port,"/metrics")
                            for line in row["metrics"].splitlines():
                                if line.startswith(("vllm:num_requests_running{","vllm:num_requests_running ")):
                                    if float(line.rsplit(None,1)[-1])>0:
                                        active.set()
                        except Exception as e:
                            row["metrics_error"]=str(e)
                        try:
                            r=subprocess.run(["ixsmi"],capture_output=True,text=True,timeout=5)
                            row.update(ixsmi=r.stdout,ixsmi_returncode=r.returncode)
                        except Exception as e:
                            row["ixsmi_error"]=str(e)
                        append(out/"metrics.jsonl",row)
                        stop_event.wait(a.sample_seconds)
                thread=threading.Thread(target=telemetry,daemon=True)
                thread.start()
                client=None
                try:
                    bcmd=[sys.executable,str(benchmark),"--model",a.model,
                          "--served-model-name","minicpm","--port",str(a.port),
                          "--test-cases",json.dumps([[n,1024,64,prompts]])]
                    with (out/("bench_"+label+".log")).open("w") as bf:
                        client=subprocess.Popen(bcmd,stdout=bf,stderr=subprocess.STDOUT,
                                                cwd=out,env=env,start_new_session=True)
                        deadline=time.monotonic()+a.workload_timeout
                        next_window=0
                        captured=0
                        while client.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError("Server exited during workload")
                            if time.monotonic()>deadline:
                                raise TimeoutError("Diagnostic workload timed out")
                            if active.is_set() and captured<a.windows and time.monotonic()>=next_window:
                                rec=dict(workload=label,index=captured,start=time.time())
                                before=set(traces.rglob("*.json*"))
                                try:
                                    api(a.port,"/start_profile",True,60)
                                    rec["start_ok"]=True
                                    until=min(deadline,time.monotonic()+a.window_seconds)
                                    while client.poll() is None and time.monotonic()<until:
                                        time.sleep(.5)
                                except Exception as e:
                                    rec["start_error"]=str(e)
                                finally:
                                    try:
                                        api(a.port,"/stop_profile",True,120)
                                        rec["stop_ok"]=True
                                    except Exception as e:
                                        rec["stop_error"]=str(e)
                                    rec["end"]=time.time()
                                    rec["files"]=[str(p.relative_to(out)) for p in sorted(set(traces.rglob("*.json*"))-before)]
                                    append(out/"windows.jsonl",rec)
                                captured+=1
                                next_window=time.monotonic()+a.window_gap
                            time.sleep(1)
                        append(out/"status.jsonl",dict(stage="serve",workload=label,
                               returncode=client.returncode,profile_windows_attempted=captured))
                        if client.returncode:
                            raise RuntimeError("Benchmark failed: "+label)
                finally:
                    if client is not None:
                        stop(client)
                    stop_event.set()
                    thread.join(timeout=20)
                    append(out/"workloads.jsonl",dict(workload=label,start=start,end=time.time()))
        finally:
            stop(server)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["all","native","serve","analyze"],default="all")
    p.add_argument("--out",type=Path)
    p.add_argument("--model",default="/workspace/MiniCPM5-2B")
    p.add_argument("--benchmark",type=Path)
    p.add_argument("--port",type=int,default=9031)
    p.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    p.add_argument("--compare-layouts",action="store_true")
    p.add_argument("--native-sections",default="all")
    p.add_argument("--native-metrics",default="")
    p.add_argument("--probe-timeout",type=int,default=600)
    p.add_argument("--startup-timeout",type=int,default=900)
    p.add_argument("--workload-timeout",type=int,default=7200)
    p.add_argument("--workloads",nargs="+",choices=["4k","16k"],default=["4k","16k"])
    p.add_argument("--windows",type=int,default=3)
    p.add_argument("--window-seconds",type=int,default=8)
    p.add_argument("--window-gap",type=int,default=90)
    p.add_argument("--sample-seconds",type=int,default=2)
    p.add_argument("--metadata-every",type=int,default=10)
    p.add_argument("--no-metadata",action="store_true")
    a=p.parse_args()
    if a.stage=="analyze":
        if not a.out or not a.out.is_dir():
            p.error("--stage analyze requires an existing --out directory")
        summarize(a.out.resolve())
        return
    if any(getattr(a,k)<1 for k in ("probe_timeout","startup_timeout","workload_timeout",
                                    "windows","window_seconds","window_gap","sample_seconds",
                                    "metadata_every")):
        p.error("timeouts, windows and sampling intervals must be positive")
    assert_port_free(a.port)
    out=(a.out or (V17.parent/("v17_memory_"+time.strftime("%Y%m%d_%H%M%S")+"_"+str(os.getpid())))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    print("Diagnostic output:",out,flush=True)
    (out/"manifest.json").write_text(json.dumps(dict(
        args=vars(a),source=str(V17),diagnostic_only=True,
        source_hashes={str(x.relative_to(V17)):sha(x) for x in [
            V17/"vllm-plugin-FL/vllm_fl/platform.py",
            V17/"vllm-plugin-FL/vllm_fl/dispatch/backends/triton_unified_attention.py",
            V17/"FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py"]}),default=str,indent=2))
    try:
        if a.stage in ("all","native"):
            native(a,out)
        if a.stage in ("all","serve"):
            serve(a,out)
    except BaseException as e:
        append(out/"status.jsonl",dict(stage=a.stage,error=repr(e)))
        raise
    finally:
        summarize(out)


if __name__=="__main__":
    main()
