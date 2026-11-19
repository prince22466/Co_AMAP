import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import metadata_capture
from analyze import union_us, trace_summary, metric_summary, native_rows
from run import verify_benchmark
from probe import attention_payload


class TestAnalysis(unittest.TestCase):
    def test_union(self):
        self.assertEqual(union_us([(0,5),(2,7),(10,12)]),9)

    def test_payload(self):
        self.assertEqual(attention_payload([1],[4096],2),4096*1024)
        self.assertEqual(attention_payload([3],[5],2),(4+5)*1024)

    def test_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"trace.json"
            path.write_text(json.dumps({"traceEvents":[
                {"ph":"X","cat":"kernel","name":"mm_kernel","ts":1,"dur":10},
                {"ph":"X","cat":"kernel","name":"mm_kernel","ts":5,"dur":10},
                {"ph":"X","cat":"cuda_runtime","name":"cudaGraphLaunch","ts":0,"dur":1},
                {"ph":"X","cat":"gpu_memcpy","name":"Memcpy HtoD","dur":2,"args":{"bytes":256}},
                {"name":"[memory]","args":{"Bytes":4096}}]}))
            overview,groups,_=trace_summary(path)
            self.assertEqual(overview["observed_kernel_union_us"],14)
            self.assertTrue(overview["graph_visibility_unverified"])
            self.assertEqual(overview["allocator_alloc_bytes"],4096)
            self.assertEqual(next(r for r in groups if r["kind"]=="device_copy")["reported_copy_bytes"],256)

    def test_replay(self):
        metadata_capture._step = 0
        metadata_capture._highwater.clear()
        metadata_capture._replay_until.clear()
        batch=SimpleNamespace(num_reqs=1,req_ids=["x"],
                              num_computed_tokens_cpu=[0],
                              num_prompt_tokens_cpu_tensor=[8])
        runner=SimpleNamespace(input_batch=batch)
        sched=SimpleNamespace(finished_req_ids=set(),
                              scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"metadata.jsonl"
            with patch.dict("os.environ",{"V17_METADATA":str(path),"V17_METADATA_EVERY":"10"}):
                metadata_capture.capture(runner,sched,[8])
                sched.scheduled_cached_reqs.resumed_req_ids={"x"}
                batch.num_computed_tokens_cpu=[2]
                metadata_capture.capture(runner,sched,[3])
                sched.scheduled_cached_reqs.resumed_req_ids=set()
                batch.num_computed_tokens_cpu=[5]
                metadata_capture.capture(runner,sched,[4])
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(sum(r["resumed_replay_scheduled_tokens"] for r in rows),6)
            self.assertEqual(sum(r["resumed_requests"] for r in rows),1)

    def test_native_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"probe.counters.csv"
            path.write_text('COMMAND ["ixkn-cli"]\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"1","mm_kernel","dram_read_bytes","byte","1024"\n')
            row=list(native_rows(path))[0]
            self.assertEqual(row["value"],"1024")
            self.assertTrue(row["memory_related"])

    def test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"metrics.jsonl"
            rows=[dict(workload="4k",metrics='vllm:num_preemptions_total{engine="0"} '+str(i))
                  for i in (3,4)]
            path.write_text("\n".join(map(json.dumps,rows)))
            result=metric_summary(path)[0]
            self.assertIsNone(result["counter_delta"])
            self.assertEqual(result["sampled_counter_delta"],1)
            rows[0]["boundary"]="before"
            rows[1]["boundary"]="after"
            path.write_text("\n".join(map(json.dumps,rows)))
            self.assertEqual(metric_summary(path)[0]["counter_delta"],1)

    def test_wide_native_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"probe.counters.csv"
            path.write_text('ID,Kernel Name,dram__bytes_read.sum,l2_hit_rate,Grid Size\n,,byte,%,\n1,mm_kernel,1024,75,32\n')
            rows=list(native_rows(path))
            self.assertEqual([(r["metric"],r["unit"]) for r in rows],
                             [("dram__bytes_read.sum","byte"),("l2_hit_rate","%")])

    def test_benchmark_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            d=root/"benchmark_results"
            d.mkdir()
            path=d/"raw_runs_test.csv"
            header="Prefill,Decode,Conc,Num Prompts,Successful Requests,Run Status,Total tok/s\n"
            good="4096,1024,64,256,256.0,SUCCESS,2800\n"
            path.write_text(header+good+good)
            self.assertEqual(verify_benchmark(root,[4096,1024,64,256]),2)
            path.write_text(header+good+good.replace("256.0,SUCCESS","255.0,FAILED"))
            with self.assertRaises(RuntimeError):
                verify_benchmark(root,[4096,1024,64,256])


if __name__=="__main__":
    unittest.main()
