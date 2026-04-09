import importlib.util
import sys
import types
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def worker_module(monkeypatch: MonkeyPatch):
    module_name = "worker_module_under_test"
    module_path = Path(__file__).resolve().parents[1] / "worker" / "worker.py"

    fake_bigquery = types.SimpleNamespace()
    fake_google_cloud = types.SimpleNamespace(bigquery=fake_bigquery)
    fake_google = types.SimpleNamespace(cloud=fake_google_cloud)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bigquery)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_partitioned_load_queries_source_and_target(worker_module, monkeypatch: MonkeyPatch, capsys):
    captured = {}

    class FakeQueryJob:
        def result(self):
            captured["result_called"] = True

    # Fake BigQuery client used to capture SQL/job_config and track close().
    class FakeClient:
        def query(self, query, job_config=None):
            captured["query"] = query
            captured["job_config"] = job_config
            return FakeQueryJob()

        def close(self):
            captured["closed"] = True

    # Minimal replacement for QueryJobConfig to verify destination/write mode.
    class FakeQueryJobConfig:
        def __init__(self, destination, write_disposition):
            self.destination = destination
            self.write_disposition = write_disposition

    monkeypatch.setattr(worker_module.bigquery, "Client", lambda project: FakeClient(), raising=False)
    monkeypatch.setattr(worker_module.bigquery, "QueryJobConfig", FakeQueryJobConfig, raising=False)

    worker_module.run_partitioned_load()

    assert worker_module.src_dtable in captured["query"]
    assert captured["job_config"].destination == worker_module.target_table
    assert captured["job_config"].write_disposition == "WRITE_TRUNCATE"
    assert captured["result_called"] is True
    assert captured["closed"] is True

    stdout = capsys.readouterr().out
    assert worker_module.target_table in stdout
