import importlib.util
import sys
import types
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def main_module(monkeypatch: MonkeyPatch):
    # `monkeypatch` is a built-in pytest fixture used to temporarily patch
    # objects/modules during a test and automatically roll those changes back.
    module_name = "api_main_under_test"
    module_path = Path(__file__).resolve().parents[1] / "api" / "main.py"

    class FakeFastAPI:
        def __init__(self, title):
            self.title = title

        def get(self, _path):
            def decorator(func):
                return func

            return decorator

    fake_fastapi_module = types.SimpleNamespace(FastAPI=FakeFastAPI)

    fake_bigquery = types.SimpleNamespace()
    fake_google_cloud = types.SimpleNamespace(bigquery=fake_bigquery)
    fake_google = types.SimpleNamespace(cloud=fake_google_cloud)

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi_module)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bigquery)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_health_endpoint_returns_ok(main_module):
    assert main_module.health() == {"status": "ok"}


def test_top10_trends_uses_bigquery_client_and_returns_rows(main_module, monkeypatch: MonkeyPatch):
    rows = [
        {"trend_item": "Python", "total_volume": 123},
        {"trend_item": "FastAPI", "total_volume": 99},
    ]
    captured = {}

    # Fake BigQuery client context manager used by top10_trends().
    # It captures the SQL string and returns deterministic row data.
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def query(self, query):
            captured["query"] = query
            return rows

    monkeypatch.setattr(main_module.bigquery, "Client", lambda project: FakeClient(), raising=False)

    result = main_module.top10_trends(limit=2)

    assert result == {"limit": 2, "items": rows}
    assert "LIMIT 2" in captured["query"]
    assert main_module.source_db in captured["query"]


def test_by_country_endpoint_passes_parameterized_query(main_module, monkeypatch: MonkeyPatch):
    rows = [{"trend_item": "AI"}, {"trend_item": "Cloud"}]
    captured = {}

    # Fake BigQuery client context manager used by by_country().
    # It captures both SQL text and the job_config object passed to query().
    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def query(self, query, job_config=None):
            captured["query"] = query
            captured["job_config"] = job_config
            return rows

    # Minimal stand-in for bigquery.QueryJobConfig so we can verify
    # query parameters are forwarded correctly to BigQuery.
    class FakeQueryJobConfig:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters

    def fake_scalar_query_parameter(name, typ, value):
        return {"name": name, "type": typ, "value": value}

    monkeypatch.setattr(main_module.bigquery, "Client", lambda project: FakeClient(), raising=False)
    monkeypatch.setattr(main_module.bigquery, "QueryJobConfig", FakeQueryJobConfig, raising=False)
    monkeypatch.setattr(main_module.bigquery, "ScalarQueryParameter", fake_scalar_query_parameter, raising=False)

    result = main_module.by_country(country="United States", limit=2)

    assert result == {
        "country": "United States",
        "limit": 2,
        "items": rows,
    }

    assert "@country" in captured["query"]
    assert "LIMIT 2" in captured["query"]
    assert captured["job_config"].query_parameters == [
        {"name": "country", "type": "STRING", "value": "United States"}
    ]
