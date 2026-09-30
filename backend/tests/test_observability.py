"""Operational endpoints: metrics, request correlation and health probes.

These run without a database. Where an endpoint's answer depends on the
database, the dependency is replaced so both branches are exercised wherever
the suite runs.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from quantedge.api import main
from quantedge.api.main import app
from quantedge.config import settings
from quantedge.logging_config import JsonFormatter
from quantedge.observability.context import RequestIdFilter, request_id_var, resolve_request_id

HEADERS = {"X-API-Key": settings.api_key}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestMetrics:
    def test_exposition_format(self, client):
        client.get("/health/live")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "quantedge_http_requests_total" in body
        assert "quantedge_http_request_duration_seconds_bucket" in body
        assert "quantedge_metrics_db_up" in body

    def test_routes_are_labelled_by_template_not_path(self, client):
        client.get("/v1/microstructure/study/NOTASYMBOL", headers=HEADERS)
        client.get("/definitely/not/a/route")
        body = client.get("/metrics").text
        assert 'route="/v1/microstructure/study/{symbol}"' in body
        assert "NOTASYMBOL" not in body
        # Unknown paths share one label, so probing random URLs cannot create
        # unbounded series.
        assert 'route="unmatched"' in body
        assert "/definitely/not/a/route" not in body

    def test_token_is_enforced_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(settings, "metrics_token", "s3cret")
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


class TestRequestId:
    def test_generated_when_absent(self, client):
        rid = client.get("/health/live").headers["X-Request-ID"]
        assert len(rid) == 32

    def test_propagated_from_caller(self, client):
        resp = client.get("/health/live", headers={"X-Request-ID": "lb-abc123"})
        assert resp.headers["X-Request-ID"] == "lb-abc123"

    def test_unsafe_ids_are_replaced(self):
        assert resolve_request_id("ok-id.1") == "ok-id.1"
        assert resolve_request_id("bad id\nwith newline") != "bad id\nwith newline"
        assert resolve_request_id("x" * 500) != "x" * 500

    def test_log_records_carry_the_id(self):
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        token = request_id_var.set("req-42")
        try:
            RequestIdFilter().filter(record)
        finally:
            request_id_var.reset(token)
        line = JsonFormatter().format(record)
        assert '"request_id": "req-42"' in line
        assert '"message": "hello world"' in line


class TestHealthProbes:
    def test_liveness_never_touches_dependencies(self, client, monkeypatch):
        def boom():
            raise AssertionError("liveness must not check the database")

        monkeypatch.setattr(main.system, "health", boom)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    @pytest.mark.parametrize(
        ("database", "expected"), [("connected", 200), ("unreachable", 503)]
    )
    def test_readiness_follows_the_database(self, client, monkeypatch, database, expected):
        monkeypatch.setattr(
            main.system, "health", lambda: {"status": "x", "database": database}
        )
        resp = client.get("/health/ready")
        assert resp.status_code == expected
        assert resp.json()["instance"] == settings.instance_name


class TestMicrostructureApi:
    def test_requires_auth(self, client):
        assert client.get("/v1/microstructure/summary").status_code == 401

    def test_unknown_symbol_is_404(self, client):
        resp = client.get("/v1/microstructure/study/NOTASYMBOL", headers=HEADERS)
        assert resp.status_code == 404


class TestAlertWebhook:
    PAYLOAD = {
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "ApiReplicasBelowTarget", "severity": "warning"},
                "annotations": {"summary": "Fewer than 2 API replicas are up"},
                "startsAt": "2026-09-30T10:00:00Z",
            }
        ]
    }

    def test_accepts_bearer_as_alertmanager_sends_it(self, client):
        resp = client.post(
            "/v1/system/alerts",
            json=self.PAYLOAD,
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"received": 1}

        recent = client.get("/v1/system/alerts", headers=HEADERS).json()
        assert recent["alerts"][0]["alertname"] == "ApiReplicasBelowTarget"

    def test_rejects_wrong_token(self, client):
        resp = client.post(
            "/v1/system/alerts", json=self.PAYLOAD, headers={"Authorization": "Bearer nope"}
        )
        assert resp.status_code == 401

    def test_alert_is_written_to_the_log(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="quantedge.api.routers.system"):
            client.post("/v1/system/alerts", json=self.PAYLOAD, headers=HEADERS)
        assert any("ApiReplicasBelowTarget" in r.getMessage() for r in caplog.records)
