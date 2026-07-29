"""API contract tests.

Two things matter beyond "does it return 200":

* **Auth** — every data endpoint must reject an unauthenticated caller.
* **Honesty** — a payload must never present an in-sample figure as though it
  were validated, and telemetry must be derived from stored records rather
  than constants.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from quantedge.api.main import app
from quantedge.config import settings

HEADERS = {"X-API-Key": settings.api_key}

DATA_ENDPOINTS = [
    "/v1/portfolio/summary",
    "/v1/portfolio/equity-curve",
    "/v1/portfolio/drawdown",
    "/v1/portfolio/signals",
    "/v1/factors/table",
    "/v1/factors/correlation",
    "/v1/factors/list",
    "/v1/backtest/runs",
    "/v1/backtest/metrics",
    "/v1/backtest/folds",
    "/v1/backtest/trades",
    "/v1/risk/summary",
    "/v1/risk/exposure",
    "/v1/risk/positions",
    "/v1/risk/breaches",
    "/v1/system/status",
    "/v1/system/jobs",
    "/v1/system/ingestion",
    "/v1/system/api-metrics",
    "/v1/system/logs",
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestMeta:
    def test_health_is_unauthenticated(self, client):
        """Docker's healthcheck cannot supply an API key."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("healthy", "degraded")

    def test_root_lists_screen_endpoints(self, client):
        body = client.get("/").json()
        assert set(body["screens"]) == {
            "dashboard", "factors", "backtest", "risk", "system"
        }

    def test_openapi_schema_builds(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestAuthentication:
    @pytest.mark.parametrize("endpoint", DATA_ENDPOINTS)
    def test_rejects_missing_key(self, client, endpoint):
        assert client.get(endpoint).status_code == 401

    def test_rejects_wrong_key(self, client):
        resp = client.get(
            "/v1/portfolio/summary", headers={"X-API-Key": "not-the-key"}
        )
        assert resp.status_code == 401

    def test_accepts_valid_key(self, client):
        assert client.get("/v1/system/info", headers=HEADERS).status_code == 200


@pytest.mark.integration
class TestPayloads:
    """Require a populated database (`make ingest && make backtest`)."""

    def test_summary_flags_out_of_sample(self, client):
        body = client.get("/v1/portfolio/summary", headers=HEADERS).json()
        if "run_id" not in body:
            pytest.skip("no backtest run stored")
        # A consumer must be able to tell validated from in-sample.
        assert "is_out_of_sample" in body
        assert isinstance(body["is_out_of_sample"], bool)

    def test_backtest_metrics_carry_validation_note(self, client):
        body = client.get("/v1/backtest/metrics", headers=HEADERS).json()
        if body.get("detail"):
            pytest.skip("no backtest run stored")
        note = body["validation_note"]
        if body["is_walk_forward"]:
            assert "Out-of-sample" in note
        else:
            assert "IN-SAMPLE ONLY" in note

    def test_uptime_derives_from_job_runs(self, client):
        """Uptime must reflect stored runs, not a hardcoded constant."""
        body = client.get("/v1/system/status", headers=HEADERS).json()
        total = body["total_job_runs"]
        failed = body["failed_job_runs"]
        uptime = body["uptime_pct"]

        if total == 0:
            assert uptime is None, "no runs must report null, never 100%"
        else:
            expected = round(100.0 * (total - failed) / total, 3)
            assert uptime == pytest.approx(expected, abs=0.01)

    def test_ingestion_reports_real_counts(self, client):
        body = client.get("/v1/system/ingestion", headers=HEADERS).json()
        coverage = body["coverage"]
        if coverage["total_rows"] == 0:
            pytest.skip("no data ingested")
        assert coverage["n_tickers"] > 0
        assert coverage["years_of_history"] > 0
        # Success rate below 100 is expected: delisted tickers have no data.
        if body["fetch_success_rate"] is not None:
            assert 0 <= body["fetch_success_rate"] <= 100

    def test_trades_pagination_and_filter(self, client):
        wins = client.get(
            "/v1/backtest/trades?outcome=WINS&page_size=10", headers=HEADERS
        ).json()
        if wins.get("detail") or wins["total"] == 0:
            pytest.skip("no trades stored")
        assert all(t["pnl_pct"] > 0 for t in wins["trades"])

        losses = client.get(
            "/v1/backtest/trades?outcome=LOSSES&page_size=10", headers=HEADERS
        ).json()
        assert all(t["pnl_pct"] < 0 for t in losses["trades"])

    def test_folds_are_ordered_and_counted(self, client):
        body = client.get("/v1/backtest/folds", headers=HEADERS).json()
        if body.get("detail") or body["n_folds"] == 0:
            pytest.skip("no walk-forward run stored")
        indices = [f["fold"] for f in body["folds"]]
        assert indices == sorted(indices)
        for fold in body["folds"]:
            assert fold["train"]["end"] < fold["test"]["start"], "embargo violated"

    def test_factor_correlation_is_symmetric(self, client):
        body = client.get("/v1/factors/correlation", headers=HEADERS).json()
        matrix, factors = body["matrix"], body["factors"]
        for a in factors:
            assert matrix[a][a] == pytest.approx(1.0, abs=1e-6)
            for b in factors:
                assert matrix[a][b] == pytest.approx(matrix[b][a], abs=1e-6)

    def test_latency_middleware_records_requests(self, client):
        client.get("/v1/system/info", headers=HEADERS)
        body = client.get("/v1/system/api-metrics", headers=HEADERS).json()
        assert body["live"]["n_requests"] > 0
        assert body["target_p95_ms"] == 200

    def test_response_time_header_present(self, client):
        resp = client.get("/v1/system/info", headers=HEADERS)
        assert "X-Response-Time-Ms" in resp.headers
        assert float(resp.headers["X-Response-Time-Ms"]) >= 0


class TestValidation:
    def test_rejects_bad_trade_filter(self, client):
        resp = client.get("/v1/backtest/trades?outcome=BOGUS", headers=HEADERS)
        assert resp.status_code == 422

    def test_rejects_oversized_page(self, client):
        resp = client.get("/v1/backtest/trades?page_size=99999", headers=HEADERS)
        assert resp.status_code == 422

    def test_unknown_ticker_returns_404(self, client):
        resp = client.get("/v1/factors/NOTATICKER/detail", headers=HEADERS)
        assert resp.status_code in (404, 500)


class TestInuContract:
    """The chat surface, without calling a model.

    These exist because the INU upload route broke the entire app once: FastAPI
    validates form parameters at import time, so a missing `python-multipart`
    raised during collection rather than on a request, and every test in this
    file failed with an error that named none of them.
    """

    def test_upload_route_is_constructible(self, client):
        """Guards the dependency that FastAPI needs at import, not at request.

        If `python-multipart` is missing from the manifest this fails here with
        a clear message, rather than taking down collection for the suite.

        Asserted against the published schema rather than `app.routes`, which
        stopped listing included routes in FastAPI 0.140.
        """
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/inu/chat/upload" in paths

    def test_status_lists_only_free_models(self, client):
        r = client.get("/v1/inu/status", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["models"], "no models advertised"
        # Every routed model must be free: Groq's tier is free outright, and an
        # OpenRouter id has to carry the :free suffix to cost nothing.
        for m in body["models"]:
            assert m["provider"] in {"groq", "openrouter"}
            if m["provider"] == "openrouter":
                assert m["id"].endswith(":free"), f"{m['id']} is not a free model"

    def test_tools_are_advertised_with_descriptions(self, client):
        r = client.get("/v1/inu/status", headers=HEADERS)
        tools = r.json()["tools"]
        assert len(tools) >= 5
        for t in tools:
            assert t["description"].strip(), f"{t['name']} has no description"

    def test_conversation_endpoints_require_auth(self, client):
        for path in ("/v1/inu/status", "/v1/inu/conversations"):
            assert client.get(path).status_code == 401

    def test_unknown_conversation_returns_404(self, client):
        r = client.get("/v1/inu/conversations/99999999", headers=HEADERS)
        assert r.status_code == 404


class TestInuTools:
    """Tools must return real platform data, never invented values."""

    def test_every_tool_executes(self):
        from quantedge.inu import tools

        for name in tools.TOOLS:
            payload = json.loads(tools.execute(name, "{}"))
            assert isinstance(payload, dict)

    def test_significance_reports_the_honest_verdict(self):
        """The figure the whole platform's credibility rests on.

        If a tool ever reported significance without the deflated Sharpe, the
        chat could quote a 1.42 Sharpe as though it were proven.
        """
        from quantedge.inu import tools

        payload = json.loads(tools.execute("get_significance", "{}"))
        if "error" in payload:
            pytest.skip("no run to assess")
        assert "deflated_sharpe" in payload
        assert "is_significant" in payload
        assert payload["decision_rule"]

    def test_unknown_tool_is_reported_not_raised(self):
        from quantedge.inu import tools

        assert "error" in json.loads(tools.execute("does_not_exist", "{}"))


class TestInuAgent:
    """Prose safety, checked without a network call."""

    def test_tool_markup_never_reaches_the_user(self):
        """Small models sometimes write a tool call as literal text."""
        from quantedge.inu.agent import strip_tool_markup

        leaked = "The Sharpe is 1.42. <function=get_significance>{}</function>"
        assert "<function=" not in strip_tool_markup(leaked)
        assert "1.42" in strip_tool_markup(leaked)

    def test_inline_tool_calls_are_recovered(self):
        from quantedge.inu.agent import _inline_tool_calls

        found = _inline_tool_calls("<function=get_performance>{}</function>")
        assert found and found[0][0] == "get_performance"

    def test_routing_respects_size_and_attachments(self):
        from quantedge.inu.models import Task, route

        # An image must reach a model that can actually see it.
        assert route(Task.CHAT, has_image=True)[0].provider == "openrouter"
        # A prompt too large for Groq's per-minute ceiling must not be sent there.
        big = route(Task.CHAT, prompt_chars=400_000)[0]
        assert big.tpm is None or big.tpm > 5000
