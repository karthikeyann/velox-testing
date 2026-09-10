# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for measuring complete, rather than first-page, queries."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from common.testing.performance_benchmarks.benchmark_keys import BenchmarkKeys
from presto.testing.performance_benchmarks import common_fixtures, query_measurement


class Cursor:
    def __init__(self, rows, final_state="FINISHED", execute_error=None, fetch_error=None):
        self.rows = rows
        self.final_state = final_state
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.stats = {"state": "RUNNING", "elapsedTimeMillis": 3}
        self._query = SimpleNamespace(query_id="query-1")
        self.events = []

    def execute(self, query):
        self.events.append(("execute", query))
        if self.execute_error:
            raise self.execute_error
        return self

    def fetchall(self):
        self.events.append(("fetchall",))
        if self.fetch_error:
            raise self.fetch_error
        self.stats.update(state=self.final_state, elapsedTimeMillis=20)
        return self.rows


@pytest.mark.parametrize("rows", [[], [[1]], [[1], [2], [3]]])
def test_reads_final_stats_after_draining(monkeypatch, rows):
    clock = iter([1_000_000_000, 1_025_000_000])
    monkeypatch.setattr(query_measurement, "perf_counter_ns", lambda: next(clock))
    cursor = Cursor(rows)

    result = query_measurement.execute_measured_query(cursor, "SELECT x FROM t")

    assert cursor.events == [("execute", "SELECT x FROM t"), ("fetchall",)]
    assert result.rows == rows
    assert result.query_id == "query-1"
    assert result.client_elapsed_ms == 25
    assert result.server_elapsed_ms == 20
    assert result.stats["state"] == "FINISHED"
    cursor.stats["elapsedTimeMillis"] = 99
    assert result.stats["elapsedTimeMillis"] == 20


@pytest.mark.parametrize("phase", ["execute", "fetch"])
def test_propagates_execution_and_late_page_errors(phase):
    error = RuntimeError("query failed")
    cursor = Cursor([], **{f"{phase}_error": error})
    with pytest.raises(RuntimeError, match="query failed"):
        query_measurement.execute_measured_query(cursor, "SELECT x FROM t")


def test_rejects_nonterminal_stats():
    with pytest.raises(RuntimeError, match="did not finish"):
        query_measurement.execute_measured_query(Cursor([], final_state="RUNNING"), "SELECT 1")


def test_benchmark_preserves_every_complete_iteration(tmp_path):
    options = {
        "--iterations": 3,
        "--profile": False,
        "--profile-script-path": None,
        "--metrics": False,
        "--hostname": "localhost",
        "--port": 8080,
        "--cache-mode": "hot",
        "--connector-id": "hive",
        "--skip-drop-cache": True,
        "--output-dir": str(tmp_path),
        "--tag": None,
    }
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=options.get),
        node=SimpleNamespace(obj=SimpleNamespace(BENCHMARK_TYPE="tpch")),
    )
    cursor = Cursor([[1], [2]])
    cursor.description = [("x",)]
    collector = {}
    run_query = common_fixtures.benchmark_query.__wrapped__(request, cursor, {"Q1": "SELECT x FROM t"}, collector, None)

    run_query("Q1")

    assert collector["tpch"][BenchmarkKeys.RAW_TIMES_KEY]["Q1"] == [20, 20, 20]
    assert sum(event == ("fetchall",) for event in cursor.events) == 3
    assert pd.read_parquet(tmp_path / "query_results/q1.parquet")["x"].tolist() == [1, 2]
    for iteration in range(3):
        rows = pd.read_parquet(tmp_path / f"query_results/iterations/q1_{iteration:03d}.parquet")
        assert rows["x"].tolist() == [1, 2]
        measurement = json.loads((tmp_path / f"measurements/Q1/{iteration:03d}.json").read_text())
        assert measurement["row_count"] == 2
        assert measurement["server_elapsed_ms"] == 20
        assert measurement["stats"]["state"] == "FINISHED"
