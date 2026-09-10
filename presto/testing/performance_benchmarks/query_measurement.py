# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Measure complete executions, including consumption of every result page."""

from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class QueryMeasurement:
    rows: list
    query_id: str
    client_elapsed_ms: float
    server_elapsed_ms: int
    stats: dict


def execute_measured_query(cursor, query: str) -> QueryMeasurement:
    """Drain the cursor before reading final server stats or starting another query.

    DBAPI execute() can return while the query still has unread result pages.
    Client elapsed time includes submission and result transfer; server elapsed
    time comes from the final statement response. Result persistence and extra
    diagnostic HTTP requests are outside both measurements.
    """
    start = perf_counter_ns()
    cursor.execute(query)
    rows = cursor.fetchall()
    client_elapsed_ms = (perf_counter_ns() - start) / 1_000_000
    stats = dict(cursor.stats)
    if stats.get("state") != "FINISHED":
        raise RuntimeError(f"Query did not finish after draining results: {stats.get('state')!r}")
    return QueryMeasurement(
        rows=rows,
        query_id=cursor._query.query_id,
        client_elapsed_ms=client_elapsed_ms,
        server_elapsed_ms=stats["elapsedTimeMillis"],
        stats=stats,
    )
