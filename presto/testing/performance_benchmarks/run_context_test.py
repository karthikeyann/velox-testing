# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from presto.testing.performance_benchmarks import run_context


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [("1", 1), ("100.0", 100), ("0.5", 0.5)],
)
def test_explicit_scale_factor_skips_schema_queries(monkeypatch, supplied, expected):
    def fail_schema_lookup(*_):
        pytest.fail("explicit scale factor must avoid schema metadata SQL")

    monkeypatch.setattr(run_context, "_get_schema_info", fail_schema_lookup)
    monkeypatch.setattr(run_context, "_get_node_count", lambda *_: 1)
    monkeypatch.setattr(run_context, "_get_engine", lambda *_: "presto-velox-cpu")
    monkeypatch.setattr(run_context, "_get_num_drivers", lambda: None)

    result = run_context.gather_run_context(
        hostname="localhost",
        port=8080,
        user="test_user",
        schema_name="sf100",
        scale_factor=supplied,
    )

    assert result["scale_factor"] == expected
    assert "data_dir" not in result
