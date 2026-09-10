# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from presto.testing.performance_benchmarks import common_fixtures
from presto.testing.performance_benchmarks.session_properties import parse_session_properties


def test_empty_settings_preserve_defaults():
    assert parse_session_properties([]) == {}


def test_system_catalog_and_literal_values():
    assert parse_session_properties(["task_concurrency=1", "hive.some_value=a=b, c", "empty="]) == {
        "task_concurrency": "1",
        "hive.some_value": "a=b, c",
        "empty": "",
    }


@pytest.mark.parametrize("setting", ["missing_value", "=x", "bad name=x", "a.b.c=x", "a;drop=x", "1name=x"])
def test_rejects_malformed_settings(setting):
    with pytest.raises(ValueError, match="expected NAME=VALUE"):
        parse_session_properties([setting])


def test_rejects_duplicate_settings():
    with pytest.raises(ValueError, match="Duplicate session property"):
        parse_session_properties(["task_concurrency=1", "task_concurrency=4"])


def test_passes_settings_through_supported_client_api(monkeypatch):
    captured = {}
    cursor = object()

    def connect(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(cursor=lambda: cursor)

    monkeypatch.setattr(common_fixtures.prestodb.dbapi, "connect", connect)
    options = {
        "--hostname": "localhost",
        "--port": 8080,
        "--user": "test_user",
        "--schema-name": "tpch",
        "--session-property": ["task_concurrency=1", "join_prefilter_build_side=true"],
    }
    request = SimpleNamespace(config=SimpleNamespace(getoption=options.__getitem__))
    assert common_fixtures.presto_cursor.__wrapped__(request) is cursor
    assert captured["session_properties"] == {
        "task_concurrency": "1",
        "join_prefilter_build_side": "true",
    }
