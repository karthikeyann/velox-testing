# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Parse explicit, reproducible Presto session settings for benchmark runs."""

import re

_PROPERTY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?")


def parse_session_properties(settings: list[str]) -> dict[str, str]:
    """Parse NAME=VALUE settings without SQL interpolation or silent overrides."""
    properties = {}
    for setting in settings:
        name, separator, value = setting.partition("=")
        if not separator or not _PROPERTY_NAME.fullmatch(name):
            raise ValueError(f"Invalid session property {setting!r}; expected NAME=VALUE")
        if name in properties:
            raise ValueError(f"Duplicate session property: {name}")
        properties[name] = value
    return properties
