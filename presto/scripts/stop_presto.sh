#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Compute the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PRESTO_IMAGE_TAG}" ]; then
  export PRESTO_IMAGE_TAG="${USER:-latest}"
fi

GPU_FILE="${SCRIPT_DIR}/../docker/docker-compose/generated/docker-compose.native-gpu.rendered.yml"
JAVA_FILE="${SCRIPT_DIR}/../docker/docker-compose.java.yml"
CPU_FILE="${SCRIPT_DIR}/../docker/docker-compose.native-cpu.yml"
# Native Presto intentionally delays shutdown; allow it to flush diagnostics.
STOP_TIMEOUT_SECONDS="${PRESTO_STOP_TIMEOUT_SECONDS:-60}"

# Bring down each variant independently to avoid path resolution issues when combining files
docker compose -f "$JAVA_FILE" down --timeout "$STOP_TIMEOUT_SECONDS"
docker compose -f "$CPU_FILE" down --timeout "$STOP_TIMEOUT_SECONDS"
if [ -f "$GPU_FILE" ]; then
  docker compose -f "$GPU_FILE" down --timeout "$STOP_TIMEOUT_SECONDS"
fi
