#!/usr/bin/env bash
# Copyright 2025-present Kensho Technologies, LLC.
set -euxo pipefail

exec python -m pytest -s --cov=grits "$@"
