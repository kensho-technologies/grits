#!/usr/bin/env bash
# Copyright 2025-present Kensho Technologies, LLC.
set -euxo pipefail

fix="";
while getopts ":f" opt; do
  case $opt in
    f) fix="--fix" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

python -m mypy --config-file pyproject.toml <your module name>

if [ "$fix" = "--fix" ]; then
  # The ruff linters (check) and formatters (format) are separate.
  # See https://docs.astral.sh/ruff/formatter/#sorting-imports
  python -m ruff --config pyproject.toml check <your module name> --fix
  python -m ruff --config pyproject.toml format <your module name>
else
  python -m ruff --config pyproject.toml check <your module name>
  python -m ruff --config pyproject.toml format <your module name> --check
fi
