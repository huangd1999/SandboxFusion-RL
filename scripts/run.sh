#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"/..

# Port priority: first command-line argument > _BYTEFAAS_RUNTIME_PORT > 8080.
PORT="${1:-${_BYTEFAAS_RUNTIME_PORT:-8080}}"

CONFIG_WORKERS=$(python3 -c "import yaml; print(yaml.safe_load(open('sandbox/configs/local.yaml'))['sandbox']['max_concurrency'])" 2>/dev/null)
WORKERS="${2:-${CONFIG_WORKERS:-192}}"

make run-online HOST="''" PORT=${PORT} WORKERS=${WORKERS}