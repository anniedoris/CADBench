#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADBENCH_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$CADBENCH_EVAL_DIR"

python -m scripts.eval_jsonl_modified ../tested_models/cadfit/noisy_mesh/benchA.jsonl --split benchA --num-workers 128 --variable-name result
