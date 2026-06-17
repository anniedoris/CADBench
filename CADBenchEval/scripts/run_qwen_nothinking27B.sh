#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADBENCH_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$CADBENCH_EVAL_DIR"

OUTPUT_ROOT="../tested_models"
DATASET="DeCoDELab/CADBench"
MODEL="Qwen/Qwen3.5-27B"
API_ENDPOINT="http://localhost:8000/v1"
OUTPUT_DIR="$OUTPUT_ROOT/qwen3.527b_nothinking"
IMAGE_FIELDS=(
    "singleview_image"
    "pbr"
)

SPLITS=(
    "bench0"
    "bench0F"
    "bench1A"
    "bench1B"
    "bench2"
    "bench3"
)

mkdir -p "$OUTPUT_DIR"

for image_field in "${IMAGE_FIELDS[@]}"; do
    for split_name in "${SPLITS[@]}"; do
        if [[ "$image_field" == "singleview_image" && "$split_name" == "bench0" ]]; then
            echo "Skipping already-finished run for split=${split_name}, image_field=${image_field}"
            continue
        fi

        output_path="$OUTPUT_DIR/model_output_${split_name}_${image_field}.jsonl"

        echo "Running eval_vlm for thinking_mode=nothinking, split=${split_name}, image_field=${image_field}"

        python -m scripts.eval_vlm \
            --dataset "$DATASET" \
            --split "$split_name" \
            --image-field "$image_field" \
            --id-field file_id \
            --model "$MODEL" \
            --api-endpoint "$API_ENDPOINT" \
            --disable-thinking \
            --temperature 0.7 \
            --top-p 0.8 \
            --top-k 20 \
            --presence-penalty 1.5 \
            --max-tokens 4096 \
            --extra-body-json '{"min_p": 0.0, "repetition_penalty": 1.0}' \
            --jsonl-only \
            --generated-out "$output_path"
    done
done
