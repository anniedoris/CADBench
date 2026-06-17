#!/bin/bash
# Converts JSON arrays to JSONL and runs eval for each VLM and split

TESTED_DIR="tested_vlms"
RESULTS_DIR="results"
NUM_WORKERS="${1:-16}"
mkdir -p "$RESULTS_DIR"

for vlm_dir in "$TESTED_DIR"/*/; do
    vlm_name=$(basename "$vlm_dir")
    echo "=== Processing $vlm_name ==="

    for split_dir in "$vlm_dir"*/; do
        split_name=$(basename "$split_dir")
        # e.g. bench0_claude -> bench0
        split=$(echo "$split_name" | sed 's/_[^_]*$//')

        out_jsonl="$RESULTS_DIR/${vlm_name}_${split_name}.jsonl"

        # Merge easy + medium + hard JSON arrays into one JSONL
        python3 -c "
import json, sys, glob, os
split_dir = sys.argv[1]
out = sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(split_dir, '*.json'))):
    rows.extend(json.load(open(f)))
with open(out, 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + '\n')
print(f'  Wrote {len(rows)} rows to {out}')
" "$split_dir" "$out_jsonl"

        metrics_out="$RESULTS_DIR/${vlm_name}_${split_name}_metrics.json"
        logs_out="$RESULTS_DIR/${vlm_name}_${split_name}_logs.json"

        echo "  Running eval for split=$split ..."
        python -m scripts.eval_jsonl "$out_jsonl" \
            --split "$split" \
            --num-workers "$NUM_WORKERS" \
            --variable-name result \
            --metrics-out "$metrics_out" \
            --logs-out "$logs_out"
    done
done
