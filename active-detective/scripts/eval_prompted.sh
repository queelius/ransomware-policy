#!/usr/bin/env bash
# Batch evaluation: run Claude against each scenario in a manifest.
#
# Usage:
#   scripts/eval_prompted.sh [manifest.jsonl] [model]
#   scripts/eval_prompted.sh eval_manifest.jsonl claude-opus-4-6
set -euo pipefail

MANIFEST=${1:-"eval_manifest.jsonl"}
MODEL=${2:-claude-opus-4-6}
RESULTS="results/prompted_${MODEL}_$(date +%Y%m%d_%H%M%S).jsonl"
PROMPT_FILE="prompts/eval_prompt.md"

mkdir -p results

# Generate manifest if not provided
if [ ! -f "$MANIFEST" ]; then
    echo "Generating manifest..."
    python scripts/generate_eval_manifest.py --n 200 --output "$MANIFEST"
fi

COUNT=$(wc -l < "$MANIFEST")
echo "Running $COUNT scenarios with $MODEL..."
echo "Results: $RESULTS"

i=0
while IFS= read -r line; do
    i=$((i + 1))
    seed=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['seed'])")
    stype=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['scenario_type'])")
    obs=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['observability'])")
    prog=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['attack_progress'])")

    echo "[$i/$COUNT] seed=$seed type=$stype obs=$obs progress=$prog"

    SCENARIO_SEED=$seed SCENARIO_TYPE=$stype OBSERVABILITY=$obs \
    ATTACK_PROGRESS=$prog RESULTS_FILE="$RESULTS" \
        claude -p "$(cat "$PROMPT_FILE")" \
        --model "$MODEL" \
        --mcp-config .mcp.json \
        2>/dev/null || echo "  WARN: scenario $seed failed"

done < "$MANIFEST"

echo ""
echo "=== Computing metrics ==="
python scripts/compute_metrics.py "$RESULTS"
