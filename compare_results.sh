#!/bin/bash
# ==============================================================================
# compare_results.sh — Compare results from two autoresearch runs
# ==============================================================================
# Usage: ./compare_results.sh <tag_a> <tag_b>
# Example: ./compare_results.sh gpt54 spark
# ==============================================================================

set -euo pipefail

TAG_A="${1:?Usage: $0 <tag_a> <tag_b>}"
TAG_B="${2:?Usage: $0 <tag_a> <tag_b>}"

DIR_A="$HOME/autoresearch_${TAG_A}"
DIR_B="$HOME/autoresearch_${TAG_B}"

for DIR in "$DIR_A" "$DIR_B"; do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: $DIR does not exist"
        exit 1
    fi
done

echo "============================================"
echo "  Autoresearch A/B Comparison"
echo "  A: ${TAG_A} ($(head -3 "$DIR_A/timing.log" 2>/dev/null | grep Model | awk '{print $2}'))"
echo "  B: ${TAG_B} ($(head -3 "$DIR_B/timing.log" 2>/dev/null | grep Model | awk '{print $2}'))"
echo "============================================"
echo ""

for TAG in "$TAG_A" "$TAG_B"; do
    DIR="$HOME/autoresearch_${TAG}"
    RESULTS="$DIR/results.tsv"
    TIMING="$DIR/timing.log"

    echo "--- ${TAG} ---"

    if [ ! -f "$RESULTS" ]; then
        echo "  No results.tsv found"
        echo ""
        continue
    fi

    TOTAL=$(( $(wc -l < "$RESULTS") - 1 ))
    KEPT=$(grep -c "keep" "$RESULTS" || true)
    DISCARDED=$(grep -c "discard" "$RESULTS" || true)
    CRASHED=$(grep -c "crash" "$RESULTS" || true)

    BASELINE_BPB=$(grep "baseline\|keep" "$RESULTS" | head -1 | cut -f2)
    BEST_BPB=$(grep "keep" "$RESULTS" | sort -t$'\t' -k2 -n | head -1 | cut -f2)

    # Calculate improvement
    if [ -n "$BASELINE_BPB" ] && [ -n "$BEST_BPB" ]; then
        IMPROVEMENT=$(python3 -c "print(f'{(($BASELINE_BPB - $BEST_BPB) / $BASELINE_BPB) * 100:.3f}%')" 2>/dev/null || echo "N/A")
    else
        IMPROVEMENT="N/A"
    fi

    # Timing stats
    TOTAL_ITERS=0
    AVG_ITER="N/A"
    if [ -f "$TIMING" ]; then
        TOTAL_ITERS=$(grep -c "^\\[${TAG}\\] iter=" "$TIMING" | head -1 || echo "0")
        TOTAL_ITERS=$((TOTAL_ITERS / 2))  # each iter has start + end lines
        ELAPSED_TIMES=$(grep "elapsed=" "$TIMING" | sed 's/.*elapsed=\([0-9]*\)s.*/\1/')
        if [ -n "$ELAPSED_TIMES" ]; then
            AVG_ITER=$(echo "$ELAPSED_TIMES" | awk '{sum+=$1; n++} END {if(n>0) printf "%.0fs", sum/n; else print "N/A"}')
        fi
    fi

    echo "  Experiments:     $TOTAL"
    echo "  Accepted:        $KEPT ($( [ "$TOTAL" -gt 0 ] && python3 -c "print(f'{$KEPT/$TOTAL*100:.0f}%')" || echo "0%"))"
    echo "  Discarded:       $DISCARDED"
    echo "  Crashed:         $CRASHED"
    echo "  Baseline val_bpb: $BASELINE_BPB"
    echo "  Best val_bpb:     $BEST_BPB"
    echo "  Improvement:      $IMPROVEMENT"
    echo "  Total iterations: $TOTAL_ITERS"
    echo "  Avg time/iter:    $AVG_ITER"
    echo ""

    echo "  Accepted changes:"
    grep "keep" "$RESULTS" | while IFS=$'\t' read -r commit bpb mem status desc; do
        echo "    $bpb  $desc"
    done
    echo ""
done

echo "============================================"
echo "  Head-to-head"
echo "============================================"
BEST_A=$(grep "keep" "$DIR_A/results.tsv" 2>/dev/null | sort -t$'\t' -k2 -n | head -1 | cut -f2)
BEST_B=$(grep "keep" "$DIR_B/results.tsv" 2>/dev/null | sort -t$'\t' -k2 -n | head -1 | cut -f2)

if [ -n "$BEST_A" ] && [ -n "$BEST_B" ]; then
    WINNER=$(python3 -c "
a, b = $BEST_A, $BEST_B
if a < b:
    print(f'${TAG_A} wins: {a} < {b} (better by {(b-a)/b*100:.3f}%)')
elif b < a:
    print(f'${TAG_B} wins: {b} < {a} (better by {(a-b)/a*100:.3f}%)')
else:
    print('Tie')
")
    echo "  $WINNER"
else
    echo "  Cannot compare — missing results"
fi
echo ""
