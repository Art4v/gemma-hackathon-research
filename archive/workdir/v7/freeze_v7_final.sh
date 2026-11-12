#!/usr/bin/env bash
# Freeze v7: ingest Karthik's round-3 verdicts, assemble with the agreed flags, run every check, and only if ALL CHECKS PASS commit
# kimi/v7/v7_final.jsonl (the 04:30 file: 03:30 build + layout rows [+ gen7R] + late round-3) to hruday/kimi-k31 and push.
set -euo pipefail
W=/Users/hrudaynara/Projects/gemma-work; V=$W/v7; PYV=$W/venv/bin/python; K=$W/wt-kimi/kimi/v7
cd $V
$PYV ingest_karthik.py
$PYV assemble_v7.py --drop-slow matmul --exam-slow drop --gen v7_gen_rows_surf_add.jsonl --layout v7_layout_rows_surf_add.jsonl --gen7r v7_rowout_rows_surf_add.jsonl --out v7_final | tail -2
nice -n 10 $PYV check_v7.py v7_final.jsonl v6_train.jsonl 2>&1 | grep -v -i "warning\|pytorch" | tee $V/check_v7.out | tail -3
grep -q "ALL CHECKS PASS" $V/check_v7.out || { echo "CHECKS FAILED: not pushing"; exit 1; }
shasum -a 256 v7_final.jsonl | awk '{print $1"  v7_final.jsonl"}' > v7_final.jsonl.sha256
cp v7_final.jsonl v7_final_val.jsonl v7_final_stats.json v7_final.jsonl.sha256 check_v7.out dev_items_gen.jsonl v7_layout_rows.jsonl $K/; mkdir -p $K/layout && cp layout/*.jsonl $K/layout/; cp a6r2/aarav_drops_latest.json a6r2/gen7R_drops.json a6r2/layout_drops_mine.json $K/a6r2/ 2>/dev/null || true
cp assemble_v7.py check_v7.py run_fix2.py run_r3.py step1_v7r.py ingest_karthik.py v7r_pipeline2.sh r3/gap.py $K/
cp surface_gen.py surface_neardup.py v7_gen_rows_surf_add.jsonl v7_layout_rows_surf_add.jsonl v7_rowout_rows.jsonl v7_rowout_rows_surf_add.jsonl $K/; for d in gen_surface layout_surface rowout_surface; do mkdir -p $K/$d && cp $d/check.json $d/neardup.json $d/variants.jsonl $K/$d/; done
mkdir -p $K/a6r2 && cp a6r2/base_drops.json a6r2/drops.json a6r2/aarav_drops.json a6r2/mm_led_slow_problems.json a6r2/compare.py a6r2/compare2.py a6r2/trace_child.py $K/a6r2/
cd $W/wt-kimi && git add kimi/v7 && git commit -q -m "v7 FINAL (04:30 file): $(wc -l < $V/v7_final.jsonl | tr -d ' ') rows, sha256 $(cut -c1-12 $V/v7_final.jsonl.sha256); val $(wc -l < $V/v7_final_val.jsonl | tr -d ' ') rows. A6 base minus Aarav 4, minus A6 rule-2 matches, minus below-bar matmul-led rows; scaled passes + fix rounds 1-2 + round 3 (v4 gap); generated rows minus rule-2 drops and the shared dev set; pass-only targets for added rows; history turns in eval form. Counts: kimi/v7/v7_final_stats.json; built on the 03:30 file ddfb79d (rows unchanged) + verified layout rows + surface variants of the generated and layout rows (Hruday confirmed 03:38Z; AST-proven, originals TPU-verified) + late round-3 passes; decisions on #5.

Claude-Session: https://claude.ai/code/session_01EHerPevrrY2t5FnSZt6YfF" && git push -q origin HEAD:hruday/kimi-k31
echo "FINAL $(git rev-parse --short HEAD) $(cat $V/v7_final.jsonl.sha256)"
$PYV -c "import json; s=json.load(open('$V/v7_final_stats.json')); print(s['rows'], json.dumps(s['by_set_kind']))"
