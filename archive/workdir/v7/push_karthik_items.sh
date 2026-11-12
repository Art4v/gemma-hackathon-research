#!/usr/bin/env bash
# Push round-3 TPU groups 5 and 6 (Karthik's TPU) to hruday/kimi-k31 kimi/v7/tpu_items/, the moment step 1 has written them.
set -e
W=/Users/hrudaynara/Projects/gemma-work; F=$W/tpu/fleet
until grep -q "STEP1 DONE r3" $W/v7/out/r3/step1.log 2>/dev/null; do sleep 15; done
cd $W/wt-kimi && mkdir -p kimi/v7/tpu_items
for k in 5 6; do cp $F/kimi-r3-s$k.items.jsonl kimi/v7/tpu_items/; done
git add kimi/v7/tpu_items/kimi-r3-s5.items.jsonl kimi/v7/tpu_items/kimi-r3-s6.items.jsonl
git commit -q -m "v7 round 3: TPU items groups 5 and 6 (ids kimiR3/<problem_id>) for Karthik's TPU

Claude-Session: https://claude.ai/code/session_01EHerPevrrY2t5FnSZt6YfF"
git push -q origin HEAD:hruday/kimi-k31
echo "PUSHED $(git rev-parse --short HEAD) $(wc -l < kimi/v7/tpu_items/kimi-r3-s5.items.jsonl) + $(wc -l < kimi/v7/tpu_items/kimi-r3-s6.items.jsonl) items"
