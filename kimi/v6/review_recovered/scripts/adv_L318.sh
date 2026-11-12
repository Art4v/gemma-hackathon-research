cd /Users/hrudaynara/Projects/gemma-dataset-claude 2>/dev/null && git branch -a 2>/dev/null | grep -i "v3\|v4" | head; git -C /Users/hrudaynara/Projects/gemma-dataset-claude show origin/aarav/dataset-v3:dataset/v3/final/train_capped.jsonl 2>/dev/null | head -c 300; echo; git -C /Users/hrudaynara/Projects/gemma-dataset-claude show origin/aarav/dataset-v3:dataset/v3/final/train_capped.jsonl 2>/dev/null | python3 -c "
import sys,json,collections
rows=[json.loads(l) for l in sys.stdin if l.strip()]
print(len(rows)); print(list(rows[0].keys())[:20])
"