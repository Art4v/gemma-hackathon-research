cd /Users/hrudaynara/Projects/gemma-work/a6_review_recovered/regen/review/adv; python3 -I -c "
import json
print(json.dumps(open('final_safe_list.txt').read().split()))
" | head -c 9000