cd /private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv; python3 -c "
import json
print(json.dumps(open('final_safe_list.txt').read().split()))
" | head -c 9000