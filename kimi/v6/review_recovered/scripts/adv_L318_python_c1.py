
import sys,json,collections
rows=[json.loads(l) for l in sys.stdin if l.strip()]
print(len(rows)); print(list(rows[0].keys())[:20])
