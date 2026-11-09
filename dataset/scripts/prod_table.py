"""Per-row contamination table for a batch: each problem's closest JAXBench task by op-multiset Jaccard, that Jaccard, and the
difflib ratio of the two references' text. Read-only; prints markdown.
  python dataset/scripts/prod_table.py [--batch prod]
"""
import difflib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATASET, PROBLEMS, read_jsonl  # noqa: E402
from signature import jaccard  # noqa: E402

ROOT = DATASET.parent
EXAM = json.load(open(PROBLEMS / "exam_signatures.json"))
EXAM.update(json.load(open(PROBLEMS / "exam_signatures_level2.json")))


def exam_code(task: str) -> str:
    for p in (ROOT / "benchmarks/jaxbench/level1" / task / "baseline.py", ROOT / "control/refs" / f"{task}.py"):
        if p.exists():
            return p.read_text()
    return ""


def main() -> None:
    batch = sys.argv[sys.argv.index("--batch") + 1] if "--batch" in sys.argv else "prod"
    print("| problem | status | closest JAXBench task | op-multiset J | text ratio |\n|---|---|---|---|---|")
    for r in read_jsonl(PROBLEMS / "problems.jsonl"):
        if r.get("batch") != batch or not r.get("signature") or "op_multiset" not in r["signature"]:
            continue
        j, t = max((jaccard(r["signature"]["op_multiset"], s["op_multiset"]), k) for k, s in EXAM.items() if "op_multiset" in s)
        code = exam_code(t)
        ratio = f"{difflib.SequenceMatcher(None, r['ref_code'], code, autojunk=False).ratio():.2f}" if code else "n/a"
        print(f"| {r['problem_id'].split('/')[-1]} | {r['status']} | {t} | {j:.2f} | {ratio} |")


if __name__ == "__main__":
    main()
