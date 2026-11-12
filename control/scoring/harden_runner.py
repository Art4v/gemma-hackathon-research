"""One-off patch of jaxbench_modal.sh per Karthik's scoring pre-flight (#5 comment 6813236025): new-digest guard, fresh-run guard, exit codes
of generate / CPU step 1 / ingest checked, turn-0 row count = 64, post-run turns-per-task report. Keeps a .bak copy."""
import pathlib, shutil
p = pathlib.Path(__file__).resolve().parent / "jaxbench_modal.sh"
shutil.copy(p, p.with_suffix(".sh.bak_0412"))
s = p.read_text()
reps = [
    ('set -u\n',
     'set -u\n# Nov 12 pre-flight (Karthik\'s audit 6813236025): digest must be a full, new digest; fresh run dirs unless RESUME=1; '
     'every step\'s exit code checked; row counts per turn.\n'),
    (': "${EXPECT_DIGEST:?set EXPECT_DIGEST to the v4 digest in the Modal volume}"\n',
     ': "${EXPECT_DIGEST:?set EXPECT_DIGEST to the new model\'s digest, from the create script\'s output}"\n'
     'echo "$EXPECT_DIGEST" | grep -Eq \'^[0-9a-f]{12,}$\' || { echo "EXPECT_DIGEST must be >= 12 hex chars"; exit 2; }\n'
     'for old in ee6656371218 0f764d3f6798 641ddde330ed 2a99a2f97822; do case "$EXPECT_DIGEST" in $old*) '
     'echo "EXPECT_DIGEST $EXPECT_DIGEST is an OLD model (base/v4/v5/mac)"; exit 2;; esac; done\n'),
    ('cd "$ROOT"\nfor turn in 0 1 2; do\n',
     'cd "$ROOT"\n'
     'if [ "${RESUME:-0}" != 1 ]; then  # fresh run: no leftovers that bench.py would resume or the merge would pick up\n'
     '  left=$(ls -d "$RUN" control/runs/cpu_checks/${PREFIX}-t* "$O"/repo/oracle/jobs/control-${PREFIX}-t* 2>/dev/null | head -3)\n'
     '  [ -n "$left" ] && { log "NOT FRESH: $left (set RESUME=1 to continue a run deliberately)"; exit 4; }\n'
     'fi\n'
     'for turn in 0 1 2; do\n'),
    ('  $PY control/bench.py --backend modal --model "$MODEL" generate --arm bare --workers "$WORKERS" 2>&1 | grep -vE MallocStack | tail -3\n',
     '  $PY control/bench.py --backend modal --model "$MODEL" generate --arm bare --workers "$WORKERS" > "$SC/scoring_kit/gen_${PREFIX}_t$turn.log" 2>&1\n'
     '  rc=$?; grep -vE MallocStack "$SC/scoring_kit/gen_${PREFIX}_t$turn.log" | tail -3; [ $rc -eq 0 ] || { log "GENERATE FAILED (rc $rc); stopping"; exit 5; }\n'
     '  nt=$($PY -c "import json; print(sum(1 for l in open(\'$RUN/bare.jsonl\') if l.strip() and json.loads(l).get(\'turn\') == $turn))")\n'
     '  log "turn $turn: $nt rows generated"; [ "$turn" != 0 ] || [ "$nt" -eq 64 ] || { log "TURN 0 HAS $nt ROWS, NOT 64; stopping"; exit 5; }\n'),
    ('  $PY control/cpu_check.py --backend modal --model "$MODEL" --tag "$tag" --py "$PY" --ingest 2>&1 | grep -vE MallocStack | tail -4\n',
     '  $PY control/cpu_check.py --backend modal --model "$MODEL" --tag "$tag" --py "$PY" --ingest > "$SC/scoring_kit/cpu_${tag}.log" 2>&1\n'
     '  rc=$?; grep -vE MallocStack "$SC/scoring_kit/cpu_${tag}.log" | tail -4; [ $rc -eq 0 ] || { log "CPU STEP 1 FAILED (rc $rc); stopping"; exit 6; }\n'),
    ('    $PY control/bench.py --backend modal --model "$MODEL" ingest "$merged" 2>&1 | grep -vE MallocStack | tail -1\n',
     '    $PY control/bench.py --backend modal --model "$MODEL" ingest "$merged" > "$SC/scoring_kit/ingest_${job}.log" 2>&1\n'
     '    rc=$?; grep -vE MallocStack "$SC/scoring_kit/ingest_${job}.log" | tail -1; [ $rc -eq 0 ] || { log "INGEST FAILED (rc $rc); stopping"; exit 7; }\n'),
]
for o, n in reps:
    assert s.count(o) == 1, o[:70]
    s = s.replace(o, n)
s += ('# post-run: tasks with fewer than 3 turns must have reached a terminal verdict (a Modal error can leave one short silently; v5 needed a catch-up)\n'
      '$PY -c "import json, collections; rows=[json.loads(l) for l in open(\'$RUN/bare.jsonl\') if l.strip()]; t=collections.defaultdict(set)\n'
      '[t[r[\'task\']].add(r[\'turn\']) for r in rows]; print(len(t), \'tasks; turns per task:\', dict(collections.Counter(len(v) for v in t.values())), '
      '\'(tasks with < 3 turns must have a terminal verdict; check bench summary)\')"\n')
p.write_text(s)
print("patched", p)
