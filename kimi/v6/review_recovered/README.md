# A6 contamination review: recovered evidence files

This directory recovers the three files from the Nov 11 A6 adversarial review of the 273 class-B problems (proposed re-admissions). The files were lost when the `/tmp` scratchpad was wiped.

Local only: nothing here has been committed, pushed or posted.

| File | Status |
|---|---|
| `final_safe_list.txt` | **Byte-exact.** The full contents were printed in the transcript, and the writer's format is known. |
| `adversarial_verdicts.jsonl` | **Regenerated, not copied.** It was rebuilt by re-running the recovered scripts on a reconstructed input. No original copy survives to diff against. It matches the original byte size (83,625) and every excerpt the transcript printed (details below). |
| `pool_contamination.json` | **Regenerated, exact by construction.** The original writer was re-run verbatim on inputs that come from the script's own literals and a complete printout in the main session. It matches the original byte size (12,448). |
| `exclusions_classB_reconstructed.jsonl` | The reconstructed **input** (`review/exclusions.jsonl` was also lost). It holds the class-B rows only and omits the `rule` field. |

## Sources (read-only)

The transcripts are in `~/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/subagents/workflows/wf_49512077-aae/`. "Line" means the line number in the `.jsonl` file (one message per line).

- **ADV** = `agent-a155bcc295ec0cc51.jsonl`, the contamination-adversary, i.e. the A6 review. sha256 `56536647b33bc057a694cf1f6ff9a1937fefe105b6d80b212ce47867cbf94d1a`.
- **AUD** = `agent-a42ffff3745e37008.jsonl`, the filter-audit agent. It built `exclusions.jsonl`. sha256 `51245c915f1d4d4ecea20195a46769add22a042a801a035c109fef6b83437637`.

- **MAIN** = the parent session `~/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62.jsonl`. Only lines 20876–20877 and 21653–21654 are used: after the workflow, they read `review/adv/pool_contamination.json`. The file is still growing, so it is identified by line number and tool_use id instead of a hash.

These on-disk files were also used, read-only:
- `gemma-dataset-claude/dataset/problems/problems.jsonl` (mtime Nov 9) and `gemma-dataset-gen/dataset/problems/problems.jsonl` (mtime Nov 8). `build_table.py` read these exact paths, and both mtimes predate the Nov 11 03:00 build.
- `gemma-hackathon-research/benchmarks/jaxbench/level1/*k_*/baseline.py`: the name regex only.
- `gemma-work/tpu/ops1/repo/dataset/kernelbook/prod_selection.jsonl` and `prod_excluded_by_algorithm.jsonl`: used for corroboration only.

No git commands were run. Nothing was executed except the recovered review scripts (plain JSON/Python) and the tools in `recovery_tools/`, all with `python3 -I`, one process at a time. No dataset reference code was run.

## Layout

- `scripts/`: recovered **verbatim** from tool_use inputs. `checks/provenance.json` maps every file to its transcript, line and tool_use id.
  - `verdicts.py`: the ADV L286 heredoc, as first written (identical to `adv_L286_heredoc1.py`).
  - `adv_L314_heredoc1.py`: the L314 patch. `verdicts_after_L314_patch.py` is the result of applying it. That file is regenerated, but its 13,676 bytes equal the L357 `ls`.
  - `adv_L<n>.sh`: every Bash command the adversary ran from L286 to L364 (21 steps), in full. Their heredoc bodies and `python3 -c` snippets are extracted as `adv_L<n>_heredoc<k>.py` and `adv_L<n>_python_c<k>.py`.
  - `labels.py`: the AUD L361 (`cat >`), L365 (`cat >>`) and L368 (`cat >>`) heredocs, concatenated. `build_table.py` is AUD L372. `filter_audit_L*.sh` are the full commands.
- `regen/`: the new run directory, which started empty.
  - `replay_L<n>.sh` is the original ADV command with two changes: the `cd` target moved to `regen/review/adv`, and `python3` changed to `python3 -I`.
  - `replay_L348_heredoc2.sh` runs only L348's second heredoc. The first heredoc needs the lost `refs.py` and only prints.
  - `regen/review/exclusions.jsonl` and `regen/review/adv/pool_relu_stacks.json` are the reconstructed inputs.
- `recovery_tools/`: my own scripts. They are **not** part of the original review.
  - `extract_scripts.py`: transcript to `scripts/` and `regen/replay_*.sh`.
  - `reconstruct_exclusions_B.py`: rebuilds the class-B input.
  - `reconstruct_pool_relu_stacks.py`: rebuilds `pool_relu_stacks.json`.
  - `check_outputs.py` and `check_v6_lists.py`: the checks.
- `checks/`: replay stdout (`replay_L<n>.out`), check logs and `provenance.json`.

To reproduce, run these in order:

```
cd recovery_tools
python3 -I extract_scripts.py
python3 -I reconstruct_exclusions_B.py
for n in 286 290 314 352 356 360 364; do /bin/zsh ../regen/replay_L$n.sh > ../checks/replay_L$n.out 2>&1; done
python3 -I reconstruct_pool_relu_stacks.py
/bin/zsh ../regen/replay_L348_heredoc2.sh > ../checks/replay_L348_heredoc2.out 2>&1
python3 -I check_outputs.py
python3 -I check_v6_lists.py
```

The replay must start from an empty `regen/review/adv/`.

## final_safe_list.txt: byte-exact

- **Written by:** ADV L352 (`scripts/adv_L352_heredoc1.py`). It writes the sorted `problem_id`s of the safe verdicts, joined with `\n`, plus a trailing `\n`.
- **Printed in full at ADV L360 → L361:** `json.dumps(open('final_safe_list.txt').read().split())`, 6,754 characters, under the command's `head -c 9000` cap, so not truncated. The ids contain no whitespace, so the printed list plus the writer's format fixes every byte.
- **Here:** regenerated by replaying L352 on the regenerated `adversarial_verdicts.jsonl`. The result is identical to the L361 list. It is also identical to the StructuredOutput's `details.final_safe_list` (ADV L368, 191 ids, same order).
- **Checks:** 191 lines; 6,181 bytes, equal to the ADV L357 `ls`. sha256 `b673026e8bc71a314c3059d38ea330ad3b2211538f3d2d842dd99bf52ad9525b`.

## adversarial_verdicts.jsonl: regenerated

- **Written by:** `verdicts.py` (ADV L286, run once). ADV L314 then patched `verdicts.py` and re-ran it:
  - `AttModel_10357`, `AttModel_13790`, `SetConv_1791` and `LinearEncoder_2787` moved to unsure.
  - `Attention_8837` moved to instance.
  - `DQN_hot1_17174` gained a SAFE note.

  Nothing touched the file between L314 and the L352/L356/L360/L364 reads.
- **Here:** `regen/replay_L286.sh`, then `replay_L314.sh`, run on `regen/review/exclusions.jsonl`. The output is sorted by (verdict, closest, problem_id), so it does not depend on the input's row order.
- **Evidence it equals the original:**
  - Replayed stdout is identical, character for character, to the transcript's tool_result at L287 (pre-patch counts), L291 (all 196 pre-patch safe rows: task, pool flag, strength, id), L315 (post-patch counters, source histogram, ordered pool-removal list), L353, L361 and L365 (every keep-excluded reason and every non-`distinct:` safe reason, verbatim). It also matches the first 8 lines of L357.
  - Size 83,625 bytes, equal to the L357 `ls`. sha256 `f6e444dbef4a55be3b41c44c5b3abf9fefbd5a083f299228abbbbfccd174b822`.
- **Per-field coverage:**
  - `problem_id`, `verdict`, `in_kimi_pool_now` and the `closest` prefix are printed for all 273 rows (L291 + L353).
  - `closest` and `audit_strength` in full are covered for all 273 rows via `B_list.txt` (below).
  - `reason`: the 82 keep and 22 noted safe reasons are printed verbatim at L365. For the other 169 rows, `reason` is `'distinct: ' + justification[:140]`, and `justification[:150]` matches the printout for all 273.
  - `source` is printed per row for 52 rows. Three independent histograms match (ADV L57, AUD L381, ADV L315).
- **Remaining assumption:** the 88 problems in `labels.py`'s section "algo review kbk + prod (88)" are taken to be exactly the 88 excludes in the lost `algo_review_kbk.jsonl`. Supporting evidence: AUD L262 reports "algo kbk excl 88", and AUD L377 gives the section's class mix (A 17 / B 71). This only affects the `source` field (`kernelbook (kbk)` versus `prod (github/authored-from-paper)`). The histograms and the byte size would expose a mismatch.

## pool_contamination.json: regenerated, exact by construction

- **Written by:** ADV L348, second heredoc (`scripts/adv_L348_heredoc2.py`), with `json.dump(out, ..., indent=1)`. It has three keys:
  1. `verified_instance_in_pool_not_in_audit`: a literal in the script, 29 ids under 8 tasks.
  2. `unsure_in_pool_not_in_audit`: a literal in the script, 7 ids under 3 groups.
  3. `relu_stacks_meeting_audit_A_standard`: `json.load(open('pool_relu_stacks.json'))`.
- **The third key's input:**
  - `pool_relu_stacks.json` was written at ADV L310 (`scripts/adv_L310_heredoc1.py`) from `pool_scan4.jsonl`. That file held jaxpr traces of the pool references, which can only be regenerated by executing dataset code. It was **not** re-run.
  - Instead, MAIN L20876 (tool_use `toolu_01XVTVggeV2Gh1g3uz7cTWoi`) loaded `review/adv/pool_contamination.json` and printed `list(v.items())[:2]` for each key. The relu-stacks value has exactly two keys, so MAIN L20877 contains it **in full**: 91 + 68 `[problem_id, flop_share, audit_class]` entries in Python repr, ending in `]])]`.
  - `recovery_tools/reconstruct_pool_relu_stacks.py` parses that output and writes it with the same call L310 used (`json.dump(res, ..., indent=1)`).
  - Nothing wrote either file between ADV L348 and MAIN L20876.
- **Here:** `regen/replay_L348_heredoc2.sh` runs the L348 writer verbatim (`python3 -I`) in `regen/review/adv`.
- **Checks:**
  - The relu stacks match ADV L311: 91 entries, 2 in the audit table, 52 with share ≥ 0.95; and 68 entries, 3 in the audit table, 49 with share ≥ 0.95.
  - `pool_relu_stacks.json` is 9,557 bytes, equal to the L357 `ls`.
  - The replay prints `29 7`, equal to L349.
  - `pool_contamination.json` is 12,448 bytes, equal to the L357 `ls`.
  - `json.dumps(value)[:400]` for all three keys equals the MAIN L21654 printout.
  - Keys 1–2's first two items equal the MAIN L20877 printout.
  - sha256 `1277f39c418b3c65611ee26825b2838e4b6397b30cd17596d6c97bd927cddb85`.
- **Why "exact by construction":** every value is either a verbatim script literal or comes from a complete repr printout (strings and floats round-trip exactly), and `json.dump` is deterministic. The original bytes do not survive, so this rests on construction plus the size match, not on a diff.
- **Also included:** `regen/review/adv/pool_relu_stacks.json`, the reconstructed intermediate. sha256 `a3048aab566747dc219aff6c94ba3ff2382045d4d31687605fb44f92fa0b1ee4`.

## exclusions.jsonl (lost input) → exclusions_classB_reconstructed.jsonl

- **Original:** AUD L372 `build_table.py`, labelled by `labels.py`. It had 418 rows (A 93, B 273, C 52; ADV L57) with fields `problem_id, source, rule, closest_jaxbench_task, class, b_strength, family, in_kimi_pool_now, justification`. AUD and ADV show no other write to it or to `labels.py`.
- **Reconstructed:** the 273 class-B rows, with every field except `rule`. Rows are sorted by `problem_id`; the original order is not recoverable, and `verdicts.py`'s output does not depend on it. sha256 `5cf3c27ccb0592b15958c4bce066eead98f738d34047901a05edea77f63803b9`.
  - `class`, `b_strength`, `closest_jaxbench_task`, `family` and `justification` come from `labels.py`, verbatim.
  - `in_kimi_pool_now` comes from `B_list.txt`. ADV L72 wrote it, one line per class-B row (task group, strength, P/- pool flag, id, `justification[:150]`). ADV L73 printed lines 1–120 and ADV L76 printed lines 120–306, so all 306 lines are recovered.
  - `source` comes from replaying `build_table.py`'s first-add order, steps 1–8. Each lost input is replaced by its printout:
    - kbkc and kbkd algo-review excludes: AUD L250.
    - karthik/kbk score-filter rows with batch: AUD L266.
    - The complete ordered score-filter list (86 karthik + 36 aarav kbkc): AUD L278, L282 and L286.
    - Steps 5 and 6 read the on-disk `problems.jsonl` files.
    - The prod batch was confirmed with `prod_selection.jsonl`.
- **Checks** (`checks/reconstruct_exclusions_B.log`, 25 PASS, 0 FAIL):
  - 273 class-B labels; the B set, tasks, strengths and `justification[:150]` all equal `B_list.txt`.
  - Section sizes 88/25/28 with the L377 class mix; the L250 lists equal the `labels.py` sections.
  - Per-row source and pool flag equal ADV L283 for all 52 gemini/kb-l2-port/kbkd rows.
  - The source histogram equals ADV L57.
  - The not-in-pool source histogram (186) and the in-pool family histogram (87) equal AUD L381.
- **Not recovered:**
  - The 145 class-A and class-C rows: their `in_kimi_pool_now` flags and `rule` strings were never printed per row.
  - The `rule` field of every row: it embeds verdict strings from the lost `algo_review_*.jsonl` and truncated `exclude_reason` texts.

## Count checks (all on the regenerated files)

| Check | Expected (source) | Result |
|---|---|---|
| total / safe / instance / unsure | 273 / 191 / 43 / 39 (StructuredOutput L368) | PASS |
| class-B already in Kimi pool | 87; 45 out = 29 copies + 16 unsure; 42 stay (L368) | PASS |
| safe and not yet in pool | 149 (L368) | PASS |
| v4 family drop | 42 of its 52 class-B stay out (L368) | PASS |
| attention-family safe | 71 (L368; L357 group counts) | PASS (L357 replay identical) |
| `final_safe_list.txt` = safe rows of verdicts = L361 printout = L368 `final_safe_list` | (identity) | PASS |
| file sizes vs L357 `ls` | 83,625 / 6,181 / 12,448 (`pool_contamination.json`) / 9,557 (`pool_relu_stacks.json`) / 13,676 (`verdicts.py`) | PASS |
| `pool_contamination.json` literal sections | 29 and 7 ids (L349 `29 7`) | PASS |
| every keep-excluded/unsure in `v6/exclude_hard.txt ∪ v6/flag_unsure.txt` | 82 of 82 | **FAIL: 45 of 82** (see below) |

**About the v6 list check** (`checks/check_v6_lists.log`):
- **The 45 that are listed:**
  - All 29 in-pool instance copies are in `exclude_hard.txt`.
  - All 16 in-pool unsure problems are in `flag_unsure.txt`.
- **The 37 that are listed in neither file:**
  - They are exactly the 14 instance and 23 unsure problems that were never in the Kimi pool.
  - None of the 82 keep/unsure problems appears in `v6/v6_train.jsonl` or `v6/v6_train_strict.jsonl` (`problem_id` field).
  - The current pool file `kimi/data/full_problems.jsonl` contains exactly the 45 listed ones.
  - So the gap is in the lists' coverage, not in the training data. The lists are not a complete record of the A6 keep set.
- **Further information:**
  - 11 problems that A6 calls safe are still in `exclude_hard.txt`/`flag_unsure.txt`, so v6 is stricter for these: `moe_expert_ffn_batched_2`, `kb/l2/70`, `Hidden2DiscreteDeal_3790`, `GCN_14619`, `NeuralNetPartialNoGradModel_3264`, `kb/l2/51`, `kb/l2/68`, `DenseSAGEConv_2033`, `LinearEmbedder_13109`, `TemporalDecayRegression_3216` and `DQN_hot1_17174`.
  - All 81 ids in `v6/a6_dependent_problem_ids.txt` are in `final_safe_list.txt`.

## Not recovered / not re-run

- **`exclusions.jsonl`:** the class-A and class-C rows, every row's `rule` field, and the original row order (see above).
- **ADV steps extracted but not executed**, because they need git (`git show` of v3 data: L318, L321, L325) or lost inputs:
  - L294–L310 and L329–L344 need `refs.py`, `pool_scan*.jsonl`, `tasks_scan.jsonl` or `kmatch.txt`.
  - L348's first heredoc needs `refs.py`.

  Apart from L310, which was replaced by the printout reconstruction above, none of these steps writes any of the three files.
