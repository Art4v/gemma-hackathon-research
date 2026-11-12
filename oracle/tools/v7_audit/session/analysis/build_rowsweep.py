"""Policy sweep, not training data: variants of the whole-row-output row-reduction template for the f32, N = 8192 items that overflowed VMEM.

  python3 -I build_rowsweep.py ITEMS.jsonl OUT.jsonl

Variants of each chosen item (same reference, so one XLA timing per item):
  outacc      accumulate in the output block, no f32 scratch (saves one (tm, N) block)
  outacc_c256 outacc, and the epilogue runs on 256-row chunks so its temporaries are a quarter of the block
  acc_c256    scratch kept, epilogue on 256-row chunks
  outacc_c256_tk512  outacc_c256 with a 512 K tile
  tm512       the original kernel with tm 512, tk 512
"""
import ast, json, re, sys

PICK = ("00020", "00072", "00018", "00015")  # non-exam chains: softmax->swish, soft_sign->center, normalize, log_softmax
OLD_TILES = "tm, tk = _tile(M, 1024), _tile(K, 512 if N <= 4096 else 256)"


def variants(code: str) -> dict:
    head, rest = code.split("\n\ndef _kernel(", 1)
    kern, work = rest.split("\n\ndef workload(", 1)
    epi = kern.split("        y = acc[...]\n", 1)[1].split("        o_ref[...] = y.astype(o_ref.dtype)\n", 1)[0]
    dot = re.search(r"    acc\[\.\.\.\] \+= (.*)\n", kern).group(1)
    sig = kern.split(")", 1)[0]  # "a_ref, b_ref, o_ref, acc"
    first = "    @pl.when(pl.program_id(1) == 0)\n    def _():\n        {t}[...] = jnp.zeros_like({t})\n    {t}[...] += " + dot + "\n\n"
    last = "    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)\n    def _():\n"
    whole = lambda t: last + f"        y = {t}[...]\n" + epi + "        o_ref[...] = y.astype(o_ref.dtype)\n"
    chunk = lambda t: (last + "        for r in range(0, o_ref.shape[0], 256):\n" + f"            y = {t}[pl.ds(r, 256), :]\n"
                       + "".join("    " + l + "\n" for l in epi.splitlines()) + "            o_ref[pl.ds(r, 256), :] = y.astype(o_ref.dtype)\n")
    noscratch = lambda w: re.sub(r"        scratch_shapes=\[.*\],\n", "", w)
    k = lambda s, body: head + "\n\ndef _kernel(" + s + "):\n" + body
    out_sig = sig.replace(", acc", "")
    return {
        "outacc": k(out_sig, first.format(t="o_ref") + whole("o_ref")) + "\n\ndef workload(" + noscratch(work),
        "outacc_c256": k(out_sig, first.format(t="o_ref") + chunk("o_ref")) + "\n\ndef workload(" + noscratch(work),
        "acc_c256": k(sig, first.format(t="acc") + chunk("acc")) + "\n\ndef workload(" + work,
        "outacc_c256_tk512": k(out_sig, first.format(t="o_ref") + chunk("o_ref")) + "\n\ndef workload(" + noscratch(work).replace(OLD_TILES, "tm, tk = _tile(M, 1024), _tile(K, 512)"),
        "tm512": code.replace(OLD_TILES, "tm, tk = _tile(M, 512), _tile(K, 512)"),
    }


def main(src: str, dst: str) -> None:
    out = []
    for line in open(src):
        it = json.loads(line)
        if it["family"] != "row_reduction_matmul" or it["id"][-5:] not in PICK:
            continue
        assert OLD_TILES in it["code"], it["id"]
        for name, code in variants(it["code"]).items():
            ast.parse(code)
            assert code != it["code"]
            out.append({"id": f"rowsweep/{it['id'][-5:]}__{name}", "task": it["task"], "ref_code": it["ref_code"], "code": code})
    open(dst, "w").write("".join(json.dumps(o) + "\n" for o in out))
    print(len(out), "sweep items from", len({o["task"] for o in out}), "references")


if __name__ == "__main__":
    main(*sys.argv[1:3])
