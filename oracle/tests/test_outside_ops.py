"""Jaxpr-level outside-kernel check (reviewer item 4): the AST heavy-op list can be bypassed by aliases, getattr, `.sum()`, lax.reduce_sum,
or a product stored in a variable. At the jaxpr level every contraction is `dot_general`, every reduction is `reduce_*`, and the only
primitives allowed outside `pallas_call` are layout and dtype ones. The walk is tested on a fake jaxpr; the JAX integration test runs only
where JAX is installed."""
import pathlib, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from outside_ops import ALLOWED, ORACLE_ALLOWED, classify, classify_avals, walk, walk_avals


class P:
    def __init__(self, name): self.name = name


class Eqn:
    def __init__(self, name, params=None): self.primitive, self.params = P(name), params or {}


class Jaxpr:
    def __init__(self, eqns): self.eqns = eqns


class Closed:
    def __init__(self, jaxpr): self.jaxpr = jaxpr


class Walk(unittest.TestCase):
    def test_counts_primitives_outside_pallas_call_and_never_descends_into_it(self):
        j = Jaxpr([Eqn("reshape"), Eqn("pallas_call", {"jaxpr": Jaxpr([Eqn("dot_general"), Eqn("mul")])}), Eqn("convert_element_type")])
        self.assertEqual(walk(j), ({"reshape": 1, "convert_element_type": 1}, 1))

    def test_descends_into_pjit_and_other_sub_jaxprs(self):
        inner = Jaxpr([Eqn("dot_general"), Eqn("pallas_call", {"jaxpr": Jaxpr([Eqn("mul")])})])
        j = Jaxpr([Eqn("pjit", {"jaxpr": Closed(inner)}), Eqn("custom_jvp_call", {"call_jaxpr": Closed(Jaxpr([Eqn("logistic")]))})])
        self.assertEqual(walk(j), ({"dot_general": 1, "logistic": 1}, 1))

    def test_no_pallas_call_is_reported_as_zero(self):
        self.assertEqual(walk(Jaxpr([Eqn("dot_general")])), ({"dot_general": 1}, 0))


class Shrink(unittest.TestCase):
    """The trace must see the same program the checker ran: check.py slices every array input to at most 256 per dimension."""

    def test_dims_are_capped_at_256_and_scalars_are_left_alone(self):
        from outside_ops import shrink_shape
        self.assertEqual(shrink_shape((8192, 8192)), (256, 256))
        self.assertEqual(shrink_shape((16, 300, 32)), (16, 256, 32))
        self.assertEqual(shrink_shape(()), ())


class Classify(unittest.TestCase):
    def test_layout_and_dtype_ops_outside_are_fine(self):
        ok, bad = classify({"reshape": 2, "pad": 1, "convert_element_type": 1, "broadcast_in_dim": 1, "transpose": 1, "slice": 1})
        self.assertEqual((ok, bad), (True, []))

    def test_any_arithmetic_or_contraction_outside_is_flagged_and_named(self):
        ok, bad = classify({"reshape": 1, "dot_general": 1, "mul": 2, "reduce_sum": 1})
        self.assertEqual((ok, bad), (False, ["dot_general", "mul", "reduce_sum"]))

    def test_allowed_set_has_no_compute(self):
        for name in ("dot_general", "mul", "add", "reduce_sum", "reduce_max", "exp", "logistic", "select_n", "conv_general_dilated"):
            self.assertNotIn(name, ALLOWED)


class Integration(unittest.TestCase):
    """The real thing: a cheat kernel that does the matmul outside and copies it through a tiled kernel, and an honest kernel. Needs JAX."""
    REF = ("import jax, jax.numpy as jnp\n"
           "def create_inputs(dtype):\n    k = jax.random.PRNGKey(0)\n    return jax.random.normal(k, (64, 128), dtype), jax.random.normal(k, (128, 128), dtype)\n"
           "def workload(x, w):\n    return x @ w\n")
    HDR = "import jax, jax.numpy as jnp\nfrom jax import lax\nfrom jax.experimental import pallas as pl\n"
    COPY = ("def _copy(y):\n    def k(y_ref, o_ref):\n        o_ref[...] = y_ref[...]\n"
            "    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype), grid=(y.shape[0] // 8,),\n"
            "        in_specs=[pl.BlockSpec((8, y.shape[1]), lambda i: (i, 0))], out_specs=pl.BlockSpec((8, y.shape[1]), lambda i: (i, 0)))(y)\n")
    CHEATS = {
        "alias": "d = lax.dot_general\ndef workload(x, w):\n    return _copy(d(x, w, (((1,), (0,)), ((), ()))))\n",
        "getattr": "def workload(x, w):\n    return _copy(getattr(jnp, 'mat' + 'mul')(x, w))\n",
        "product_then_sum": "def workload(x, w):\n    p = x[:, :, None] * w[None]\n    return _copy(p.sum(axis=1))\n",
        "reduce_sum": "def workload(x, w):\n    return _copy(lax.reduce_sum(x[:, :, None] * w[None], axes=(1,)))\n",
        "add_reduce": "def workload(x, w):\n    return _copy(jnp.add.reduce(x[:, :, None] * w[None], axis=1))\n",
    }
    HONEST = ("def workload(x, w):\n    def k(x_ref, w_ref, o_ref):\n        o_ref[...] = jnp.dot(x_ref[...], w_ref[...])\n"
              "    xp = x.astype(jnp.float32).reshape(x.shape)\n"
              "    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((x.shape[0], w.shape[1]), jnp.float32), grid=(x.shape[0] // 8,),\n"
              "        in_specs=[pl.BlockSpec((8, x.shape[1]), lambda i: (i, 0)), pl.BlockSpec(w.shape, lambda i: (0, 0))],\n"
              "        out_specs=pl.BlockSpec((8, w.shape[1]), lambda i: (i, 0)))(xp, w)\n")

    def run_script(self, cand):
        import json, os, subprocess, tempfile
        py = os.environ.get("SERA_JAX_PY", "")
        if not py or not pathlib.Path(py).exists():
            self.skipTest("JAX interpreter not available here")
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "sandbox"))
        from sandbox_args import wrap
        script = pathlib.Path(__file__).parent.parent / "outside_ops.py"
        with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
            p = pathlib.Path(d); (p / "ref.py").write_text(self.REF); (p / "cand.py").write_text(self.HDR + self.COPY + cand); (p / "outside_ops.py").write_text(script.read_text())
            out = subprocess.run(wrap([str(py), "-I", "outside_ops.py", "ref.py", "cand.py"]), cwd=d, capture_output=True, text=True, timeout=240)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, out.stderr[-400:])
        return json.loads(line[-1][7:])

    def test_every_cheat_is_caught_and_the_honest_kernel_passes(self):
        for name, cheat in self.CHEATS.items():
            r = self.run_script(cheat)
            self.assertFalse(r["outside_ok"], (name, r))
            self.assertTrue(set(r["outside_disallowed"]) & {"dot_general", "mul", "reduce_sum"}, (name, r))
        r = self.run_script(self.HONEST)
        self.assertEqual((r["outside_ok"], r["outside_disallowed"], r["pallas_calls"]), (True, [], 1), r)

    def test_batch_mode_scores_many_kernels_in_one_process_and_survives_a_broken_one(self):
        """--batch pack.json out.jsonl: one JSON line per item, in order, with the item's id; a candidate that fails at import is a row, not a crash."""
        import json, os, subprocess, tempfile
        py = os.environ.get("SERA_JAX_PY", "")
        if not py or not pathlib.Path(py).exists():
            self.skipTest("JAX interpreter not available here")
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "sandbox"))
        from sandbox_args import wrap
        script = pathlib.Path(__file__).parent.parent / "outside_ops.py"
        pack = [{"id": "cheat", "ref": self.REF, "cand": self.HDR + self.COPY + self.CHEATS["alias"]},
                {"id": "broken", "ref": self.REF, "cand": "import nonexistent_module_xyz\n"},
                {"id": "honest", "ref": self.REF, "cand": self.HDR + self.COPY + self.HONEST}]
        with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
            p = pathlib.Path(d); (p / "pack.json").write_text(json.dumps(pack)); (p / "outside_ops.py").write_text(script.read_text())
            out = subprocess.run(wrap([str(py), "-I", "outside_ops.py", "--batch", "pack.json", "out.jsonl"]), cwd=d, capture_output=True, text=True, timeout=300)
            self.assertEqual(out.returncode, 0, out.stderr[-400:])
            rows = [json.loads(l) for l in (p / "out.jsonl").read_text().splitlines()]
        self.assertEqual([r["id"] for r in rows], ["cheat", "broken", "honest"])
        self.assertEqual([r["outside_ok"] for r in rows], [False, False, True])
        self.assertEqual(rows[1]["error"], "ModuleNotFoundError")


class Aval:
    def __init__(self, dtype, shape=()): self.dtype, self.shape = dtype, shape


class AVar:
    def __init__(self, dtype, shape=()): self.aval = Aval(dtype, shape)


class ALit(AVar):  # a Literal: has .val; never tainted, its shape is ignored
    def __init__(self, dtype, val=1): super().__init__(dtype, (10 ** 6,)); self.val = val


class AEqn(Eqn):
    def __init__(self, name, invars, outvars, params=None):
        super().__init__(name, params); self.invars, self.outvars = invars, outvars


class TJaxpr:
    def __init__(self, invars, eqns, outvars=()): self.invars, self.eqns, self.outvars = invars, eqns, list(outvars)


def i32(*shape): return AVar("int32", shape)
def f32(*shape): return AVar("float32", shape)
def b1(*shape): return AVar("bool", shape)


class Metadata(unittest.TestCase):
    """R2 (the oracle's rule): metadata = small values computed only from non-float inputs. A primitive outside ORACLE_ALLOWED
    (gather and scatter* included) is allowed only if none of its operands is tainted (derived from a floating workload input) and
    every operand and result has at most 2**16 elements. Dtype does not matter for untainted values: float bin edges computed from
    int group sizes are metadata (Megablox's make_group_metadata, via jnp.histogram)."""

    def check(self, invars, *eqns):
        outside, heavy, n = walk_avals(TJaxpr(invars, list(eqns) + [AEqn("pallas_call", [], [])]))
        self.assertEqual(n, 1)
        return classify_avals(outside, heavy)

    def test_small_int_metadata_from_int_inputs_is_allowed(self):
        g, ends, one, m, sel, x, flat = i32(8), i32(8), i32(8), b1(8), i32(8), f32(64, 128), f32(8192)
        r = self.check([g, x], AEqn("cumsum", [g], [ends]), AEqn("add", [ends, ALit("int32")], [one]),
                       AEqn("lt", [ends, one], [m]), AEqn("select_n", [m, ends, one], [sel]), AEqn("reshape", [x], [flat]))
        self.assertEqual(r, (True, [], ["add", "cumsum", "lt", "select_n"]))

    def test_untainted_float_values_are_metadata(self):  # histogram's float32 bin edges from int tile ids
        g, gf, edges, m = i32(8), f32(8), f32(129), b1(129)
        r = self.check([g], AEqn("convert_element_type", [g], [gf]), AEqn("iota", [], [edges]),
                       AEqn("mul", [edges, ALit("float32")], [edges]), AEqn("lt", [edges, ALit("float32")], [m]))
        self.assertEqual(r, (True, [], ["lt", "mul"]))

    def test_float_work_on_a_float_input_is_disallowed_however_small(self):
        x, b, y = f32(64), f32(64), f32(64)
        self.assertEqual(self.check([x, b], AEqn("add", [x, b], [y])), (False, ["add"], []))
        s, t = f32(), f32()
        self.assertEqual(self.check([s], AEqn("mul", [s, ALit("float32")], [t])), (False, ["mul"], []))  # even a scalar

    def test_an_int_derived_from_a_float_input_is_tainted(self):
        x, i, j = f32(64), i32(), i32()
        r = self.check([x], AEqn("argmax", [x], [i]), AEqn("add", [i, ALit("int32")], [j]))
        self.assertEqual(r, (False, ["add", "argmax"], []))

    def test_taint_flows_through_allowed_layout_ops(self):
        x, xi, y = f32(64), i32(64), i32(64)
        r = self.check([x], AEqn("convert_element_type", [x], [xi]), AEqn("cumsum", [xi], [y]))
        self.assertEqual(r, (False, ["cumsum"], []))

    def test_untainted_arrays_past_2_16_elements_are_disallowed(self):
        a, b, c, d = i32(2 ** 16), i32(2 ** 16), i32(2 ** 16 + 1), i32(2 ** 16 + 1)
        self.assertEqual(self.check([a], AEqn("cumsum", [a], [b])), (True, [], ["cumsum"]))
        self.assertEqual(self.check([c], AEqn("cumsum", [c], [d])), (False, ["cumsum"], []))
        e = f32(2 ** 16 + 1)
        self.assertEqual(self.check([], AEqn("iota", [], [e]), AEqn("exp", [e], [f32(2 ** 16 + 1)])), (False, ["exp"], []))

    def test_literals_are_never_tainted_and_their_shape_is_ignored(self):
        g, y = i32(8), i32(8)
        self.assertEqual(self.check([g], AEqn("mul", [g, ALit("float32")], [y])), (True, [], ["mul"]))

    def test_one_heavy_occurrence_disallows_the_primitive(self):
        g, x, a, b = i32(8), f32(8), i32(8), f32(8)
        self.assertEqual(self.check([g, x], AEqn("add", [g, g], [a]), AEqn("add", [x, x], [b])), (False, ["add"], []))

    def test_stack_and_memory_space_constraints_are_layout(self):
        # Google's 54p wraps its operands in with_memory_space_constraint, 53p stacks: placement and layout, no arithmetic
        x, y = f32(4096, 128), f32(2, 4096, 128)
        self.assertEqual(self.check([x], AEqn("with_memory_space_constraint", [x], [x]), AEqn("stack", [x, x], [y])), (True, [], []))
        self.assertTrue({"stack", "with_memory_space_constraint"}.isdisjoint(ALLOWED))  # the frozen study's list is unchanged

    def test_a_dynamic_slice_at_data_derived_offsets_is_a_gather(self):
        # Lebron James 6711257577: a Python loop of lax.dynamic_slice(kv_pages, (page_indices[i], 0, 0)) is the paged-gather shortcut
        # with every op allowed; a start index that is not a literal comes from the inputs
        pool, table, start, page = f32(4096, 16, 128), i32(64), i32(), f32(1, 16, 128)
        r = self.check([pool, table], AEqn("dynamic_slice", [table, ALit("int32")], [start]),
                       AEqn("dynamic_slice", [pool, start, ALit("int32"), ALit("int32")], [page]))
        self.assertFalse(r[0])
        self.assertIn("dynamic_slice(dynamic start)", r[1])

    def test_a_dynamic_update_slice_at_data_derived_offsets_is_a_scatter(self):
        out, upd, idx, i, res = f32(4096, 128), f32(1, 128), i32(8), i32(), f32(4096, 128)
        r = self.check([out, upd, idx], AEqn("dynamic_slice", [idx, ALit("int32")], [i]),
                       AEqn("dynamic_update_slice", [out, upd, i, ALit("int32")], [res]))
        self.assertEqual(r[:2], (False, ["dynamic_update_slice(dynamic start)"]))

    def test_a_dynamic_slice_with_literal_starts_is_still_layout(self):
        x, y = f32(4096, 128), f32(128, 128)
        self.assertEqual(self.check([x], AEqn("dynamic_slice", [x, ALit("int32"), ALit("int32")], [y])), (True, [], []))

    def test_a_dynamic_slice_of_small_int_metadata_is_metadata(self):
        g, i, s = i32(8), i32(), i32(1)
        r = self.check([g], AEqn("dynamic_slice", [g, ALit("int32")], [i]), AEqn("dynamic_slice", [g, i], [s]))
        self.assertEqual(r, (True, [], ["dynamic_slice(dynamic start)"]))

    def test_gather_and_scatter_follow_the_same_rule(self):
        self.assertNotIn("gather", ORACLE_ALLOWED)
        self.assertIn("gather", ALLOWED)  # the frozen study's rule is unchanged
        pages, idx, kv = f32(16, 8, 128), i32(4, 1), f32(4, 8, 128)
        self.assertEqual(self.check([pages, idx], AEqn("gather", [pages, idx], [kv])), (False, ["gather"], []))
        buf, upd, out = f32(64), f32(4), f32(64)
        self.assertEqual(self.check([buf, idx, upd], AEqn("scatter-add", [buf, idx, upd], [out])), (False, ["scatter-add"], []))
        g, s, gi = i32(64), i32(64), i32(4)
        self.assertEqual(self.check([g, idx], AEqn("scatter", [g, idx, gi], [s]), AEqn("gather", [s, idx], [i32(4)])),
                         (True, [], ["gather", "scatter"]))

    def test_an_input_or_operand_without_an_aval_is_not_metadata(self):
        u, y = object(), i32(8)
        self.assertEqual(self.check([u], AEqn("add", [u], [y])), (False, ["add"], []))

    def test_pallas_call_outputs_are_tainted_only_by_tainted_operands(self):
        x, g, px, pg, a, b = f32(64), i32(8), i32(8), i32(8), i32(8), i32(8)
        outside, heavy, n = walk_avals(TJaxpr([x, g], [AEqn("pallas_call", [x], [px]), AEqn("pallas_call", [g], [pg]),
                                                       AEqn("add", [px, px], [a]), AEqn("sub", [pg, pg], [b])]))
        self.assertEqual((n, classify_avals(outside, heavy)), (2, (False, ["add"], ["sub"])))

    def test_taint_is_mapped_through_a_jit_sub_jaxpr(self):
        ix, ig, iy, io = f32(8, 8), i32(8), f32(8, 8), i32(8)
        inner = TJaxpr([ix, ig], [AEqn("cumsum", [ig], [io]), AEqn("dot_general", [ix, ix], [iy])], [iy, io])
        x, g, y, o, z = f32(8, 8), i32(8), f32(8, 8), i32(8), i32(8)
        j = TJaxpr([x, g], [AEqn("pjit", [x, g], [y, o], {"jaxpr": Closed(inner)}), AEqn("add", [o, o], [z]), AEqn("pallas_call", [], [])])
        outside, heavy, n = walk_avals(j)
        self.assertEqual((outside, n), walk(j))
        self.assertEqual(heavy, {"dot_general": 1})  # cumsum and the outer add see only the int input

    def test_loops_are_conservative_any_tainted_operand_taints_everything(self):
        ic, ix, iy = i32(), f32(8), i32()
        body = TJaxpr([ic, ix], [AEqn("add", [ic, ALit("int32")], [iy])], [iy, ix])
        c, x, c2, x2, z = i32(), f32(8), i32(), f32(8), i32()
        j = TJaxpr([c, x], [AEqn("scan", [c, x], [c2, x2], {"jaxpr": Closed(body)}), AEqn("neg", [c2], [z]), AEqn("pallas_call", [], [])])
        outside, heavy, _ = walk_avals(j)
        self.assertEqual(classify_avals(outside, heavy), (False, ["add", "neg"], []))
        body2 = TJaxpr([ic], [AEqn("add", [ic, ALit("int32")], [iy])], [iy])
        j2 = TJaxpr([c], [AEqn("scan", [c], [c2], {"jaxpr": Closed(body2)}), AEqn("neg", [c2], [z]), AEqn("pallas_call", [], [])])
        self.assertEqual(classify_avals(*walk_avals(j2)[:2]), (True, [], ["add", "neg"]))


if __name__ == "__main__":
    unittest.main()
