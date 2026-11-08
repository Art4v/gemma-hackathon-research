"""outside_ops.py, the kernel check's rule, on fake jaxprs (no JAX). The same rule on real kernels, including the five ways of hiding a
matmul outside the kernel, is in test_tpu_oracle.py (CheatsAreNotKernels, OutsideMetadata)."""
import pathlib, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from outside_ops import ALLOWED, classify_avals, walk, walk_avals


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


class Allowed(unittest.TestCase):
    def test_allowed_set_has_no_compute_and_no_gather(self):
        for name in ("dot_general", "mul", "add", "reduce_sum", "reduce_max", "exp", "logistic", "select_n", "conv_general_dilated",
                     "gather", "scatter", "scatter-add"):
            self.assertNotIn(name, ALLOWED)


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
    """The metadata rule: small values computed only from non-float inputs. A primitive outside ALLOWED
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
        self.assertNotIn("gather", ALLOWED)
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
