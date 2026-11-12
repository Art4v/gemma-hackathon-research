import json, pathlib, sys, types
import jax
idx = json.load(open("refs/index.json"))
for name, m in idx.items():
    mod = types.ModuleType("r"); exec(compile(pathlib.Path(f"refs/{name}.py").read_text(), name, "exec"), mod.__dict__)
    a = jax.eval_shape(mod.create_inputs); o = jax.eval_shape(mod.workload, *a)
    ins = [tuple(x.shape) for x in jax.tree_util.tree_leaves(a)]; outs = [tuple(x.shape) for x in jax.tree_util.tree_leaves(o)]
    ok = ins[0] == (m["M"], m["K"]) and ins[1] == (m["K"], m["N"]) and outs[0] == (m["M"], m["N"])
    print(f"{'OK ' if ok else 'BAD'} {name[:52]:52} in {ins} -> out {outs}")
