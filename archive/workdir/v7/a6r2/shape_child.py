"""python -I shape_child.py IN.jsonl OUT.jsonl : per {key, ref}, the abstract input shapes/dtypes of create_inputs() (jax.eval_shape)."""
import json, signal, sys, types
import jax
def alarm(*_): raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
with open(sys.argv[2], "w") as fo:
    for line in open(sys.argv[1]):
        x = json.loads(line)
        try:
            signal.alarm(60); mod = types.ModuleType("ref"); exec(compile(x["ref"], "ref.py", "exec"), mod.__dict__)
            a = jax.eval_shape(mod.create_inputs); leaves = [l for l in jax.tree_util.tree_leaves(a) if hasattr(l, "shape")]
            fo.write(json.dumps({"key": x["key"], "shapes": [list(l.shape) for l in leaves], "dtypes": [str(l.dtype) for l in leaves]}) + "\n")
        except BaseException as e:
            fo.write(json.dumps({"key": x["key"], "error": f"{type(e).__name__}: {str(e)[:150]}"}) + "\n")
        finally:
            signal.alarm(0)
