# curation

Tools for curating the dataset: checking and filtering kernels before they reach the TPU. This is later work. None of it decides
a verdict: the oracle (`oracle/`, see its DESCRIPTION.md) is the only judge of whether a solution compiles, is correct and is fast.

| file | role |
|---|---|
| `lower_filter.py` | a CPU pre-filter: lowers each kernel for TPU v6e on a Mac, in a sandbox, and rejects those that cannot lower. It can only reject, never pass, and abstains when the reference is the problem. `filter` writes repair feedback for rejected kernels. |
| `sandbox/sandbox_profile.py` | the macOS sandbox `lower_filter.py` runs model-written code in (no network, no exec, no reads under `$HOME`, no `.env`) |
| `int_inputs.py` | a JAXBench audit: which tasks have integer, bool or static inputs, or constant float inputs, that stay the same across the oracle's seeds. Imports the oracle; CPU only. |
| `tests/test_lower_filter.py` | the pre-filter's bookkeeping tests (no JAX needed) |

Not ready to run as is: `lower_filter.py validate` reads the chip's judged kernels from `oracle/round2/`, `oracle/results/` and
`oracle/queue/`, which are not in this repository yet.
