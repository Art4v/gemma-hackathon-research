#!/usr/bin/env bash
# Create dataset/.venv on Python 3.12 with the oracle's pinned CPU JAX stack (root requirements.txt), plus torch for the port check.
#   bash dataset/scripts/env.sh            # then: source dataset/.venv/bin/activate
# The system python3 (3.15) has no jaxlib wheel; jaxlib 0.10.2 ships macOS arm64 wheels for cp311-cp314.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON312:-/Users/aarav/.local/bin/python3.12}"
VENV="$ROOT/dataset/.venv"
[ -x "$PY" ] || { echo "need a Python 3.12 interpreter at $PY (set PYTHON312)" >&2; exit 1; }
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt"
"$VENV/bin/pip" install -q -r "$ROOT/dataset/requirements-port.txt"
"$VENV/bin/python" -c "import jax, numpy; print('jax', jax.__version__, 'numpy', numpy.__version__, 'devices', jax.devices())"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__)"
echo "export SERA_JAX_PY=$VENV/bin/python   # for oracle/tests"
