#!/usr/bin/env bash
# Run tool_sandbox CLI directly from source without reinstalling the package.
# Usage:
#   ./run.sh [args...]
# Examples:
#   ./run.sh --agent GPT_4_o_2024_05_13 --user GPT_4_o_2024_05_13 -t
#   ./run.sh --agent ... --enable_tool_suggestion --tool_graph_path tool_graph.json

set -euo pipefail

# Resolve the directory this script lives in (the repo root containing tool_sandbox/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the local package importable without `pip install -e .`.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Pick the python interpreter:
#   1) explicit $PYTHON
#   2) active conda env ($CONDA_PREFIX/bin/python)
#   3) python / python3 on PATH
#   4) local .venv as last resort
if [[ -n "${PYTHON:-}" ]]; then
    PY="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif [[ -x "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
    PY="${SCRIPT_DIR}/../.venv/bin/python"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PY="${SCRIPT_DIR}/.venv/bin/python"
else
    echo "[run.sh] ERROR: no python interpreter found" >&2
    exit 1
fi

echo "[run.sh] Using python: ${PY}"
echo "[run.sh] PYTHONPATH=${PYTHONPATH}"
echo "[run.sh] CWD: ${SCRIPT_DIR}"

cd "${SCRIPT_DIR}"
# tool_sandbox.cli is a package without __main__.py; call main() directly.
exec "${PY}" -c "from tool_sandbox.cli import main; main()" "$@"
