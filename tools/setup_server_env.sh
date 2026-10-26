#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
server_venv="${1:-.venv/server}"
server_python="${SERVER_PYTHON:-python3.12}"
if [[ ! -e "$server_venv/bin/python" ]]; then
    "$server_python" -m venv "$server_venv"
fi
"$server_venv/bin/python" -m pip install --upgrade 'pip==25.3'
"$server_venv/bin/python" -m pip install \
    'torch==2.8.0' 'torchvision==0.23.0' \
    --index-url https://download.pytorch.org/whl/cu128
if [[ -f requirements-server-lock.txt ]]; then
    "$server_venv/bin/python" -m pip install -r requirements-server-lock.txt -r requirements-data-index.txt
else
    "$server_venv/bin/python" -m pip install -r requirements-server.txt -r requirements-data-index.txt
fi
"$server_venv/bin/python" -m pip install --no-deps -e .
"$server_venv/bin/python" -m pip check
"$server_venv/bin/python" -m pip freeze > "$server_venv/installed-lock.txt"
"$server_venv/bin/python" -c 'import torch; from esm.models.esmc import ESMC; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.device_count())'
