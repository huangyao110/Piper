#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
    echo 'Usage: bash tools/run_server_training.sh CONFIG OUTPUT [GPU_IDS] [extra training arguments]' >&2
    exit 2
fi
server_config="$1"
server_output="$2"
server_devices="${3:-0,1,2,3}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
IFS=',' read -r -a server_gpu_ids <<< "$server_devices"
server_python="${SERVER_TRAIN_PYTHON:-.venv/server/bin/python}"
export CUDA_VISIBLE_DEVICES="$server_devices"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$server_python" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${#server_gpu_ids[@]}" -m piper.train \
    --config "$server_config" --output "$server_output" "$@"
