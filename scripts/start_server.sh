#!/bin/bash
# Usage: bash scripts/start_server.sh [MODEL] [NUM_KV_BLOCKS]
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm
unset http_proxy https_proxy
export VLLM_USE_FLASHINFER_SAMPLER=0   # system nvcc is 12.4; FlashInfer sampler JIT fails
MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}
BLOCKS=${2:-800}
pkill -f "vllm serve" && sleep 3       # stop any old server (port/GPU memory)
vllm serve $MODEL \
  --dtype bfloat16 --max-model-len 4096 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --num-gpu-blocks-override $BLOCKS \
  --block-size 16 --max-num-seqs 64 --seed 0 \
  --port 8000
