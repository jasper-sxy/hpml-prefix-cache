#!/bin/bash
# Phase 3.4 C: Nsight Systems capture of a short window (a few hundred engine steps) while a
# trace is being replayed. vLLM's "cuda" profiler calls cudaProfilerStart/Stop when we POST
# /start_profile, and nsys records only that range (--capture-range=cudaProfilerApi).
#
# Usage (from the repo root, nothing else running on the GPU):
#   bash profiling/nsys_capture.sh <trace.jsonl> <name> [blocks=1600] [steps=500] [delay_s=60]
# Output: profiling/nsys/<name>.nsys-rep (+ .server.log, .replay.log)
set -u
TRACE=$1; NAME=$2; BLOCKS=${3:-1600}; STEPS=${4:-500}; DELAY=${5:-60}
NSYS=${NSYS:-/root/autodl-tmp/cuda-12.4/bin/nsys}
OUT=profiling/nsys
mkdir -p "$OUT"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export VLLM_USE_FLASHINFER_SAMPLER=0
pkill -f "vllm serve" && sleep 5

echo "[1/5] starting vLLM under nsys (blocks=$BLOCKS, profile $STEPS steps)"
"$NSYS" profile -o "$OUT/$NAME" --force-overwrite=true \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true \
  vllm serve Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 --max-model-len 4096 \
    --block-size 16 --max-num-seqs 64 --seed 0 --port 8000 --enable-prompt-tokens-details \
    --num-gpu-blocks-override "$BLOCKS" --enable-prefix-caching \
    --profiler-config "{\"profiler\": \"cuda\", \"max_iterations\": $STEPS}" \
  > "$OUT/$NAME.server.log" 2>&1 &
NSYS_PID=$!
for i in $(seq 1 180); do
  curl -sf localhost:8000/health > /dev/null && break
  kill -0 $NSYS_PID 2>/dev/null || { echo "server died, see $OUT/$NAME.server.log"; exit 1; }
  sleep 5
done

echo "[2/5] replaying $TRACE"
python client/replay.py --trace "$TRACE" --out-dir "$OUT/${NAME}_replay" > "$OUT/$NAME.replay.log" 2>&1 &
REPLAY_PID=$!

echo "[3/5] waiting ${DELAY}s for a steady state, then profiling $STEPS steps"
sleep "$DELAY"
curl -sf -X POST localhost:8000/start_profile && echo "    profiler started"
sleep 30   # several hundred steps at ~50 steps/s; vLLM stops by itself after $STEPS
curl -sf -X POST localhost:8000/stop_profile > /dev/null 2>&1

echo "[4/5] stopping replay and server; nsys writes the report (can take a minute)"
kill $REPLAY_PID 2>/dev/null
pkill -INT -f "vllm serve"
wait $NSYS_PID
ls -lh "$OUT/$NAME".nsys-rep 2>/dev/null || { echo "no report written; see $OUT/$NAME.server.log"; exit 1; }

echo "[5/5] kernel summary"
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$OUT/$NAME" "$OUT/$NAME.nsys-rep" > /dev/null 2>&1
"$NSYS" stats --report cuda_gpu_kern_sum "$OUT/$NAME.nsys-rep" 2>/dev/null | head -25
