"""Phase 2.4: replay client.

Replays a trace against a running vLLM server: each request is sent at its trace
timestamp (without waiting for earlier ones), streamed, and timed token by token.

Output (in --out-dir):
  requests.jsonl  one line per request: TTFT, end-to-end time, inter-token gaps, token counts
  run.json        run-level info: trace, model, wall time, prefix-cache hit rate, client lag
  timeseries.jsonl  server state sampled every --sample-interval seconds: running/waiting
                  requests, KV-cache usage, prefix-cache counters, preemptions, GPU util/memory

Usage:
  python client/replay.py --trace workload/traces/<model>/<name>.jsonl --limit 40   # quick test
  python client/replay.py --trace workload/traces/<model>/<name>.jsonl             # full trace
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp
import numpy as np


def read_metric(text, name):
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.split()[-1])
            found = True
    return total if found else float("nan")


async def prefix_cache_counters(session, url):
    async with session.get(url + "/metrics") as r:
        text = await r.text()
    return {"queries": read_metric(text, "vllm:prefix_cache_queries_total"),
            "hits": read_metric(text, "vllm:prefix_cache_hits_total")}


async def save_metrics(session, url, path):
    """Full /metrics page (Prometheus text), for profiling/server_breakdown.py."""
    try:
        async with session.get(url + "/metrics") as r:
            text = await r.text()
        with open(path, "w") as f:
            f.write(text)
    except Exception as e:  # profiling extra: never fail the run for it
        print("could not save", path, repr(e))


# vLLM metric names (summed over label sets). Names differ a little between versions,
# so every candidate is tried and missing ones are simply left out.
SAMPLED = [
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
]


async def gpu_stats():
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        util, mem = out.decode().splitlines()[0].split(",")
        return {"gpu_util_pct": float(util), "gpu_mem_mib": float(mem)}
    except Exception:
        return {}


async def sampler(session, url, t0, interval, rows, stop):
    while not stop.is_set():
        tick = time.perf_counter()
        row = {"t_s": round(tick - t0, 3)}
        try:
            async with session.get(url + "/metrics") as r:
                text = await r.text()
            for name in SAMPLED:
                v = read_metric(text, name)
                if v == v:  # skip NaN (metric not exported by this version)
                    row[name.split(":", 1)[1]] = v
        except Exception as e:
            row["error"] = repr(e)
        row.update(await gpu_stats())
        rows.append(row)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval - (time.perf_counter() - tick)))
        except asyncio.TimeoutError:
            pass


async def one_request(session, url, model, rec, t0, results):
    delay = t0 + rec["t"] - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)  # wait until this request's arrival time
    send = time.perf_counter()
    body = {"model": model, "prompt": rec["prompt_ids"], "max_tokens": rec["max_tokens"],
            "temperature": 0, "ignore_eos": True,  # fixed output length
            "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True}  # generated token IDs, for eval/compare_outputs.py
    token_times, usage, err, out_ids = [], None, None, []
    try:
        async with session.post(url + "/v1/completions", json=body) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:300]}"
            else:
                async for raw in r.content:  # server-sent events, one line at a time
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    msg = json.loads(data)
                    if msg.get("choices"):
                        token_times.append(time.perf_counter())
                        out_ids.extend(msg["choices"][0].get("token_ids") or [])
                    if msg.get("usage"):
                        usage = msg["usage"]
    except Exception as e:  # keep going; the failure is recorded
        err = repr(e)
    end = time.perf_counter()
    details = (usage or {}).get("prompt_tokens_details") or {}
    results.append({
        "req_id": rec["req_id"], "doc_id": rec["doc_id"], "qid": rec["qid"],
        "phase": rec["phase"], "prefix_len": rec["prefix_len"],
        "scheduled_s": rec["t"], "start_s": round(send - t0, 4), "end_s": round(end - t0, 4),
        "send_lag_ms": round((send - t0 - rec["t"]) * 1000, 3),
        "ok": err is None and bool(token_times), "error": err,
        "ttft_ms": round((token_times[0] - send) * 1000, 3) if token_times else None,
        "e2e_ms": round((end - send) * 1000, 3),
        "itl_ms": [round((b - a) * 1000, 3) for a, b in zip(token_times, token_times[1:])],
        "n_chunks": len(token_times),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),  # only if the server reports it
        "output_ids": out_ids,  # generated token IDs (empty if the server does not return them)
    })


async def progress(results, total, t0):
    while len(results) < total:
        await asyncio.sleep(10)
        print(f"  [{time.perf_counter() - t0:6.0f}s] finished {len(results)}/{total}", flush=True)


async def run(a):
    with open(a.trace) as f:
        recs = [json.loads(line) for line in f]
    if a.limit:
        recs = recs[: a.limit]
    meta_path = a.trace.replace(".jsonl", ".meta.json")
    model = a.model
    if model is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            model = json.load(f)["args"]["model"]
    assert model, "pass --model (no .meta.json found next to the trace)"

    name = os.path.basename(a.trace).replace(".jsonl", "")
    out_dir = a.out_dir or os.path.join("results/raw", time.strftime("%Y%m%d-%H%M%S") + "_" + name)
    os.makedirs(out_dir, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.timeout)
    connector = aiohttp.TCPConnector(limit=0)  # no client-side cap on concurrent requests
    # aiohttp ignores http_proxy unless trust_env=True, so localhost is reached directly
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as s:
        before = await prefix_cache_counters(s, a.url)
        await save_metrics(s, a.url, os.path.join(out_dir, "metrics_before.prom"))
        results = []
        print(f"replaying {len(recs)} requests over {recs[-1]['t']:.0f}s -> {out_dir}")
        t0 = time.perf_counter() + 1.0  # one second head start to schedule everything
        prog = asyncio.create_task(progress(results, len(recs), t0))
        series, stop = [], asyncio.Event()
        samp = asyncio.create_task(sampler(s, a.url, t0, a.sample_interval, series, stop))
        await asyncio.gather(*(one_request(s, a.url, model, r, t0, results) for r in recs))
        prog.cancel()
        wall = time.perf_counter() - t0
        stop.set()
        await samp
        after = await prefix_cache_counters(s, a.url)
        await save_metrics(s, a.url, os.path.join(out_dir, "metrics_after.prom"))

    with open(os.path.join(out_dir, "timeseries.jsonl"), "w") as f:
        for row in series:
            f.write(json.dumps(row) + "\n")

    results.sort(key=lambda r: r["req_id"])
    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if r["ok"]]
    ttft = np.array([r["ttft_ms"] for r in ok]) if ok else np.array([np.nan])
    lag = np.array([r["send_lag_ms"] for r in results])
    dq, dh = after["queries"] - before["queries"], after["hits"] - before["hits"]
    run_info = {
        "trace": a.trace, "model": model, "url": a.url, "n_requests": len(results),
        "n_ok": len(ok), "n_errors": len(results) - len(ok), "wall_s": round(wall, 2),
        "prefix_cache_queries": dq, "prefix_cache_hits": dh,
        "prefix_cache_hit_rate": round(dh / dq, 4) if dq else None,
        "ttft_ms_p50": round(float(np.percentile(ttft, 50)), 1),
        "ttft_ms_p99": round(float(np.percentile(ttft, 99)), 1),
        "send_lag_ms_p99": round(float(np.percentile(lag, 99)), 2),
        "sampled_metrics": sorted({k for row in series for k in row} - {"t_s", "error"}),
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print(json.dumps(run_info, indent=2))
    errors = [r["error"] for r in results if r["error"]]
    if errors:
        print("first error:", errors[0])
    if run_info["send_lag_ms_p99"] > 50:
        print("WARNING: the client fell behind the trace schedule (p99 send lag > 50 ms)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--model", default=None, help="default: read from the trace's .meta.json")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--limit", type=int, default=0, help="replay only the first N requests")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--timeout", type=float, default=600, help="seconds without data before giving up")
    p.add_argument("--sample-interval", type=float, default=1.0, help="seconds between server samples")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
