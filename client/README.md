# client/  (Phase 2)
Replay client: sends requests at trace timestamps (asyncio + streaming), logs send time, first-token time, per-token times.
Metrics: TTFT / inter-token latency p50/p95/p99, tokens/s, goodput under frozen SLOs, prefix-cache hit rate from /metrics.
One-click runner: start server, warm up, replay, collect, stop, upload to W&B.
Status: TODO
