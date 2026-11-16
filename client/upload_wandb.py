"""Phase 2.7: upload finished runs to Weights & Biases (run this where W&B is reachable).

Each run directory produced by run_experiment.py is self-contained (experiment.json,
trace_meta.json, run.json, summary.json, requests.jsonl, timeseries.jsonl, timeline.png),
so this script can run on any machine. Uploaded runs get a .wandb_uploaded marker and are
skipped next time (use --force to upload again).

Usage:
  python client/upload_wandb.py results/raw                      # everything not yet uploaded
  python client/upload_wandb.py results/raw/<experiment> --entity <team-name>
"""
import argparse
import json
import os

import wandb


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def find_runs(root):
    for dirpath, _, files in os.walk(root):
        if "summary.json" in files:
            yield dirpath


def upload(run_dir, project, entity, window):
    exp = load(os.path.join(run_dir, "experiment.json"), {})
    summary = load(os.path.join(run_dir, "summary.json"), {})
    run_info = load(os.path.join(run_dir, "run.json"), {})
    trace_meta = load(os.path.join(run_dir, "trace_meta.json"), {})
    cfg = dict(exp.get("config", {}))
    cfg["trace_args"] = trace_meta.get("args", {})
    cfg["trace_stats"] = {k: v for k, v in trace_meta.items() if k not in ("args", "trace")}
    cfg.update({k: exp.get(k) for k in ("git_commit", "gpu", "versions", "repeat")})
    name = os.path.basename(os.path.normpath(run_dir))
    tags = [str(t) for t in (cfg.get("label"), cfg.get("policy"), cfg.get("model")) if t]

    run = wandb.init(project=project, entity=entity, name=name, group=exp.get("experiment"),
                     job_type="replay", config=cfg, tags=tags, reinit=True)
    # server state over time, x axis = seconds since trace start
    run.define_metric("server/t_s", hidden=True)  # x axis only; no panel of its own
    run.define_metric("server/*", step_metric="server/t_s")
    for row in load_jsonl(os.path.join(run_dir, "timeseries.jsonl")):
        run.log({f"server/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
    # request hit rate per time window
    reqs = [r for r in load_jsonl(os.path.join(run_dir, "requests.jsonl")) if r.get("ok")]
    run.define_metric("window/t_s", hidden=True)
    run.define_metric("window/*", step_metric="window/t_s")
    if reqs and all(r.get("cached_tokens") is not None for r in reqs):
        end, t = max(r["start_s"] for r in reqs), 0.0
        while t <= end:
            win = [r for r in reqs if t <= r["start_s"] < t + window]
            if win:
                run.log({"window/t_s": t + window / 2,
                         "window/request_hit_rate": sum(r["cached_tokens"] > 0 for r in win) / len(win),
                         "window/ttft_ms_mean": sum(r["ttft_ms"] for r in win) / len(win)})
            t += window
    cols = ["req_id", "phase", "doc_id", "qid", "start_s", "ttft_ms", "e2e_ms", "cached_tokens", "prompt_tokens"]
    run.log({"requests": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in reqs])})
    if os.path.exists(os.path.join(run_dir, "timeline.png")):
        run.log({"timeline": wandb.Image(os.path.join(run_dir, "timeline.png"))})
    run.summary.update(flatten(summary))
    run.summary.update({f"client/{k}": v for k, v in run_info.items()
                        if isinstance(v, (int, float)) and k.startswith(("send_lag", "wall"))})
    url = run.url
    run.finish()
    with open(os.path.join(run_dir, ".wandb_uploaded"), "w") as f:
        f.write((url or "offline") + "\n")
    return url


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run directories or folders that contain them")
    p.add_argument("--project", default="hpml-prefix-cache")
    p.add_argument("--entity", default=None, help="W&B team name; default: your personal space")
    p.add_argument("--window", type=float, default=15.0, help="seconds per hit-rate window")
    p.add_argument("--force", action="store_true", help="upload again even if already uploaded")
    a = p.parse_args()

    run_dirs = sorted({d for path in a.paths for d in find_runs(path)})
    todo = [d for d in run_dirs if a.force or not os.path.exists(os.path.join(d, ".wandb_uploaded"))]
    print(f"found {len(run_dirs)} runs, {len(todo)} to upload")
    for d in todo:
        print(" ->", d, flush=True)
        print("    ", upload(d, a.project, a.entity, a.window))


if __name__ == "__main__":
    main()
