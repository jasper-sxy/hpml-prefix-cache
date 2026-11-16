"""Phase 2.7: one-click experiment runner (runs on the GPU server).

For every run in a YAML config (x repeats) it:
  1. stops any running vLLM server, then starts a fresh one (empty prefix cache)
  2. waits until /health answers
  3. replays the trace (client/replay.py), computes metrics, draws the timeline
  4. writes experiment.json (full config, server command, versions, git commit) and
     copies the trace's .meta.json next to the results, so the run directory is
     self-contained and can be uploaded to W&B from any machine
  5. stops the server
A failed run is recorded and the runner moves on to the next one.

Usage (inside screen; it takes a while):
  python client/run_experiment.py configs/phase2_three_scenarios.yaml
  python client/run_experiment.py configs/phase2_three_scenarios.yaml --dry-run   # print the plan only
"""
import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

DEFAULTS = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "num_gpu_blocks": 800,
    "prefix_caching": True,
    "policy": "lru",  # exported as KV_EVICT_POLICY; vLLM ignores it until Phase 6
    "max_model_len": 4096,
    "max_num_seqs": 64,
    "block_size": 16,
    "seed": 0,
    "port": 8000,
    "repeats": 1,
    "sample_interval": 1.0,
    "slo_ttft_ms": 500.0,
    "slo_tpot_ms": 50.0,
    "extra_server_args": [],
    "env": {},
    "startup_timeout_s": 900,
}


def server_cmd(c):
    if c.get("server_cmd"):  # advanced/testing: a full custom command
        return list(c["server_cmd"])
    vllm = os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = [vllm, "serve", c["model"],
           "--dtype", "bfloat16",
           "--max-model-len", str(c["max_model_len"]),
           "--block-size", str(c["block_size"]),
           "--max-num-seqs", str(c["max_num_seqs"]),
           "--seed", str(c["seed"]),
           "--port", str(c["port"]),
           "--enable-prompt-tokens-details"]
    if c.get("num_gpu_blocks"):
        cmd += ["--num-gpu-blocks-override", str(c["num_gpu_blocks"])]
    cmd += ["--enable-prefix-caching"] if c["prefix_caching"] else ["--no-enable-prefix-caching"]
    return cmd + [str(x) for x in c["extra_server_args"]]


def server_env(c):
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)  # the server only talks to localhost
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # system nvcc is too old for FlashInfer's JIT
    env["KV_EVICT_POLICY"] = str(c["policy"])
    env.update({k: str(v) for k, v in c["env"].items()})
    return env


def http_ok(url):
    try:
        handler = urllib.request.ProxyHandler({})  # never go through a proxy for localhost
        with urllib.request.build_opener(handler).open(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def stop_all_servers():
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(3)


def wait_healthy(proc, port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return f"server exited with code {proc.returncode} during startup (see server.log)"
        if http_ok(f"http://localhost:{port}/health"):
            return None
        time.sleep(2)
    return f"server not healthy after {timeout}s (see server.log)"


def stop_server(proc):
    """Stop the server and every process it started (it runs in its own process group)."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        time.sleep(2)  # main process is gone; remove any child it left behind
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def vllm_fork_commit():
    """Commit of third_party/vllm (our fork); '-dirty' if it has uncommitted changes."""
    repo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "vllm")
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) or None
    except Exception:
        return None


def model_revision(model):
    """Hugging Face commit of the cached model (refs/main in the HF cache)."""
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ref = os.path.join(root, "hub", "models--" + model.replace("/", "--"), "refs", "main")
    try:
        with open(ref) as f:
            return f.read().strip()
    except OSError:
        return None


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return out.strip().splitlines()[0]
    except Exception:
        return None


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def expand(cfg):
    base = copy.deepcopy(DEFAULTS)
    base.update(cfg.get("defaults", {}))
    plans = []
    for i, override in enumerate(cfg["runs"]):
        c = copy.deepcopy(base)
        c.update(override)
        c.setdefault("label", f"run{i}")
        for rep in range(int(c["repeats"])):
            plans.append((c, rep))
    return plans


def run_one(exp_name, exp_dir, c, rep):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(exp_dir, f"{c['label']}_rep{rep}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    meta_src = c["trace"].replace(".jsonl", ".meta.json")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(run_dir, "trace_meta.json"))

    cmd = server_cmd(c)
    info = {
        "experiment": exp_name, "label": c["label"], "repeat": rep, "run_dir": run_dir,
        "config": c, "server_cmd": cmd, "git_commit": git_commit(), "gpu": gpu_name(),
        "versions": {"vllm": version("vllm"), "torch": version("torch"),
                     "vllm_fork_commit": vllm_fork_commit(), "model_revision": model_revision(c["model"])},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running",
    }

    def save(status, error=None):
        info.update({"status": status, "error": error, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(os.path.join(run_dir, "experiment.json"), "w") as f:
            json.dump(info, f, indent=2)

    save("running")
    stop_all_servers()
    print(f"\n=== {c['label']} rep{rep} -> {run_dir}\n    {' '.join(cmd)}", flush=True)
    log = open(os.path.join(run_dir, "server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=server_env(c),
                            start_new_session=True)  # own process group, see stop_server()
    try:
        err = wait_healthy(proc, c["port"], c["startup_timeout_s"])
        if err:
            save("failed", err)
            print("   FAILED:", err, flush=True)
            return False
        print("    server ready, replaying ...", flush=True)
        py = sys.executable
        here = os.path.dirname(os.path.abspath(__file__))
        steps = [
            [py, os.path.join(here, "replay.py"), "--trace", c["trace"], "--model", c["model"],
             "--url", f"http://localhost:{c['port']}", "--out-dir", run_dir,
             "--sample-interval", str(c["sample_interval"])],
            [py, os.path.join(here, "metrics.py"), run_dir,
             "--slo-ttft-ms", str(c["slo_ttft_ms"]), "--slo-tpot-ms", str(c["slo_tpot_ms"])],
            [py, os.path.join(here, "plot_run.py"), run_dir, "--slo-ttft-ms", str(c["slo_ttft_ms"])],
        ]
        for step in steps:
            out = subprocess.run(step, capture_output=True, text=True)
            with open(os.path.join(run_dir, "runner.log"), "a") as f:
                f.write(f"$ {' '.join(step)}\n{out.stdout}\n{out.stderr}\n")
            if out.returncode != 0:
                save("failed", f"step failed: {os.path.basename(step[1])} (see runner.log)")
                print(f"   FAILED at {os.path.basename(step[1])}; see runner.log", flush=True)
                return False
        with open(os.path.join(run_dir, "summary.json")) as f:
            s = json.load(f)
        print(f"    done: goodput {s['goodput_req_s']} req/s, hit {s.get('request_hit_rate')}, "
              f"TTFT p50 {s['ttft_ms']['p50']} ms", flush=True)
        save("done")
        return True
    finally:
        stop_server(proc)
        log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    exp_name = cfg.get("name") or os.path.splitext(os.path.basename(a.config))[0]
    plans = expand(cfg)
    print(f"experiment {exp_name}: {len(plans)} runs")
    for c, rep in plans:
        assert os.path.exists(c["trace"]), f"trace not found: {c['trace']}"
        print(f"  - {c['label']} rep{rep}: blocks={c['num_gpu_blocks']} caching={c['prefix_caching']} "
              f"policy={c['policy']} trace={os.path.basename(c['trace'])}")
    if a.dry_run:
        return
    exp_dir = os.path.join("results/raw", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(a.config, os.path.join(exp_dir, "config.yaml"))
    ok = sum(run_one(exp_name, exp_dir, c, rep) for c, rep in plans)
    print(f"\nfinished: {ok}/{len(plans)} runs succeeded -> {exp_dir}")


if __name__ == "__main__":
    main()
