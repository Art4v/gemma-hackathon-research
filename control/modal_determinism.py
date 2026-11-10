"""Is greedy generation reproducible on Modal, and does streaming change the tokens? One fresh A10G container, Ollama restarted (empty
cache, model reloaded) before each request: A non-streamed, B streamed, C non-streamed; each compared with the stored run row.
Separate app; the deployed gemma-bench app is not touched.   MODAL_PROFILE=newacct modal run modal_determinism.py"""
import json, os, signal, subprocess, sys, time, urllib.request
import modal

OLLAMA_VERSION = "0.40.14"
MODELS_DIR = "/ollama-models"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "ca-certificates", "zstd")
         .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh"))
volume = modal.Volume.from_name("gemma-bench-ollama")
app = modal.App("gemma-bench-diag", image=image)
ENV = {"OLLAMA_MODELS": MODELS_DIR, "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_HOST": "127.0.0.1:11434",
       "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1"}
URL = "http://127.0.0.1:11434"


def _serve():
    p = subprocess.Popen(["ollama", "serve"], env={**os.environ, **ENV}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{URL}/api/version", timeout=2); return p
        except OSError:
            time.sleep(0.5)


def _stop(p):
    os.killpg(p.pid, signal.SIGKILL); p.wait(); time.sleep(3)


def _req(body, stream):
    req = urllib.request.Request(f"{URL}/api/chat", json.dumps({**body, "stream": stream}).encode(), {"Content-Type": "application/json"})
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=1500))
        return r["message"]["content"], r.get("eval_count"), r.get("done_reason")
    out, n, reason = [], 0, None
    with urllib.request.urlopen(req, timeout=1500) as resp:
        for line in resp:
            c = json.loads(line)
            if "error" in c:
                return "".join(out), n, "error: " + c["error"]
            out.append((c.get("message") or {}).get("content") or ""); n += 1
            if c.get("done"):
                reason = c.get("done_reason"); n = c.get("eval_count", n)
    return "".join(out), n, reason


@app.function(gpu="A10G", volumes={MODELS_DIR: volume}, timeout=1800)
def run(body: dict, stored: str) -> dict:
    res = {}
    for name, stream in (("A_nonstream", False), ("B_stream", True), ("C_nonstream", False)):
        p = _serve()
        text, n, reason = _req(body, stream)
        _stop(p)
        res[name] = {"tokens": n, "finish": reason, "chars": len(text), "same_as_stored": text == stored, "_text": text}
    a = res["A_nonstream"]["_text"]
    res["B_equals_A"] = res["B_stream"]["_text"] == a
    res["C_equals_A"] = res["C_nonstream"]["_text"] == a
    for v in res.values():
        if isinstance(v, dict): v.pop("_text", None)
    res["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return res


@app.local_entrypoint()
def main(task: str = "1p_Flash_Attention"):
    sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
    import bench
    model = "gemma4-e4b-lora:v4final-e2"
    stored = next(r for r in map(json.loads, open(f"{bench.RUNS}/gemma4-e4b-lora_v4final-e2__modal/bare.jsonl")) if r["task"] == task and r["turn"] == 0)
    ref, _ = bench.tasks()[task]
    body = bench.request_body(bench.conversation(ref, [], bench.docs_for("bare")), bench.seed_for(task, 0), model)
    print("stored:", stored["output_tokens"], "tokens,", len(stored["response"]), "chars, finish", stored["finish"])
    print(json.dumps(run.remote(body, stored["response"]), indent=1))
