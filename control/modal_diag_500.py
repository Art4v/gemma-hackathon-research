"""Why does v4's 62p_Causal_Conv1d_GDN turn-0 request get HTTP 500 from Ollama on Modal (twice, deterministic)? Same image, GPU, env
and volume as control/modal_gemma.py; the exact request bench.py sends. Captures: the error body, the streamed output up to the
failure, and the Ollama server log. Separate app, so the deployed gemma-bench app is not touched while the run is live.
  MODAL_PROFILE=newacct modal run modal_diag_500.py"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
import modal

OLLAMA_VERSION = "0.40.14"
MODELS_DIR = "/ollama-models"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "ca-certificates", "zstd")
         .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh"))
volume = modal.Volume.from_name("gemma-bench-ollama")
app = modal.App("gemma-bench-diag", image=image)
ENV = {"OLLAMA_MODELS": MODELS_DIR, "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_HOST": "127.0.0.1:11434",
       "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1"}


@app.function(gpu="A10G", volumes={MODELS_DIR: volume}, timeout=1800)
def diag(body: dict) -> dict:
    log = open("/tmp/ollama.log", "w")
    subprocess.Popen(["ollama", "serve"], env={**os.environ, **ENV}, stdout=log, stderr=log)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2); break
        except OSError:
            time.sleep(0.5)
    out = {}
    # 1. streamed: what the model produced before the failure
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    content, thinking, last, errs = [], [], None, []
    try:
        with urllib.request.urlopen(req, timeout=1500) as r:
            for line in r:
                c = json.loads(line)
                if "error" in c:
                    errs.append(c["error"])
                m = c.get("message", {})
                content.append(m.get("content", "")); thinking.append(m.get("thinking", "") or "")
                if m.get("tool_calls"):
                    errs.append(f"tool_calls: {json.dumps(m['tool_calls'])[:500]}")
                last = {k: v for k, v in c.items() if k != "message"}
    except urllib.error.HTTPError as e:
        errs.append(f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"{type(e).__name__}: {e}")
    text = "".join(content)
    out["stream"] = {"errors": errs, "chars": len(text), "tail": text[-1500:], "thinking_chars": len("".join(thinking)), "last": last}
    # 2. non-streamed (what bench.py sends): the error body
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1500))
        out["nonstream"] = {"ok": True, "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason")}
    except urllib.error.HTTPError as e:
        out["nonstream"] = {"ok": False, "code": e.code, "body": e.read().decode(errors="replace")[:2000]}
    log.flush()
    out["server_log_tail"] = open("/tmp/ollama.log").read()[-4000:]
    return out


@app.local_entrypoint()
def main(task: str = "62p_Causal_Conv1d_GDN", model: str = "gemma4-e4b-lora:v4final-e2"):
    sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
    import bench
    ref, _ = bench.tasks()[task]
    body = bench.request_body(bench.conversation(ref, [], bench.docs_for("bare")), bench.seed_for(task, 0), model)
    print("request: model", body["model"], "options", body["options"], "messages", len(body["messages"]),
          "chars", sum(len(m["content"]) for m in body["messages"]))
    out = diag.remote(body)
    print(json.dumps(out, indent=1))
