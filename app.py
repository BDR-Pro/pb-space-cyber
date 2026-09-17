"""Chat UI for dealignai/Qwen3.6-35B-A3B-MXFP4-CRACK-MTP, served on a Petabyte GPU.

This is an MXFP4-quantised Qwen3.5 MoE (Qwen3_5MoeForConditionalGeneration) — plain transformers
cannot load MXFP4 MoE cleanly, so it is served by vLLM, which supports the architecture in-tree.
vLLM runs as a subprocess exposing its OpenAI-compatible API on loopback; a thin Gradio front
streams from it so the space URL is a usable chat page, not a raw API.

Passwordless by design (the operator asked for an open link): no auth, no shell.
"""
import os
import subprocess
import time
import urllib.request

# The Petabyte gateway terminates TLS and forwards X-Forwarded-Proto, but uvicorn (under Gradio)
# only believes it from forwarded_allow_ips, default 127.0.0.1, while the request arrives from the
# docker bridge. Without this Gradio emits http:// asset URLs and the page renders blank over https.
os.environ.setdefault("FORWARDED_ALLOW_IPS", "*")
os.environ.setdefault("HF_HOME", "/cache/hf")

import gradio as gr
from openai import OpenAI

MODEL = os.environ.get("MODEL_ID", "dealignai/Qwen3.6-35B-A3B-MXFP4-CRACK-MTP")
SERVED = "cyber"
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
PORT = int(os.environ.get("PORT", "7860"))
MAX_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
GPU_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")


def start_vllm():
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--served-model-name", SERVED,
        "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_LEN),
        "--gpu-memory-utilization", GPU_UTIL,
        "--trust-remote-code",
    ]
    print("launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd)


_proc = start_vllm()
_client = OpenAI(base_url=f"http://127.0.0.1:{VLLM_PORT}/v1", api_key="local")


def _wait_ready(timeout=2400):
    """Block until vLLM answers /health. If vLLM dies first, die too, so the watchdog reports the
    real failure instead of leaving a UI that can never reach a backend."""
    start = time.time()
    url = f"http://127.0.0.1:{VLLM_PORT}/health"
    while time.time() - start < timeout:
        if _proc.poll() is not None:
            raise SystemExit(f"vLLM exited with code {_proc.returncode} before becoming ready")
        try:
            urllib.request.urlopen(url, timeout=3)
            print("vLLM is ready", flush=True)
            return
        except Exception:  # noqa: BLE001 — not up yet
            time.sleep(4)
    raise SystemExit("vLLM did not become ready in time")


_wait_ready()


def chat(message, history, temperature, max_tokens):
    # Gradio 6 hands history as OpenAI-style [{role, content}, ...] already.
    msgs = list(history or []) + [{"role": "user", "content": message}]
    stream = _client.chat.completions.create(
        model=SERVED, messages=msgs, stream=True,
        temperature=float(temperature), max_tokens=int(max_tokens), top_p=0.9,
    )
    out = ""
    for chunk in stream:
        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        if delta:
            out += delta
            yield out


gr.ChatInterface(
    chat,
    title="Qwen3.6-35B-A3B Cyber (uncensored)",
    description=("36B MoE (~3B active), MXFP4, served on a Petabyte H100 via vLLM · "
                 f"{MODEL}"),
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(64, 8192, value=2048, step=64, label="Max new tokens"),
    ],
).launch(server_name="0.0.0.0", server_port=PORT)
