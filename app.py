"""Chat UI for a 27B uncensored cyber model, served on a Petabyte H100 via llama.cpp.

Why llama.cpp and not vLLM/transformers: the custom-quantised cyber checkpoints (MXFP4 / NVFP4 /
W4A16) all quantise layers stock vLLM's loaders don't expect (e.g. quantised embeddings), so they
need a bespoke vLLM build. GGUF + llama.cpp loads regardless — no architecture registration, no
quant negotiation — and an IQ4_XS 27B is ~16 GB, so it loads in a couple of minutes, not twenty.

Passwordless by design (open link): no auth, no shell.
"""
import os

os.environ.setdefault("FORWARDED_ALLOW_IPS", "*")   # gateway X-Forwarded-Proto -> no blank page over https
os.environ.setdefault("HF_HOME", "/cache/hf")

# The CUDA llama.cpp wheel links libcudart.so.12 / libcublas*, which the slim base image lacks.
# We install them from pip (nvidia-*-cu12) but the dynamic loader won't find them at dlopen time,
# so preload each with RTLD_GLOBAL (in dependency order) to satisfy libllama.so's NEEDED entries.
import ctypes
import glob
import site
import sys

_libdirs = []
for _base in set(site.getsitepackages() + [sys.prefix + "/lib"]):
    _libdirs += glob.glob(_base + "/nvidia/*/lib")
for _soname in ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"):
    for _d in _libdirs:
        _p = os.path.join(_d, _soname)
        if os.path.exists(_p):
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import gradio as gr

REPO = os.environ.get("MODEL_REPO", "cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF")
FILE = os.environ.get("MODEL_FILE", "Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf")
N_CTX = int(os.environ.get("N_CTX", "8192"))
PORT = int(os.environ.get("PORT", "7860"))

print(f"downloading {REPO}/{FILE} ...", flush=True)
model_path = hf_hub_download(REPO, FILE, cache_dir="/cache/hf")
print("loading into GPU ...", flush=True)
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=N_CTX, n_batch=512, verbose=True)
print("model ready", flush=True)


def chat(message, history, temperature, max_tokens):
    # History back to the model must be the clean answers only, or its own leaked reasoning feeds
    # back in and compounds. Gradio already holds the stripped text we yielded, so pass it through.
    msgs = list(history or []) + [{"role": "user", "content": message}]
    out = ""
    for ch in llm.create_chat_completion(
        messages=msgs, stream=True,
        temperature=float(temperature), max_tokens=int(max_tokens), top_p=0.9,
    ):
        delta = ch["choices"][0]["delta"].get("content", "")
        if not delta:
            continue
        out += delta
        # This model emits a reasoning preamble ending in </think>; show only the answer after it.
        if "</think>" in out:
            yield out.split("</think>", 1)[1].lstrip("\n")
        else:
            yield "_thinking…_"


gr.ChatInterface(
    chat,
    title="Qwen3.8-27B Uncensored Cyber (agentic)",
    description=("27B GGUF (IQ4_XS imatrix), served on a Petabyte H100 via llama.cpp · "
                 f"{REPO}"),
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(64, 8192, value=2048, step=64, label="Max new tokens"),
    ],
).launch(server_name="0.0.0.0", server_port=PORT)
