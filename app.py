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
    msgs = list(history or []) + [{"role": "user", "content": message}]
    out = ""
    for ch in llm.create_chat_completion(
        messages=msgs, stream=True,
        temperature=float(temperature), max_tokens=int(max_tokens), top_p=0.9,
    ):
        delta = ch["choices"][0]["delta"].get("content", "")
        if delta:
            out += delta
            yield out


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
