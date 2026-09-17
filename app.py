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

import tools

REPO = os.environ.get("MODEL_REPO", "cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF")
FILE = os.environ.get("MODEL_FILE", "Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf")
N_CTX = int(os.environ.get("N_CTX", "8192"))
PORT = int(os.environ.get("PORT", "7860"))
TOOL_ROUNDS = int(os.environ.get("TOOL_ROUNDS", "3"))
# Passwordless space -> shell stays off (an open URL must not hand out a shell). search + fetch
# give the model live internet; the container already has egress.
SHELL_OK = False

print(f"downloading {REPO}/{FILE} ...", flush=True)
model_path = hf_hub_download(REPO, FILE, cache_dir="/cache/hf")
print("loading into GPU ...", flush=True)
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=N_CTX, n_batch=512, verbose=True)
print("model ready", flush=True)


def _answer(text):
    """Drop the model's <think> preamble; return the answer body."""
    return text.split("</think>", 1)[1] if "</think>" in text else text


def _gen(msgs, temperature, max_tokens):
    """Stream one turn. `stop` cuts generation right after a tool call so we can act on it."""
    for ch in llm.create_chat_completion(
        messages=msgs, stream=True, stop=["</tool>"],
        temperature=float(temperature), max_tokens=int(max_tokens), top_p=0.9,
    ):
        yield ch["choices"][0]["delta"].get("content", "") or ""


def chat(message, history, temperature, max_tokens):
    # History back to the model is the clean answers only (Gradio holds what we yielded), or the
    # model's own leaked reasoning/tool syntax feeds back in and compounds.
    msgs = ([{"role": "system", "content": tools.system_prompt(SHELL_OK)}]
            + list(history or []) + [{"role": "user", "content": message}])
    shown = ""
    for _ in range(TOOL_ROUNDS):
        out = ""
        for delta in _gen(msgs, temperature, max_tokens):
            if not delta:
                continue
            out += delta
            body = _answer(out)
            yield shown + (body.lstrip("\n") if "</think>" in out else "_thinking…_")

        answer = _answer(out).strip()
        call = tools.parse_tool_call(answer)
        if not call:
            return                                   # ordinary reply, already streamed
        name, arg = call

        prose = answer.split("<tool>")[0].strip()
        shown += (prose + "\n\n") if prose else ""
        shown += f"> 🔧 **{name}** · `{arg.splitlines()[0][:120]}`\n"
        yield shown + "\n_running…_"

        result = tools.run(name, arg, shell_ok=SHELL_OK)
        shown += "\n```\n" + result + "\n```\n\n"
        yield shown

        msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})

    yield shown + f"\n_(stopped after {TOOL_ROUNDS} tool calls)_"


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
