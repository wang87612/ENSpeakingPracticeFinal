# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Modified for CosyVoice3 compatibility (prompt_wav must be a file path).
# Relocated into ENSpeakingPracticeFinal/servers/ for unified version control.
"""CosyVoice 3 FastAPI server: zero-shot / cross-lingual / instruct / vc TTS.

Run (from repo root):
    cd ~/work/CosyVoice
    CUDA_VISIBLE_DEVICES=0 python ~/audio-stack/servers/cosyvoice_server.py \
        --host 127.0.0.1 --port 50000 \
        --model_dir pretrained_models/Fun-CosyVoice3-0.5B

The working directory MUST be ~/work/CosyVoice (or wherever the CosyVoice repo
lives) so that `cosyvoice.cli.cosyvoice` and third_party/Matcha-TTS are on the
Python path.

Endpoints:
    POST /inference_zero_shot   tts_text + prompt_text + prompt_wav -> streaming PCM
    POST /inference_cross_lingual
    POST /inference_instruct
    POST /inference_instruct2
    POST /inference_vc
    POST /inference_sft
    GET  /health
"""
import os
import sys
import argparse
import logging
import tempfile

logging.getLogger('matplotlib').setLevel(logging.WARNING)

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: ensure CosyVoice repo and Matcha-TTS are importable.
# The start script `cd`s into ~/work/CosyVoice before launching this file,
# so cwd-relative imports work. We also add explicit paths as fallback.
# ---------------------------------------------------------------------------
_COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", os.getcwd())
sys.path.insert(0, _COSYVOICE_REPO)
sys.path.insert(0, os.path.join(_COSYVOICE_REPO, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def _spool_upload(upload: UploadFile) -> str:
    """Persist an UploadFile to a temp .wav and return its path.

    CosyVoice3's frontend reads prompt audio via torchaudio.load(file_path).
    """
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="cv3_prompt_")
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _stream_and_cleanup(model_output, *paths):
    try:
        for chunk in generate_data(model_output):
            yield chunk
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except Exception:
                pass


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": _MODEL_DIR, "sample_rate": cosyvoice.sample_rate}


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_path = _spool_upload(prompt_wav)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_path)
    return StreamingResponse(_stream_and_cleanup(model_output, prompt_path))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_path = _spool_upload(prompt_wav)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_path)
    return StreamingResponse(_stream_and_cleanup(model_output, prompt_path))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    # Note: only supported by CosyVoice 1.0 instruct model.
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_path = _spool_upload(prompt_wav)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_path)
    return StreamingResponse(_stream_and_cleanup(model_output, prompt_path))


@app.get("/inference_vc")
@app.post("/inference_vc")
async def inference_vc(source_wav: UploadFile = File(), prompt_wav: UploadFile = File()):
    src_path = _spool_upload(source_wav)
    prompt_path = _spool_upload(prompt_wav)
    model_output = cosyvoice.inference_vc(src_path, prompt_path)
    return StreamingResponse(_stream_and_cleanup(model_output, src_path, prompt_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='bind address; use 0.0.0.0 to expose on all interfaces')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()

    _MODEL_DIR = args.model_dir
    cosyvoice = AutoModel(model_dir=args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)
