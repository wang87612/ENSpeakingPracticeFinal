"""FunASR FastAPI server: SenseVoiceSmall + VAD + Punc on cuda:1.

Relocated into ENSpeakingPracticeFinal/servers/ for unified version control.

Run:
    python servers/funasr_server.py --port 10095 --device cuda:1

Endpoints:
    POST /asr     multipart/form-data: file=<audio>; optional form fields:
                  language (auto|zh|en|yue|ja|ko|nospeech), use_itn (bool)
    GET  /health
"""
import os, time, argparse, tempfile, traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import re

# Model weights directory. Override via --models-dir or FUNASR_MODELS env var.
MODELS = os.environ.get("FUNASR_MODELS", "/home/ec2-user/audio-stack/models")

# SenseVoiceSmall sometimes emits "< | zh | >"-style tokens with spaces, while
# rich_transcription_postprocess looks for the canonical "<|zh|>" form. Strip
# the inner spaces before postprocessing.
_TOKEN_NORMALIZER = re.compile(r"<\s*\|\s*([^|>]+?)\s*\|\s*>")

def _normalize_tokens(s: str) -> str:
    return _TOKEN_NORMALIZER.sub(lambda m: f"<|{m.group(1).strip().replace(' ', '')}|>", s)

app = FastAPI(title="FunASR-SenseVoice Server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_model = None
_device = "cuda:1"


def _ensure_model():
    global _model
    if _model is None:
        t0 = time.time()
        _model = AutoModel(
            model=f"{MODELS}/SenseVoiceSmall",
            vad_model=f"{MODELS}/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model=f"{MODELS}/punc_ct-transformer_cn-en-common-vocab471067-large",
            device=_device,
            disable_update=True,
        )
        print(f"[model] loaded on {_device} in {time.time()-t0:.1f}s", flush=True)
    return _model


@app.on_event("startup")
def _startup():
    _ensure_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": _device, "model": "SenseVoiceSmall"}


@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    use_itn: bool = Form(True),
):
    if file is None:
        raise HTTPException(400, "missing file")
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        t0 = time.time()
        model = _ensure_model()
        res = model.generate(
            input=tmp_path,
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        dt = time.time() - t0
        raw = res[0]["text"] if res else ""
        text = rich_transcription_postprocess(_normalize_tokens(raw))
        return JSONResponse({"text": text, "raw": raw, "elapsed_sec": round(dt, 3)})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; use 0.0.0.0 to expose on all interfaces")
    p.add_argument("--port", type=int, default=10095)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--models-dir", default=None,
                   help="path to model weights directory (overrides FUNASR_MODELS env)")
    args = p.parse_args()
    if args.models_dir:
        MODELS = args.models_dir
    _device = args.device
    globals()["_device"] = args.device
    uvicorn.run(app, host=args.host, port=args.port)
