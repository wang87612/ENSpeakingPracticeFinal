"""Download FunASR models (SenseVoiceSmall + fsmn-vad + ct-punc) from ModelScope.
Stored under ~/audio-stack/models/ for reuse outside the FunASR repo."""
import os, sys, time, traceback

BASE = "/home/ec2-user/audio-stack/models"
JOBS = [
    ("iic/SenseVoiceSmall", os.path.join(BASE, "SenseVoiceSmall")),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", os.path.join(BASE, "speech_fsmn_vad_zh-cn-16k-common-pytorch")),
    ("iic/punc_ct-transformer_cn-en-common-vocab471067-large", os.path.join(BASE, "punc_ct-transformer_cn-en-common-vocab471067-large")),
]

from modelscope import snapshot_download

for model_id, local in JOBS:
    print(f"[{time.strftime('%H:%M:%S')}] downloading {model_id} -> {local}", flush=True)
    try:
        os.makedirs(local, exist_ok=True)
        p = snapshot_download(model_id, local_dir=local)
        print(f"[{time.strftime('%H:%M:%S')}]   OK: {p}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
