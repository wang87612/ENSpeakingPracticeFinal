"""Download Fun-CosyVoice3-0.5B-2512 from ModelScope into the
CosyVoice repo's pretrained_models directory."""
import os, sys, time, traceback

LOCAL_DIR = "/home/ec2-user/work/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"[{time.strftime('%H:%M:%S')}] downloading {MODEL_ID} -> {LOCAL_DIR}", flush=True)

try:
    from modelscope import snapshot_download
    p = snapshot_download(MODEL_ID, local_dir=LOCAL_DIR)
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {p}", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
