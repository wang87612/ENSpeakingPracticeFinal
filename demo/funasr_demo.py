"""FunASR offline inference demo using SenseVoiceSmall + fsmn-vad + ct-punc.

Usage:
    python funasr_demo.py [audio.wav]

If no audio path given, falls back to FunASR's bundled example zh.mp3.
"""
import os, sys, time

MODELS = "/home/ec2-user/audio-stack/models"

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import re

# SenseVoiceSmall sometimes emits "< | zh | >" form; canonicalize to "<|zh|>".
_TOKEN_NORMALIZER = re.compile(r"<\s*\|\s*([^|>]+?)\s*\|\s*>")
def _normalize_tokens(s: str) -> str:
    return _TOKEN_NORMALIZER.sub(lambda m: f"<|{m.group(1).strip().replace(' ', '')}|>", s)

print("[setup] loading SenseVoiceSmall + VAD + Punc on cuda:1 ...", flush=True)
t0 = time.time()
model = AutoModel(
    model=f"{MODELS}/SenseVoiceSmall",
    vad_model=f"{MODELS}/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    punc_model=f"{MODELS}/punc_ct-transformer_cn-en-common-vocab471067-large",
    device="cuda:1",
    disable_update=True,
)
print(f"[setup] loaded in {time.time()-t0:.1f}s", flush=True)

# pick an audio file
if len(sys.argv) > 1:
    audio = sys.argv[1]
else:
    cand = os.path.join(MODELS, "SenseVoiceSmall", "example", "zh.mp3")
    audio = cand if os.path.exists(cand) else None

if audio is None or not os.path.exists(audio):
    print(f"[error] no audio file found (tried {audio})", file=sys.stderr)
    sys.exit(1)

print(f"[infer] audio = {audio}", flush=True)
t0 = time.time()
res = model.generate(
    input=audio,
    cache={},
    language="auto",        # auto/zh/en/yue/ja/ko/nospeech
    use_itn=True,           # inverse text normalization (digits etc.)
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
dt = time.time() - t0
text = rich_transcription_postprocess(_normalize_tokens(res[0]["text"]))
print(f"[result] ({dt:.2f}s)")
print(text)
