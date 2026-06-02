"""Hit the CosyVoice3 FastAPI /inference_zero_shot endpoint and save the
streaming int16 PCM as a 24kHz wav."""
import sys, time, requests, numpy as np, soundfile as sf

HOST = "127.0.0.1"
PORT = 50000
SR_IN  = 16000   # prompt
SR_OUT = 24000   # CosyVoice3 sample rate

PROMPT_WAV = "/home/ec2-user/work/CosyVoice/asset/zero_shot_prompt.wav"
TTS_TEXT = "这是来自 CosyVoice 三号 FastAPI 服务的实时合成验证，能够听见就说明部署成功。"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
OUT = "/home/ec2-user/audio-stack/demo/cosyvoice3_server_zero_shot.wav"

url = f"http://{HOST}:{PORT}/inference_zero_shot"
data = {"tts_text": TTS_TEXT, "prompt_text": PROMPT_TEXT}
files = {"prompt_wav": open(PROMPT_WAV, "rb")}

print(f"[client] POST {url}", flush=True)
t0 = time.time()
resp = requests.post(url, data=data, files=files, stream=True, timeout=300)
resp.raise_for_status()

buf = bytearray()
for chunk in resp.iter_content(chunk_size=16000):
    if chunk:
        buf.extend(chunk)
elapsed = time.time() - t0

audio = np.frombuffer(bytes(buf), dtype=np.int16)
sf.write(OUT, audio, SR_OUT, subtype="PCM_16")
dur = len(audio) / SR_OUT
print(f"[client] OK in {elapsed:.2f}s -> {OUT} ({dur:.2f}s audio, {len(audio)} samples @ {SR_OUT}Hz)")
