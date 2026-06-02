"""Minimal CosyVoice 3.0 zero-shot TTS smoke test.

Run:
    cd ~/work/CosyVoice
    CUDA_VISIBLE_DEVICES=0 python ~/audio-stack/demo/cosyvoice3_demo.py
"""
import os, sys, time

REPO = "/home/ec2-user/work/CosyVoice"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

OUT_DIR = "/home/ec2-user/audio-stack/demo"
os.makedirs(OUT_DIR, exist_ok=True)

print("[load] AutoModel(Fun-CosyVoice3-0.5B) ...", flush=True)
t0 = time.time()
cosyvoice = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
print(f"[load] OK in {time.time()-t0:.1f}s, sample_rate={cosyvoice.sample_rate}", flush=True)

text = "你好，我是 CosyVoice 三号语音大模型，这是一段零样本声音复刻的测试。"
prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
prompt_wav = "./asset/zero_shot_prompt.wav"

print(f"[infer] zero_shot -> '{text}'", flush=True)
t0 = time.time()
saved = []
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)):
    out = os.path.join(OUT_DIR, f"cosyvoice3_zero_shot_{i}.wav")
    torchaudio.save(out, j["tts_speech"], cosyvoice.sample_rate)
    saved.append(out)
print(f"[infer] OK in {time.time()-t0:.1f}s; wrote {saved}", flush=True)
