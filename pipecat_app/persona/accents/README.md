# 口音参考音频

每种口音需要两个文件：
- `{accent_id}.wav` — 5-10秒的参考音频（16kHz mono WAV，清晰无噪音）
- `{accent_id}.txt` — 参考音频对应的文字内容（末尾需要加 `<|endofprompt|>`）

accent_id 对应前端下拉框的 value：
- american
- british
- indian
- chinese
- japanese

示例：
```
# indian.txt
Hello, I am calling from Mumbai and I need some help with my account.<|endofprompt|>
```

找参考音频的建议：
- YouTube 上找对应口音的英语演讲/播客片段
- 截取一段清晰的 5-10 秒
- 用 ffmpeg 转换：ffmpeg -i input.mp3 -ar 16000 -ac 1 indian.wav
