"""Pipecat real-time voice agent: FunASR (STT) + Qwen3.6-27B (LLM) + CosyVoice 3 (TTS).

Run:
    python bot.py --host 127.0.0.1 --port 7860
Then SSH-tunnel 7860 to your laptop and open http://127.0.0.1:7860/ in a browser.

Env (loaded from ~/audio-stack/.secrets/qwen.env):
    QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL
"""
import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from loguru import logger

from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.services.tts_service import TTSService, TextAggregationMode
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_stop_strategies,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# ----- env -----
load_dotenv("/home/ec2-user/audio-stack/.secrets/qwen.env", override=True)

FUNASR_URL = os.environ.get("FUNASR_URL", "http://127.0.0.1:10095")
COSYVOICE_URL = os.environ.get("COSYVOICE_URL", "http://127.0.0.1:50000")
QWEN_BASE_URL = os.environ["QWEN_BASE_URL"]
QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_MODEL = os.environ["QWEN_MODEL"]

# CosyVoice prompt (a 16kHz mono wav from the official repo) and its transcript.
# NOTE: CosyVoice 3 requires the special token <|endofprompt|> (id 151646) to
# appear inside prompt_text or tts_text, otherwise the LLM aborts with an
# AssertionError and the HTTP body is closed mid-stream (clients see a
# TransferEncodingError "Not enough data to satisfy transfer length header").
PROMPT_WAV = "/home/ec2-user/work/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

# Persona files (edit and restart to update; no code change needed).
PERSONA_DIR = os.environ.get("PERSONA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona"))
SYSTEM_PROMPT_FILE = os.path.join(PERSONA_DIR, "system_prompt.txt")
GREETING_FILE = os.path.join(PERSONA_DIR, "greeting.txt")

def _read_text(path: str, default: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or default
    except FileNotFoundError:
        return default

SYSTEM_PROMPT = _read_text(
    SYSTEM_PROMPT_FILE,
    "你是一个友好、自然的中文语音助手。直接用口语化的中文回答，"
    "不要使用列表、emoji、markdown 等不能朗读的格式。回答要简短、自然，"
    "通常每次回复控制在 1 到 3 句之内。每句话要完整，不要用省略号或断句，避免只说两三个字的碎片。",
)
GREETING = _read_text(
    GREETING_FILE,
    "你好，请用一句话简短地自我介绍。",
)


# =============== Sensitivity knobs (env-tunable + runtime-tunable via /api/config) ===============
# These follow Pipecat's official recommendations:
#   - VAD stop_secs/start_secs are kept at defaults (0.2/0.2) because the
#     official user-turn-stop strategies (SmartTurn/SpeechTimeout) are tuned
#     against those values.
#   - "Don't interrupt me on a single grunt while the bot is talking" is
#     handled by MinWordsUserTurnStartStrategy, which only enforces the word
#     threshold while the bot is speaking — outside of bot-speech, a single
#     word still triggers (so the user can say "yes" / "stop" naturally).
CONFIG: dict[str, Any] = {
    "vad_confidence":         float(os.environ.get("VAD_CONFIDENCE", "0.7")),
    "vad_min_volume":         float(os.environ.get("VAD_MIN_VOLUME", "0.6")),
    "vad_stop_secs":          float(os.environ.get("VAD_STOP_SECS", "0.8")),
    "speech_timeout":         float(os.environ.get("SPEECH_TIMEOUT", "1.0")),
    "min_words_to_interrupt": int(os.environ.get("MIN_WORDS_TO_INTERRUPT", "3")),
}

CONFIG_BOUNDS: dict[str, tuple[float, float]] = {
    "vad_confidence":         (0.1, 1.0),
    "vad_min_volume":         (0.0, 1.0),
    "vad_stop_secs":          (0.2, 5.0),
    "speech_timeout":         (0.3, 10.0),
    "min_words_to_interrupt": (1, 20),
}


# CJK ranges: Chinese, Japanese hiragana/katakana, Korean Hangul.
_CJK_RE = __import__("re").compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


def _cjk_aware_word_count(text: str) -> int:
    """Count words in a way that makes sense for mixed CJK + Latin text.

    Pipecat's stock MinWordsUserTurnStartStrategy uses ``len(text.split())``,
    which treats "你好啊" as 1 word (no whitespace). For Chinese conversation
    that means a perfectly normal 3-character utterance can never reach
    min_words=3. Here, each CJK character counts as 1, and any contiguous
    Latin run inside the same whitespace-separated chunk that contains at
    least one alphanumeric character also counts as 1 (so pure punctuation
    like "。" does not inflate the count).
    """
    if not text:
        return 0
    import re
    has_alnum = re.compile(r"\w", re.UNICODE)
    words = 0
    for chunk in text.split():
        cjk = _CJK_RE.findall(chunk)
        non_cjk = _CJK_RE.sub("", chunk)
        words += len(cjk)
        # Only count a Latin/number run if it has any actual alphanumeric.
        # \w in Python re with UNICODE flag matches CJK too, so we strip CJK
        # first then check for word-character residue.
        if non_cjk and has_alnum.search(non_cjk):
            words += 1
    return words


class CJKAwareMinWordsStartStrategy(MinWordsUserTurnStartStrategy):
    """``MinWordsUserTurnStartStrategy`` with CJK-aware word counting.

    Same semantics as the upstream class (the threshold is only enforced
    while the bot is speaking; otherwise a single word triggers the turn),
    but CJK characters each count as one word so Chinese / Japanese /
    Korean utterances are not always counted as 1.
    """

    async def _handle_transcription(self, frame):
        # Mirror the upstream logic but with CJK-aware counting.
        from pipecat.frames.frames import InterimTranscriptionFrame

        min_words = self._min_words if self._bot_speaking else 1
        word_count = _cjk_aware_word_count(frame.text or "")
        should_trigger = word_count >= min_words
        is_interim = isinstance(frame, InterimTranscriptionFrame)

        logger.debug(
            f"{self} should_trigger={should_trigger} "
            f"num_spoken_words={word_count} min_words={min_words} "
            f"bot_speaking={self._bot_speaking} interim={is_interim} "
            f"text={frame.text!r}"
        )

        if should_trigger:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        await self.trigger_reset_aggregation()
        return ProcessFrameResult.CONTINUE


# =============== Trace logging (per-session JSONL + SSE broadcast) ===============
TRACE_DIR = os.environ.get(
    "TRACE_DIR",
    "/home/ec2-user/audio-stack/logs/trace",
)
os.makedirs(TRACE_DIR, exist_ok=True)
TRACE_AGG_FILE = os.path.join(TRACE_DIR, "trace.jsonl")
TRACE_HISTORY_MAX = 500  # keep this many recent events for late subscribers


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class Tracer:
    """Per-process tracer.

    Every event is:
      1. appended to a per-session JSONL file (TRACE_DIR/<pc_id>.jsonl)
      2. appended to the aggregated TRACE_DIR/trace.jsonl
      3. broadcast to all live SSE subscribers
      4. kept in an in-memory ring buffer so a freshly-opened /trace page can
         backfill what already happened in the current run.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history: list[dict[str, Any]] = []

    # ---- subscribe / unsubscribe (used by SSE endpoint) ----
    def subscribe(self) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q, list(self._history)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ---- emit ----
    def emit(self, pc_id: str | None, stage: str, **fields: Any) -> None:
        evt: dict[str, Any] = {
            "t": _now_iso(),
            "pc": pc_id or "-",
            "stage": stage,
            **fields,
        }
        # 1. files (best-effort, never raise)
        try:
            line = json.dumps(evt, ensure_ascii=False)
        except Exception as e:
            line = json.dumps({"t": evt["t"], "pc": evt["pc"], "stage": "trace.error",
                               "error": f"json dump failed: {e}"})
            evt = {"t": evt["t"], "pc": evt["pc"], "stage": "trace.error",
                   "error": f"json dump failed: {e}"}

        if pc_id and pc_id != "-":
            try:
                with open(os.path.join(TRACE_DIR, f"{pc_id}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        try:
            with open(TRACE_AGG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

        # 2. ring buffer
        self._history.append(evt)
        if len(self._history) > TRACE_HISTORY_MAX:
            self._history = self._history[-TRACE_HISTORY_MAX:]

        # 3. broadcast (drop on slow consumers)
        for q in list(self._subscribers):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                pass


tracer = Tracer()


def _truncate(s: str, n: int = 4000) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"...<truncated {len(s) - n} chars>"


class TraceObserver(BaseObserver):
    """Pipecat observer that converts pipeline frames into trace events.

    Scoped to a single WebRTC session (one pc_id). Keeps a small per-session
    buffer so we can emit one consolidated llm.end event per response instead
    of one event per LLM token chunk.
    """

    def __init__(self, pc_id: str, t: Tracer) -> None:
        super().__init__()
        self.pc_id = pc_id
        self.tracer = t
        self._llm_chunks: list[str] = []
        self._llm_started_at: float | None = None
        self._llm_input_seen: bool = False  # avoid duplicate llm.in per turn

    async def on_push_frame(self, data: FramePushed) -> None:
        src = data.source
        dst = data.destination
        frame = data.frame

        # --- STT output -------------------------------------------------
        if isinstance(src, STTService) and isinstance(frame, TranscriptionFrame):
            self.tracer.emit(
                self.pc_id,
                "stt",
                text=frame.text,
                user_id=getattr(frame, "user_id", None),
                language=str(getattr(frame, "language", "") or ""),
            )
            return

        # --- LLM input (the moment a context frame arrives at the LLM) --
        if isinstance(dst, LLMService) and isinstance(
            frame, (LLMContextFrame, OpenAILLMContextFrame)
        ):
            if self._llm_input_seen:
                return
            try:
                if isinstance(frame, OpenAILLMContextFrame):
                    msgs = frame.context.messages
                else:
                    msgs = frame.context.get_messages()
                # Compact: keep role + content; truncate huge content fields
                compact = []
                for m in msgs:
                    if isinstance(m, dict):
                        role = m.get("role", "?")
                        content = m.get("content", "")
                        if isinstance(content, list):
                            # multimodal; render as JSON
                            content = json.dumps(content, ensure_ascii=False)
                        elif not isinstance(content, str):
                            content = str(content)
                        compact.append({"role": role, "content": _truncate(content, 2000)})
                    else:
                        compact.append({"role": "?", "content": _truncate(str(m), 2000)})
                self.tracer.emit(self.pc_id, "llm.in", messages=compact)
                self._llm_input_seen = True
            except Exception as e:
                self.tracer.emit(self.pc_id, "llm.in.error", error=str(e))
            return

        # --- LLM output -------------------------------------------------
        if isinstance(src, LLMService):
            if isinstance(frame, LLMFullResponseStartFrame):
                self._llm_chunks = []
                self._llm_started_at = time.monotonic()
                self.tracer.emit(self.pc_id, "llm.start")
            elif isinstance(frame, LLMTextFrame):
                self._llm_chunks.append(frame.text or "")
            elif isinstance(frame, LLMFullResponseEndFrame):
                full = "".join(self._llm_chunks)
                latency = (time.monotonic() - self._llm_started_at) if self._llm_started_at else None
                self.tracer.emit(
                    self.pc_id,
                    "llm.end",
                    text=full,
                    latency_sec=round(latency, 3) if latency is not None else None,
                )
                self._llm_chunks = []
                self._llm_started_at = None
                # next turn will see a fresh llm.in
                self._llm_input_seen = False
            return


# =============== Custom STT: FunASR ===============
# SenseVoice emits a language token (e.g. "<|en|>", "<|zh|>") at the start of
# its raw output. Map FunASR's language codes to Pipecat's Language enum so the
# TranscriptionFrame carries the *actual* spoken language instead of a fixed
# label. Falls back to English when unknown (this stack's practice flow is
# English-first; the Chinese-assistant flow forces language="zh" explicitly).
_FUNASR_LANG_MAP: dict[str, Language] = {
    "zh": Language.ZH,
    "en": Language.EN,
    "ja": Language.JA,
    "ko": Language.KO,
    "yue": Language.ZH,  # Cantonese: no dedicated enum; treat as Chinese.
}

_RAW_LANG_TOKEN_RE = __import__("re").compile(r"<\|([a-z]{2,3})\|>")


def _resolve_language(forced: str, raw: str, default: Language) -> Language:
    """Pick the TranscriptionFrame language.

    1. If STT was called with an explicit language (not "auto"), trust it.
    2. Otherwise read SenseVoice's leading language token from ``raw``.
    3. Fall back to ``default``.
    """
    forced = (forced or "").lower()
    if forced and forced != "auto":
        return _FUNASR_LANG_MAP.get(forced, default)
    m = _RAW_LANG_TOKEN_RE.search(raw or "")
    if m:
        return _FUNASR_LANG_MAP.get(m.group(1), default)
    return default


class FunASRSTTService(SegmentedSTTService):
    """Segmented STT that calls FunASR /asr once per VAD-bounded segment."""

    def __init__(self, *, base_url: str, language: str = "auto", use_itn: bool = True,
                 sample_rate: int = 16000,
                 default_language: Language = Language.EN,
                 pc_id: str | None = None, tracer: Tracer | None = None,
                 **kw):
        super().__init__(sample_rate=sample_rate, **kw)
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._use_itn = use_itn
        self._default_language = default_language
        self._session: aiohttp.ClientSession | None = None
        self._pc_id = pc_id
        self._tracer = tracer

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def cancel(self, frame):
        await super().cancel(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if not audio or len(audio) < 1024:
            return
        await self.start_ttfb_metrics()
        if self._tracer:
            self._tracer.emit(
                self._pc_id, "stt.in",
                audio_bytes=len(audio),
                language=self._language,
            )
        data = aiohttp.FormData()
        data.add_field("file", audio, filename="seg.wav", content_type="audio/wav")
        data.add_field("language", self._language)
        data.add_field("use_itn", str(self._use_itn).lower())
        try:
            async with self._session.post(
                f"{self._base_url}/asr",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status != 200:
                    err = await r.text()
                    if self._tracer:
                        self._tracer.emit(self._pc_id, "stt.err",
                                          status=r.status, error=err[:500])
                    yield ErrorFrame(f"FunASR {r.status}: {err}")
                    return
                j = await r.json()
        except Exception as e:
            logger.error(f"FunASR error: {e}")
            if self._tracer:
                self._tracer.emit(self._pc_id, "stt.err", error=str(e))
            yield ErrorFrame(f"FunASR error: {e}")
            return

        await self.stop_ttfb_metrics()
        text = (j.get("text") or "").strip()
        raw = j.get("raw") or ""
        elapsed = j.get("elapsed_sec")
        if not text:
            if self._tracer:
                self._tracer.emit(self._pc_id, "stt.empty",
                                  elapsed_sec=elapsed, audio_bytes=len(audio))
            return
        logger.info(f"[STT] {text} ({elapsed}s)")
        lang = _resolve_language(self._language, raw, self._default_language)
        # The TraceObserver will see the TranscriptionFrame and emit "stt".
        # We additionally emit "stt.timing" so /trace can show how long FunASR took.
        # NOTE: noise/short-utterance filtering is handled by Pipecat's
        # CJKAwareMinWordsStartStrategy further downstream, NOT here. That
        # strategy correctly only enforces the threshold while the bot is
        # speaking, so the user can still say single-character commands like
        # "停" / "yes" when the bot is silent.
        if self._tracer:
            self._tracer.emit(self._pc_id, "stt.timing",
                              elapsed_sec=elapsed, audio_bytes=len(audio),
                              text=text, language=str(lang))
        yield TranscriptionFrame(
            text=text,
            user_id="user",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            language=lang,
        )


# =============== Custom TTS: CosyVoice 3 ===============
class CosyVoice3TTSService(TTSService):
    """Stream raw int16 PCM (24kHz mono) from CosyVoice /inference_zero_shot."""

    def __init__(self, *, base_url: str, prompt_text: str, prompt_wav_path: str,
                 sample_rate: int = 24000,
                 pc_id: str | None = None, tracer: Tracer | None = None,
                 **kw):
        super().__init__(
            sample_rate=sample_rate,
            # Let the base class own the audio-context lifecycle. Without this,
            # multi-sentence LLM replies cause the base class state machine and
            # our manual create_audio_context/remove_audio_context calls to
            # collide — the second sentence's audio is then dropped after a
            # KeyError in _audio_context_task_handler. See pipecat docs in
            # tts_service.tts_process_generator for the rule.
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=True,
            **kw,
        )
        self._base_url = base_url.rstrip("/")
        self._prompt_text = prompt_text
        self._prompt_wav_path = prompt_wav_path
        # Load prompt wav once into memory (it's small ~330KB).
        with open(prompt_wav_path, "rb") as f:
            self._prompt_bytes = f.read()
        self._session: aiohttp.ClientSession | None = None
        self._pc_id = pc_id
        self._tracer = tracer

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def cancel(self, frame):
        await super().cancel(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return
        logger.info(f"[TTS] synth: {text}")
        if self._tracer:
            self._tracer.emit(self._pc_id, "tts.in",
                              text=text, context_id=context_id)
        await self.start_tts_usage_metrics(text)

        data = aiohttp.FormData()
        data.add_field("tts_text", text)
        data.add_field("prompt_text", self._prompt_text)
        data.add_field(
            "prompt_wav",
            self._prompt_bytes,
            filename="prompt.wav",
            content_type="audio/wav",
        )

        t0 = time.monotonic()
        total_bytes = 0
        try:
            async with self._session.post(
                f"{self._base_url}/inference_zero_shot",
                data=data,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    if self._tracer:
                        self._tracer.emit(self._pc_id, "tts.err",
                                          status=resp.status, error=err[:500],
                                          text=text)
                    yield ErrorFrame(f"CosyVoice {resp.status}: {err}")
                    return
                # Read the FULL audio before emitting any frame so playback is
                # gapless (same as the /api/tts_clip replay path). Streaming
                # network chunks directly caused choppy first-time playback.
                full_audio = await resp.read()
                await self.stop_ttfb_metrics()

                async def _one_shot():
                    yield full_audio

                async for f in self._stream_audio_frames_from_iterator(
                    _one_shot(),
                    in_sample_rate=24000,
                    context_id=context_id,
                ):
                    audio = getattr(f, "audio", None)
                    if isinstance(audio, (bytes, bytearray)):
                        total_bytes += len(audio)
                    yield f
            if self._tracer:
                self._tracer.emit(self._pc_id, "tts.out",
                                  text=text,
                                  audio_bytes=total_bytes,
                                  latency_sec=round(time.monotonic() - t0, 3))
        except Exception as e:
            logger.error(f"CosyVoice TTS error: {e}")
            if self._tracer:
                self._tracer.emit(self._pc_id, "tts.err",
                                  error=str(e), text=text,
                                  latency_sec=round(time.monotonic() - t0, 3))
            yield ErrorFrame(f"CosyVoice error: {e}")
        # NOTE: do NOT call remove_audio_context here — base class closes the
        # context via _synthesize_text after we return. Calling it manually
        # races with the base-class state machine and breaks subsequent
        # sentences in the same LLM turn.


# =============== App ===============
app = FastAPI()
pcs_map: dict[str, SmallWebRTCConnection] = {}
tasks_map: dict[str, tuple] = {}  # pc_id -> (worker, context)
ice_servers = [
    IceServer(urls="stun:stun.cloudflare.com:3478"),
    IceServer(urls="stun:stun.l.google.com:19302"),
]
app.mount("/client", SmallWebRTCPrebuiltUI)

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_HTML = os.path.join(_HERE, "static", "index.html")
_TRACE_HTML = os.path.join(_HERE, "static", "trace.html")
_PRACTICE_HTML = os.path.join(_HERE, "static", "practice.html")


async def run_bot(webrtc_connection: SmallWebRTCConnection,
                  system_prompt: str | None = None,
                  greeting: str | None = None,
                  voice_id: str | None = None,
                  vad_stop_secs: float | None = None,
                  speech_timeout: float | None = None,
                  stt_language: str | None = None):
    pc_id = webrtc_connection.pc_id
    logger.info(f"Starting bot pipeline pc_id={pc_id}")
    sp = (system_prompt or SYSTEM_PROMPT).strip() or SYSTEM_PROMPT
    gr = (greeting or GREETING).strip() or GREETING
    tts_wav, tts_prompt_text = _get_voice(voice_id)
    # Session language drives STT. Practice (English) passes "en" so SenseVoice
    # is not allowed to misdetect accented English as Chinese; the Chinese
    # assistant passes nothing -> "auto" with a ZH fallback (unchanged behavior).
    stt_lang = (stt_language or "auto").lower()
    stt_default_lang = Language.EN if stt_lang == "en" else Language.ZH
    logger.info(f"persona: system_prompt[{len(sp)} chars] greeting[{len(gr)} chars] "
                f"voice={voice_id} stt_language={stt_lang}")
    tracer.emit(pc_id, "session.start",
                system_prompt=_truncate(sp, 2000),
                greeting=_truncate(gr, 500),
                qwen_model=QWEN_MODEL)

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,    # match FunASR expectation
            audio_out_sample_rate=24000,   # match CosyVoice output
            # Per Pipecat docs: VAD start_secs controls how long the user
            # must sustain speech before VAD confirms "speaking started" and
            # emits VADUserStartedSpeakingFrame. We set it to 1.0s so brief
            # noises / back-channel sounds don't interrupt the bot, but
            # sustained speech (1+ second) reliably triggers interruption.
            # NOTE: This is higher than the default 0.2s. Quick utterances
            # like "yes"/"no" won't interrupt while bot is speaking, which
            # is acceptable for this practice scenario.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=float(CONFIG["vad_confidence"]),
                    start_secs=1.0,
                    stop_secs=float(vad_stop_secs if vad_stop_secs is not None else CONFIG["vad_stop_secs"]),
                    min_volume=float(CONFIG["vad_min_volume"]),
                ),
            ),
        ),
    )

    stt = FunASRSTTService(
        base_url=FUNASR_URL, language=stt_lang,
        default_language=stt_default_lang,
        pc_id=pc_id, tracer=tracer,
    )
    tts = CosyVoice3TTSService(
        base_url=COSYVOICE_URL,
        prompt_text=tts_prompt_text,
        prompt_wav_path=tts_wav,
        pc_id=pc_id, tracer=tracer,
    )
    llm = OpenAILLMService(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL,
        params=OpenAILLMService.InputParams(
            temperature=0.5,
            max_tokens=256,
            # Aliyun Bailian (DashScope) hosts Qwen3 with "thinking mode" on by
            # default, which streams long `reasoning_content` deltas before any
            # actual `content`. That ruins voice-agent latency, so disable it
            # via the OpenAI SDK's extra_body passthrough.
            extra={"extra_body": {"enable_thinking": False}},
        ),
    )

    context = LLMContext(messages=[{"role": "system", "content": sp}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Interruption strategy:
            # VADUserTurnStartStrategy triggers based on VAD speech detection,
            # NOT on transcription text. This is critical because FunASR is a
            # SegmentedSTTService that only returns text AFTER the user stops
            # speaking — so word-count-based strategies can never interrupt
            # during bot speech. VAD-based detection works immediately.
            #
            # The VAD start_secs (configured on the transport above) controls
            # how long speech must be sustained before VAD confirms "user is
            # speaking". Combined with VADUserTurnStartStrategy, this means:
            # user speaks for start_secs → VAD fires → turn starts → bot interrupted.
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(),
                    CJKAwareMinWordsStartStrategy(
                        min_words=int(CONFIG["min_words_to_interrupt"]),
                    ),
                ],
                stop=[SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=float(speech_timeout if speech_timeout is not None else CONFIG["speech_timeout"]),
                )],
            ),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[TraceObserver(pc_id, tracer)],
    )
    tasks_map[pc_id] = (worker, context)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — sending greeting")
        tracer.emit(pc_id, "client.connected")
        context.add_message({
            "role": "user",
            "content": gr,
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        tracer.emit(pc_id, "client.disconnected")
        tasks_map.pop(pc_id, None)
        await worker.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(worker)
    finally:
        tasks_map.pop(pc_id, None)
        tracer.emit(pc_id, "session.end")


@app.get("/", include_in_schema=False)
async def root():
    with open(_INDEX_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/prebuilt", include_in_schema=False)
async def prebuilt_redirect():
    return RedirectResponse(url="/client/")


@app.get("/practice", include_in_schema=False)
async def practice_page():
    with open(_PRACTICE_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "funasr": FUNASR_URL,
        "cosyvoice": COSYVOICE_URL,
        "qwen_base_url": QWEN_BASE_URL,
        "qwen_model": QWEN_MODEL,
        "active_pcs": len(pcs_map),
    }


VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "voices")


def _load_voices():
    path = os.path.join(VOICES_DIR, "voices.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    return []


def _get_voice(voice_id: str | None):
    """Return (wav_path, prompt_text) for a voice_id, or defaults."""
    if not voice_id:
        return PROMPT_WAV, PROMPT_TEXT
    for v in _load_voices():
        if v["id"] == voice_id:
            wav = os.path.join(VOICES_DIR, v["file"])
            if os.path.isfile(wav):
                return wav, v["prompt_text"]
    return PROMPT_WAV, PROMPT_TEXT


@app.get("/api/voices")
def list_voices():
    """List available voice options."""
    voices = _load_voices()
    return {"voices": [{"id": v["id"], "name": v["name"]} for v in voices]}


SCENARIOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "scenarios")


@app.get("/api/scenarios")
def list_scenarios():
    """List available practice scenarios."""
    scenarios = []
    if os.path.isdir(SCENARIOS_DIR):
        for f in sorted(os.listdir(SCENARIOS_DIR)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(SCENARIOS_DIR, f), "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read())
                scenarios.append({"id": data["id"], "name": data["name"], "description": data.get("description", "")})
            except Exception:
                continue
    return {"scenarios": scenarios}


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    """Get full scenario prompt by id.

    If the scenario JSON contains a 'scenario_content' field (instead of a
    full 'system_prompt'), the server reads `base_prompt.txt` from the
    scenarios directory and merges them by replacing the {{SCENARIO_CONTENT}}
    placeholder. This allows authors to only write the topic-specific parts
    (main problem + sub-problems) while reusing the shared prompt scaffold.
    """
    if "/" in scenario_id or ".." in scenario_id:
        raise HTTPException(status_code=400, detail="invalid id")
    path = os.path.join(SCENARIOS_DIR, f"{scenario_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="scenario not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    # If the scenario already has a fully composed system_prompt, return as-is.
    if "system_prompt" not in data and "scenario_content" in data:
        base_path = os.path.join(SCENARIOS_DIR, "base_prompt.txt")
        try:
            with open(base_path, "r", encoding="utf-8") as bf:
                base_template = bf.read()
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="base_prompt.txt not found")
        data["system_prompt"] = base_template.replace(
            "{{SCENARIO_CONTENT}}", data["scenario_content"]
        )

    return data


# =============== Qwen prompt templates (centralized for debugging) ===============
# These are the system-side templates used to drive Qwen for the practice page.
# They are exposed verbatim via GET /api/debug/prompts so you can inspect what
# the model is actually receiving from the browser's devtools console.

TRANSLATE_SYSTEM_PROMPT = (
    "将以下英文翻译成自然的中文。只返回中文翻译结果，不要任何解释。"
)

SUGGESTIONS_SYSTEM_PROMPT = (
    "你是一个英语口语教练，帮助 AWS 技术支持工程师练习英语。"
    "根据对话记录，建议工程师接下来可以说什么。\n\n"
    "风格要求（非常重要）：\n"
    "- 真实电话场景的英语。非母语者用短句和直接的表达效果最好。\n"
    "- 每条建议只有一句话，理想情况 6-12 个词，绝不超过 14 个词。\n"
    "- 口语化英语。禁止使用 \"Could you please...\"、\"I would like to...\"、"
    "\"Just to clarify...\"。用简单动词（check、see、share、try、send）和常用短语。\n"
    "- 清晰明确。越简单越好。\n\n"
    "混合回复类型——至少包含以下适合当前场景的各一条：\n"
    "- 简短的追问（如 \"What's the exact error?\"、\"Which region?\"）\n"
    "- 简短的状态/发现（如 \"I checked the policy. There's a deny rule.\"）\n"
    "- 简短的下一步动作（如 \"Let me check that now.\"、\"Try removing it and tell me.\"）\n\n"
    "每条建议必须直接回应客户的最后一句话，并推动对话前进。"
    "不要重复工程师已经说过的内容。对话开始后不要用通用开场白。\n\n"
    "如果工程师刚说了要去检查某件事，那建议必须假设检查已完成——"
    "报告一个简短结论、提一个问题、或建议一个修复方案。\n\n"
    "输出格式（严格遵守）：\n"
    "- 只返回一个 JSON 对象，包含一个 key \"suggestions\"，其值为包含 3 到 5 个对象的数组。\n"
    "- 数组里每个对象有两个 key：\"en\"（英语建议）和 \"zh\"（对应的简短中文翻译）。\n"
    "- 中文必须准确对应英文含义，也要简短。"
)

SUMMARY_SYSTEM_PROMPT = (
    "你是一个英语教练，正在复盘一位 AWS 技术支持工程师的练习对话。"
    "对话中 'Engineer' 是练习的人类，'Customer' 是角色扮演机器人。\n\n"
    "你的任务：评判工程师的英语口语表达（不是客户的），帮助他改进。"
    "重点关注：语法错误、用词别扭、没有清楚表达工程师本意的句子。"
    "对于每个问题，给出一个简单、实用的替代句——"
    "真正的 AWS 技术支持工程师在电话中会说的话。"
    "美式商务英语。简短清楚 > 花哨复杂。\n\n"
    "只返回一个 JSON 对象，格式如下：\n"
    "{\n"
    "  \"score\": <1到10的整数，英语沟通综合评分>,\n"
    "  \"summary\": \"<2-3句总体反馈，用简体中文>\",\n"
    "  \"grammar_tips\": [\n"
    "    {\"original\": \"<工程师原句>\", \"issue\": \"<简短中文说明问题>\", \"better\": \"<简单清楚的英文替代句>\"}\n"
    "  ]\n"
    "}\n\n"
    "grammar_tips 规则：\n"
    "- 有问题时给 3 到 6 条；如果工程师英语很好，给 1-2 条表扬（issue 写\"表达自然\"，better 重复原句即可）。\n"
    "- 挑最值得改进的句子。忽略拼写错误和缩写。\n"
    "- 'better' 字段必须简短（不超过 25 个词）、口语化、意思明确。\n"
    "- JSON 对象之外不要输出任何内容。"
)


@app.get("/api/debug/prompts")
def debug_prompts():
    """Return the static prompt templates used to drive Qwen on the practice
    page. Useful for browser-console inspection."""
    return {
        "translate_system": TRANSLATE_SYSTEM_PROMPT,
        "suggestions_system": SUGGESTIONS_SYSTEM_PROMPT,
        "summary_system": SUMMARY_SYSTEM_PROMPT,
        "qwen_base_url": QWEN_BASE_URL,
        "qwen_model": QWEN_MODEL,
    }


@app.post("/api/translate")
async def api_translate(req: dict):
    """Translate English text to Chinese via Qwen."""
    text = (req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    payload = [
        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(
                f"{QWEN_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": QWEN_MODEL,
                    "messages": payload,
                    "temperature": 0.3,
                    "max_tokens": 200,
                    # Disable Bailian/Qwen3 thinking mode (it leaves content=null
                    # and emits long reasoning_content instead).
                    "enable_thinking": False,
                },
            ) as r:
                j = await r.json()
        msg = (j.get("choices") or [{}])[0].get("message") or {}
        zh = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        return {
            "zh": zh,
            "prompts": {"system": TRANSLATE_SYSTEM_PROMPT, "user": text},
        }
    except Exception as e:
        logger.exception(f"/api/translate failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/suggestions")
async def api_suggestions(req: dict):
    """Given conversation context, return 3-5 diverse suggested English replies for the practitioner (AWS support engineer)."""
    raw_messages = req.get("messages") or []
    scenario_id = (req.get("scenario_id") or "").strip()

    # Frontend role mapping (see static/practice.html addMessage):
    #   role == "user"      -> what the practitioner (Engineer) said (from STT)
    #   role == "assistant" -> what the bot (Customer) said
    # Qwen will be confused if we feed it raw chat with those role labels and
    # then ask it to predict the next "user" message. Instead we hand it a
    # transcript with explicit speaker labels so it can clearly see who said
    # what and reply *as* the Engineer.
    transcript_lines: list[str] = []
    for m in raw_messages[-12:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        speaker = "Engineer" if role == "user" else "Customer"
        transcript_lines.append(f"{speaker}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no conversation yet)"

    # Pull in the scenario description (if any) for grounding.
    scenario_blurb = ""
    if scenario_id and "/" not in scenario_id and ".." not in scenario_id:
        path = os.path.join(SCENARIOS_DIR, f"{scenario_id}.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read())
                desc = (data.get("description") or "").strip()
                name = (data.get("name") or "").strip()
                if name or desc:
                    scenario_blurb = f"Scenario: {name}. {desc}".strip()
            except Exception:
                scenario_blurb = ""

    last_customer_line = next(
        (m.get("content", "").strip()
         for m in reversed(raw_messages)
         if m.get("role") == "assistant" and (m.get("content") or "").strip()),
        "",
    )

    system_prompt = SUGGESTIONS_SYSTEM_PROMPT

    user_prompt_parts = []
    if scenario_blurb:
        user_prompt_parts.append(scenario_blurb)
    user_prompt_parts.append("目前对话记录（最新在最下面）：\n" + transcript)
    if last_customer_line:
        user_prompt_parts.append(f"工程师需要回复客户的最后一句话：\"{last_customer_line}\"")
    user_prompt_parts.append("只返回一个 JSON 对象，其中 \"suggestions\" 是包含 3-5 条工程师下一句建议的数组。")
    user_prompt_parts.append(
        "输出格式示例（严格按照此结构，替换为你的建议）：\n"
        "{\"suggestions\":[{\"en\":\"What's the exact error?\",\"zh\":\"具体的错误是什么？\"},"
        "{\"en\":\"Let me check the policy.\",\"zh\":\"我看下策略。\"},"
        "{\"en\":\"Try it again and tell me.\",\"zh\":\"再试一次然后告诉我。\"}]}"
    )
    user_prompt = "\n\n".join(user_prompt_parts)

    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(
                f"{QWEN_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": QWEN_MODEL,
                    "messages": payload,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_tokens": 600,
                    # Bailian's qwen3 / qwen3.6 family default to "thinking
                    # mode": they emit reasoning_content and leave content
                    # null until reasoning finishes, which then often
                    # overruns max_tokens. Disable thinking for this
                    # short-suggestion task.
                    "enable_thinking": False,
                    # Force valid JSON. DashScope requires the word "json" to
                    # appear in the messages (the prompts above already do).
                    # The regex fallbacks below stay as a safety net in case
                    # the model still returns malformed output.
                    "response_format": {"type": "json_object"},
                },
            ) as r:
                j = await r.json()
        msg = (j.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        import re as _re

        def _extract_quoted(s: str) -> list[str]:
            # Pull every double-quoted substring out of `s`, handling JSON
            # backslash escapes. Robust to: missing commas between strings,
            # missing closing bracket, leading prose, line breaks.
            quoted = _re.findall(r'"((?:[^"\\]|\\.)*)"', s, _re.DOTALL)
            out = []
            for q in quoted:
                v = q.replace('\\"', '"').replace('\\n', ' ').replace('\\\\', '\\').strip()
                if v:
                    out.append(v)
            return out

        # Each entry is {"en": str, "zh": str}.
        suggestions: list[dict] = []

        def _collect(items) -> None:
            for item in items:
                if isinstance(item, dict):
                    en = str(item.get("en") or item.get("english") or "").strip()
                    zh = str(item.get("zh") or item.get("chinese") or "").strip()
                    if en:
                        suggestions.append({"en": en, "zh": zh})
                elif isinstance(item, str) and item.strip():
                    suggestions.append({"en": item.strip(), "zh": ""})

        # Strategy 0: JSON mode returns a top-level object, e.g.
        # {"suggestions": [...]}. Parse it and pull the first list value.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                arr = obj.get("suggestions")
                if not isinstance(arr, list):
                    arr = next((v for v in obj.values() if isinstance(v, list)), None)
                if isinstance(arr, list):
                    _collect(arr)
            elif isinstance(obj, list):
                _collect(obj)
        except Exception:
            pass

        # Strategy 0b: JSON mode output but truncated mid-stream (max_tokens).
        # The outer object never closes, so json.loads fails above. Salvage
        # every *complete* {...} object so we still render cards instead of
        # dumping raw JSON text into the suggestion box.
        if not suggestions and '"en"' in text:
            for blk in _re.findall(r'\{[^{}]*\}', text):
                try:
                    item = json.loads(blk)
                except Exception:
                    continue
                if isinstance(item, dict) and (item.get("en") or item.get("english")):
                    _collect([item])

        # Strategy 1: strict JSON array of objects.
        m = _re.search(r'\[.*\]', text, _re.DOTALL)
        if not suggestions and m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    _collect(parsed)
            except Exception:
                pass

        # Strategy 2: array literal with broken JSON — extract quoted
        # strings, skip ones that look like JSON keys, then pair adjacent
        # strings if the second has CJK (i.e. en/zh).
        if not suggestions and m:
            KEY_LITERALS = {"en", "zh", "english", "chinese"}
            quoted = [q for q in _extract_quoted(m.group())
                      if q.strip().lower() not in KEY_LITERALS]
            if len(quoted) >= 2:
                def _has_cjk(s: str) -> bool:
                    return any('\u4e00' <= ch <= '\u9fff' for ch in s)
                i = 0
                while i < len(quoted):
                    en_cand = quoted[i]
                    zh_cand = ""
                    if (i + 1 < len(quoted) and _has_cjk(quoted[i + 1])
                            and not _has_cjk(en_cand)):
                        zh_cand = quoted[i + 1]
                        i += 2
                    else:
                        i += 1
                    if en_cand:
                        suggestions.append({"en": en_cand, "zh": zh_cand})

        # Strategy 3: line-split fallback (numbered / bulleted lines).
        if not suggestions:
            for ln in text.strip().splitlines():
                ln = ln.strip().lstrip('[').rstrip(']').rstrip(',')
                ln = _re.sub(r'^\s*(?:\d+[\.\)]|[-*•])\s*', '', ln)
                ln = ln.strip().strip('"').strip("'").strip(',')
                if ln:
                    suggestions.append({"en": ln, "zh": ""})

        # Sanitize, dedupe (case-insensitive on en), clamp to 5.
        seen: set[str] = set()
        clean: list[dict] = []
        for s in suggestions:
            en = (s.get("en") or "").strip().strip('"').strip("'")
            zh = (s.get("zh") or "").strip().strip('"').strip("'")
            if not en:
                continue
            key = en.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append({"en": en, "zh": zh})
            if len(clean) >= 5:
                break
        return {
            "suggestions": clean,
            "prompts": {"system": system_prompt, "user": user_prompt},
        }
    except Exception as e:
        logger.exception(f"/api/suggestions failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# =============== Practice session history ===============

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    if not session_id or "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session id")
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _new_session_id() -> str:
    # Sortable filename + small random tail to avoid collisions.
    import secrets
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _format_transcript_for_summary(messages: list[dict]) -> str:
    lines = []
    for m in messages or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            tag = "🎤" if m.get("type") != "text" else "💬"
            lines.append(f"Engineer {tag}: {content}")
        else:
            lines.append(f"Customer: {content}")
    return "\n".join(lines) if lines else "(empty conversation)"


async def _generate_summary(messages: list[dict], settings: dict) -> dict:
    """Call Qwen to grade the Engineer's English and produce simple, practical
    fix-up suggestions. Returns a dict with score / summary / grammar_tips /
    improved_replies. Failures are reported as a structured error inside the
    dict so the session can still be saved."""
    transcript = _format_transcript_for_summary(messages)

    # Calculate voice ratio for scoring
    user_msgs = [m for m in (messages or []) if m.get("role") == "user" and (m.get("content") or "").strip()]
    voice_count = sum(1 for m in user_msgs if m.get("type") != "text")
    total_user = len(user_msgs)
    voice_ratio = voice_count / total_user if total_user > 0 else 0

    system_prompt = SUMMARY_SYSTEM_PROMPT

    settings_blurb = json.dumps(settings or {}, ensure_ascii=False)
    user_prompt = (
        f"对话记录（Engineer 是练习的人类，🎤=语音消息，💬=文本消息）：\n\n{transcript}\n\n"
        f"练习设置：{settings_blurb}\n"
        f"口语比例：{voice_count}/{total_user} 条消息通过语音发送（{voice_ratio:.0%}）\n\n"
        "评分规则补充：这是口语练习，口语比例直接影响评分。"
        "如果口语比例低于50%，最高分不超过6分；低于30%，最高不超过4分。"
        "100%语音则不扣分。请在 summary 中提及口语比例情况。\n\n"
        "现在输出 JSON 对象。"
    )

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(
                f"{QWEN_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": QWEN_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": 900,
                    "enable_thinking": False,
                    # Summary already expects a top-level JSON object, so JSON
                    # mode is a natural fit. The {...} regex extraction below
                    # remains as a safety net.
                    "response_format": {"type": "json_object"},
                },
            ) as r:
                j = await r.json()
        msg = (j.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""

        import re as _re

        # Pull the first {...} block.
        obj = None
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except Exception:
                obj = None
        if not isinstance(obj, dict):
            return {
                "score": None,
                "summary": "（生成总结失败，原始返回已保存）",
                "grammar_tips": [],
                "raw": text[:2000],
                "prompts": {"system": system_prompt, "user": user_prompt},
            }

        # Normalize.
        try:
            score = int(obj.get("score"))
            if score < 1: score = 1
            if score > 10: score = 10
        except Exception:
            score = None

        summary_text = str(obj.get("summary") or "").strip()
        tips_raw = obj.get("grammar_tips") or []
        tips = []
        if isinstance(tips_raw, list):
            for t in tips_raw:
                if not isinstance(t, dict):
                    continue
                tips.append({
                    "original": str(t.get("original") or "").strip(),
                    "issue": str(t.get("issue") or "").strip(),
                    "better": str(t.get("better") or "").strip(),
                })
        return {
            "score": score,
            "summary": summary_text,
            "grammar_tips": tips,
            "prompts": {"system": system_prompt, "user": user_prompt},
        }
    except Exception as e:
        logger.exception(f"summary generation failed: {e}")
        return {
            "score": None,
            "summary": f"（生成总结失败：{e}）",
            "grammar_tips": [],
            "prompts": {"system": system_prompt, "user": user_prompt},
        }


@app.get("/api/sessions")
def list_sessions():
    """List all saved practice sessions, newest first. Returns a compact view."""
    items = []
    if os.path.isdir(SESSIONS_DIR):
        for fn in os.listdir(SESSIONS_DIR):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(SESSIONS_DIR, fn)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read())
                items.append({
                    "id": data.get("id") or fn[:-5],
                    "created_at": data.get("created_at"),
                    "scenario_name": data.get("scenario_name"),
                    "settings": data.get("settings") or {},
                    "score": (data.get("summary") or {}).get("score"),
                    "message_count": len(data.get("messages") or []),
                })
            except Exception:
                continue
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"sessions": items}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Get the full saved session by id."""
    path = _session_path(session_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="session not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


@app.post("/api/sessions")
async def create_session(req: dict):
    """Generate a Qwen-based summary for a finished practice session.
    No longer saves to disk — the frontend stores in localStorage.

    Body:
      messages       : [{role:"user"|"assistant", content:str, type?:str}]
      settings       : dict (optional)
    """
    messages = req.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages required")
    settings = req.get("settings") or {}

    summary = await _generate_summary(messages, settings)
    return {"summary": summary}


@app.post("/api/tts_clip")
async def api_tts_clip(req: dict):
    """Synthesize text via CosyVoice and return audio/wav for replay."""
    text = (req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    voice_id = req.get("voice_id")
    wav_path, prompt_text = _get_voice(voice_id)
    with open(wav_path, "rb") as f:
        prompt_bytes = f.read()
    data = aiohttp.FormData()
    data.add_field("tts_text", text)
    data.add_field("prompt_text", prompt_text)
    data.add_field("prompt_wav", prompt_bytes, filename="prompt.wav", content_type="audio/wav")
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(f"{COSYVOICE_URL}/inference_zero_shot", data=data, timeout=timeout) as r:
                if r.status != 200:
                    raise HTTPException(status_code=502, detail=await r.text())
                audio_bytes = await r.read()
        import struct
        sr = 24000
        nc = 1
        bps = 16
        byte_rate = sr * nc * bps // 8
        block_align = nc * bps // 8
        data_size = len(audio_bytes)
        header = struct.pack('<4sI4s4sIHHIIHH4sI',
                             b'RIFF', 36 + data_size, b'WAVE',
                             b'fmt ', 16, 1, nc, sr, byte_rate, block_align, bps,
                             b'data', data_size)
        from fastapi.responses import Response
        return Response(content=header + audio_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/models/health")
async def models_health():
    """Probe the 3 upstream services so the UI can show live status.
    CosyVoice has no /health, so any HTTP response counts as 'listening'."""
    async def check(url, headers=None, ok_any=False):
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url, headers=headers) as r:
                    return True if ok_any else r.status == 200
        except Exception:
            return False
    funasr, cosyvoice, qwen = await asyncio.gather(
        check(f"{FUNASR_URL}/health"),
        check(COSYVOICE_URL, ok_any=True),
        check(f"{QWEN_BASE_URL}/models", headers={"Authorization": f"Bearer {QWEN_API_KEY}"}),
    )
    return {"funasr": funasr, "cosyvoice": cosyvoice, "qwen": qwen}


@app.get("/api/persona")
def get_persona():
    """Return the current default persona (used to pre-fill the UI)."""
    return {
        "system_prompt_default": SYSTEM_PROMPT,
        "greeting_default": GREETING,
    }


# =============== Runtime config API ===============
@app.get("/api/config")
def get_config():
    """Return the current sensitivity knobs and their bounds."""
    return {
        "config": dict(CONFIG),
        "bounds": {k: list(v) for k, v in CONFIG_BOUNDS.items()},
        "active_pcs": len(pcs_map),
        "note": "VAD changes apply to the next session; STT filters apply immediately.",
    }


@app.post("/api/config")
def set_config(req: dict):
    """Update one or more knobs. Validates types and bounds."""
    accepted: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for k, v in req.items():
        if k not in CONFIG:
            errors[k] = "unknown key"
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors[k] = "must be a number"
            continue
        if k == "min_words_to_interrupt":
            v = int(v)
        lo, hi = CONFIG_BOUNDS[k]
        if v < lo or v > hi:
            errors[k] = f"out of range [{lo}, {hi}]"
            continue
        accepted[k] = v

    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors, "accepted": accepted})

    CONFIG.update(accepted)
    logger.info(f"runtime config updated: {accepted}")
    return {"config": dict(CONFIG), "applied": accepted}


# =============== Trace API ===============
@app.get("/trace", include_in_schema=False)
async def trace_page():
    with open(_TRACE_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/trace/sessions")
def trace_sessions():
    """List sessions on disk (any <pc_id>.jsonl in TRACE_DIR), newest first."""
    out = []
    try:
        for name in os.listdir(TRACE_DIR):
            if not name.endswith(".jsonl") or name == "trace.jsonl":
                continue
            path = os.path.join(TRACE_DIR, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            out.append({
                "pc_id": name[:-len(".jsonl")],
                "size": st.st_size,
                "mtime": st.st_mtime,
            })
    except FileNotFoundError:
        pass
    out.sort(key=lambda x: x["mtime"], reverse=True)
    active = set(pcs_map.keys())
    for s in out:
        s["active"] = s["pc_id"] in active
    return {"sessions": out, "active_pcs": list(active)}


@app.get("/api/trace/session/{pc_id}")
def trace_session(pc_id: str):
    """Return all events for a single session as a JSON array."""
    # sanitize: only allow simple ids, no path traversal
    if not pc_id or "/" in pc_id or ".." in pc_id:
        raise HTTPException(status_code=400, detail="invalid pc_id")
    path = os.path.join(TRACE_DIR, f"{pc_id}.jsonl")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="session not found")
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return {"pc_id": pc_id, "events": events}


@app.get("/api/trace/stream")
async def trace_stream(pc_id: str | None = None):
    """SSE stream of every trace event in real time.

    Optional ?pc_id= filter restricts events to that session. Late subscribers
    get a backfill of recent events from the in-memory ring buffer first.
    """
    q, history = tracer.subscribe()

    async def gen():
        try:
            yield ": connected\n\n"
            # backfill: replay history (filtered)
            for evt in history:
                if pc_id and evt.get("pc") != pc_id:
                    continue
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=15.0)
                    if pc_id and evt.get("pc") != pc_id:
                        continue
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            tracer.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/inject_text")
async def inject_text(request: dict):
    pc_id = request.get("pc_id")
    text = (request.get("text") or "").strip()
    if not pc_id or not text:
        raise HTTPException(400, "pc_id and text required")
    entry = tasks_map.get(pc_id)
    if not entry:
        raise HTTPException(404, "session not found")
    worker, context = entry
    context.add_message({"role": "user", "content": f"[文本消息] {text}"})
    await worker.queue_frames([StartInterruptionFrame(), LLMRunFrame()])
    return {"ok": True}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    system_prompt = request.get("system_prompt")
    greeting = request.get("greeting")
    voice_id = request.get("voice_id")
    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing connection {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding pc {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, pipecat_connection, system_prompt, greeting, voice_id,
                                   request.get("vad_stop_secs"), request.get("speech_timeout"),
                                   request.get("stt_language"))

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs_map.clear()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-certfile", default=None)
    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
