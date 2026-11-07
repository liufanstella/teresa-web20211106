# -*- coding: utf-8 -*-
"""Build the 遠離能量吸血鬼 YouTube Shorts (1080x1920, 30 fps).

Usage:
    python build.py            # all episodes
    python build.py 1 3        # selected episodes

Narration voice:
  * If ELEVENLABS_API_KEY is set, the ElevenLabs "Jessica" voice is used
    (model eleven_multilingual_v2). Override with ELEVENLABS_VOICE_ID.
  * Otherwise a local offline Mandarin female voice (sherpa-onnx,
    matcha-icefall-zh-baker) is used. Set TTS_MODEL_DIR to the folder that
    holds matcha-icefall-zh-baker/ and vocos.onnx.
  * To use a recording you already have for an episode, put it at
    voice/epNN.wav (or .mp3); the subtitles are then spread over it by
    character count.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

import music
from scripts import EPISODES

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SR = 44100
W, H, FPS = 1080, 1920, 30
LEAD_IN, TAIL = 0.8, 2.2
JESSICA = "cgSgspJ2msm6clMCkdW9"
SUB_MARGIN_V = 440   # bottom of the last subtitle line (px from bottom)
SUB_LINE_STEP = 112  # distance between subtitle lines (font 62px -> ~1.8x spacing)


# ---------------------------------------------------------------- TTS
def decode(path):
    """Decode any audio file to mono float32 at SR via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


class ElevenLabsTTS:
    def __init__(self, key):
        self.key = key
        self.voice = os.environ.get("ELEVENLABS_VOICE_ID", JESSICA)

    def __call__(self, text, prev="", nxt=""):
        body = json.dumps({
            "text": text, "model_id": "eleven_multilingual_v2",
            "previous_text": prev, "next_text": nxt,
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.8,
                               "style": 0.15, "use_speaker_boost": True, "speed": 0.92},
        }).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?output_format=mp3_44100_128",
            data=body, headers={"xi-api-key": self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r, tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(r.read())
            f.flush()
            return decode(f.name)


class LocalTTS:
    def __init__(self, model_dir):
        import opencc
        import sherpa_onnx
        d = Path(model_dir)
        m = d / "matcha-icefall-zh-baker"
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=str(m / "model-steps-3.onnx"), vocoder=str(d / "vocos.onnx"),
                    lexicon=str(m / "lexicon.txt"), tokens=str(m / "tokens.txt"),
                    dict_dir=str(m / "dict")),
                num_threads=4),
            rule_fsts=",".join(str(m / f) for f in ("phone.fst", "date.fst", "number.fst")))
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.t2s = opencc.OpenCC("t2s").convert  # model's lexicon is simplified

    def __call__(self, text, prev="", nxt=""):
        spoken = self.t2s(text).replace("「", "").replace("」", "")
        a = self.tts.generate(spoken, sid=0, speed=0.82)
        y = np.asarray(a.samples, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, y, a.sample_rate)
            y = decode(f.name)
        return y


def get_tts():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        print("Voice: ElevenLabs Jessica")
        return ElevenLabsTTS(key)
    print("Voice: local offline Mandarin female (sherpa-onnx baker)")
    return LocalTTS(os.environ.get("TTS_MODEL_DIR", HERE / "models"))


def trim(y, thr=0.01):
    idx = np.where(np.abs(y) > thr)[0]
    if len(idx) == 0:
        return y
    s = max(0, idx[0] - int(0.03 * SR))
    e = min(len(y), idx[-1] + int(0.08 * SR))
    return y[s:e]


def pause_after(line):
    return 0.85 if line.endswith(("。", "？", "！", "嗎？")) else 0.6


def narrate(ep, lines, tts):
    """Return (voice track, [(start, end, text), ...])."""
    for ext in ("wav", "mp3", "m4a"):
        rec = HERE / "voice" / f"ep{ep:02d}.{ext}"
        if rec.exists():
            print(f"  using recording {rec.name}")
            y = decode(rec)
            voice = np.concatenate([np.zeros(int(LEAD_IN * SR), np.float32), y])
            dur = len(y) / SR
            weights = np.array([len(l) + 4 for l in lines], float)
            bounds = np.concatenate([[0], np.cumsum(weights / weights.sum() * dur)]) + LEAD_IN
            return voice, [(bounds[i], bounds[i + 1] - 0.1, l) for i, l in enumerate(lines)]

    segs, cues, t = [np.zeros(int(LEAD_IN * SR), np.float32)], [], LEAD_IN
    for i, line in enumerate(lines):
        y = trim(tts(line, lines[i - 1] if i else "", lines[i + 1] if i + 1 < len(lines) else ""))
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
        d = len(y) / SR
        cues.append((t, t + d + 0.25, line))
        gap = pause_after(line)
        segs += [y, np.zeros(int(gap * SR), np.float32)]
        t += d + gap
    return np.concatenate(segs), cues


# ---------------------------------------------------------------- mixing
def mix(voice, ep):
    total = len(voice) + int(TAIL * SR)
    v = np.zeros(total, np.float32)
    v[: len(voice)] = voice
    bgm = music.compose(total / SR, variant=ep - 1, seed=ep * 11)[:total]
    # duck music under narration (smoothed voice envelope)
    env = np.abs(v)
    k = int(0.4 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    active = np.clip(env / 0.05, 0, 1)
    gain = 0.30 - 0.14 * active  # ~-10 dB alone, ~-16 dB under voice
    out = bgm * gain[:, None] + v[:, None] * 0.95
    return out.astype(np.float32)


# ---------------------------------------------------------------- subtitles
def wrap(text, width=13):
    """Wrap Chinese text into lines of <= width chars, preferring punctuation."""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= width:
            # break after the last punctuation if it's reasonably far in
            cut = max(cur.rfind(p) for p in "，：；、")
            if cut >= width // 2 and cut < len(cur) - 1:
                out.append(cur[: cut + 1])
                cur = cur[cut + 1:]
            else:
                out.append(cur)
                cur = ""
    if cur:
        if out and len(cur) <= 2:
            out[-1] += cur
        else:
            out.append(cur)
    # never start a line with closing punctuation
    for i in range(1, len(out)):
        while out[i] and out[i][0] in "，。？！：；、」":
            out[i - 1] += out[i][0]
            out[i] = out[i][1:]
    return [l for l in out if l]


def ts(sec):
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(cues, path):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,WenQuanYi Zen Hei,62,&H00243040,&H00243040,&H30E6F2F8,&H00000000,0,0,0,0,100,100,2,0,3,16,0,2,70,70,440,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = []
    for s, e, text in cues:
        # ASS has no line-spacing control, so each wrapped line is its own
        # event stacked upward from the bottom margin with a roomy gap.
        lines = wrap(text)
        for k, line in enumerate(lines):
            mv = SUB_MARGIN_V + (len(lines) - 1 - k) * SUB_LINE_STEP
            rows.append(f"Dialogue: 0,{ts(s)},{ts(e)},Sub,,0,0,{mv},,{{\\fad(180,180)}}{line}")
    path.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- video
def render(ep, info, audio, cues):
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "mix.wav"
        sf.write(wav, audio, SR)
        ass = tmp / "subs.ass"
        write_ass(cues, ass)
        (OUT / f"ep{ep:02d}.srt").write_text(to_srt(cues), encoding="utf-8")
        dur = len(audio) / SR
        frames = int(np.ceil(dur * FPS))
        img = HERE / info["image"]
        # Slow, smooth push-in (upscale first so zoompan doesn't jitter).
        vf = (f"scale={W*4}:{H*4}:flags=lanczos,setsar=1,"
              f"zoompan=z='1+0.07*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih*0.42-(ih*0.42/zoom)':"
              f"d={frames}:s={W}x{H}:fps={FPS},"
              f"subtitles={ass}:fontsdir=/usr/share/fonts/truetype/wqy,"
              f"fade=t=in:st=0:d=0.6,fade=t=out:st={dur-1.0:.2f}:d=1.0,format=yuv420p")
        out = OUT / f"遠離能量吸血鬼_第{ep}集.mp4"
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(img),
            "-i", str(wav), "-vf", vf, "-frames:v", str(frames),
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)],
            check=True)
        print(f"  -> {out.relative_to(HERE)} ({dur:.1f}s)")


def to_srt(cues):
    def t(sec):
        ms = int(round(sec * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    return "\n".join(f"{i}\n{t(s)} --> {t(e)}\n{text}\n" for i, (s, e, text) in enumerate(cues, 1))


def main():
    eps = [int(a) for a in sys.argv[1:]] or sorted(EPISODES)
    tts = get_tts()
    for ep in eps:
        info = EPISODES[ep]
        print(f"第{ep}集 {info['title']}")
        voice, cues = narrate(ep, info["lines"], tts)
        render(ep, info, mix(voice, ep), cues)


if __name__ == "__main__":
    main()
