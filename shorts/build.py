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
SUB_LINE_STEP = 106  # distance between Chinese subtitle lines (font 56px -> ~1.8x spacing)
EN_LINE_STEP = 66    # distance between English subtitle lines (font 38px)
EN_GAP = 34          # extra space between the English block and the Chinese block


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
    if line.endswith(("，", "：")):
        return 0.35  # continuation of a sentence split for subtitles
    return 0.85 if line.endswith(("。", "？", "！", "嗎？")) else 0.6


def find_pauses(y, thr_db=-38, min_len=0.12):
    """Return [(start, end)] of silent stretches in a mono SR signal."""
    hop = int(0.01 * SR)
    frames = len(y) // hop
    rms = np.sqrt(np.mean(y[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    quiet = 20 * np.log10(rms) < thr_db
    out, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.01 >= min_len:
                out.append((start * 0.01, i * 0.01))
            start = None
    return out


def align_recording(y, lines):
    """Map script lines onto a recording: pick one pause per line boundary
    (dynamic programming) so each line's duration best matches its length,
    favouring long pauses. Returns [(start, end)] in seconds."""
    total = len(y) / SR
    pauses = find_pauses(y)
    speech_start = pauses[0][1] if pauses and pauses[0][0] < 0.05 else 0.0
    speech_end = pauses[-1][0] if pauses and pauses[-1][1] > total - 0.05 else total
    inner = [p for p in pauses if speech_start < p[0] and p[1] < speech_end]
    n, m = len(lines), len(inner)
    if m < n - 1:  # not enough pauses: fall back to proportional split
        w = np.array([len(l) for l in lines], float)
        b = speech_start + np.concatenate([[0], np.cumsum(w / w.sum())]) * (speech_end - speech_start)
        return [(b[i], b[i + 1] - 0.1) for i in range(n)]
    chars = np.array([len(l) for l in lines], float)
    rate = (speech_end - speech_start) / chars.sum()
    # node k: boundary at inner pause k (k = -1 is speech start, m is speech end)
    def pos_end(k):
        return speech_end if k == m else inner[k][0]

    def pos_start(k):
        return speech_start if k == -1 else inner[k][1]

    def cost(i, a, b):
        dur = pos_end(b) - pos_start(a)
        exp = chars[i] * rate
        c = (dur - exp) ** 2 / exp
        if b < m:
            c -= 4.0 * (inner[b][1] - inner[b][0])  # prefer long pauses
        return c

    INF = float("inf")
    best = {(-1, 0): (0.0, None)}  # (boundary idx, lines done) -> (cost, prev)
    for i in range(n):
        last = i == n - 1
        for (a, done), (c0, _) in list(best.items()):
            if done != i:
                continue
            targets = [m] if last else range(a + 1, m)
            for b in targets:
                c = c0 + cost(i, a, b)
                key = (b, i + 1)
                if c < best.get(key, (INF, None))[0]:
                    best[key] = (c, (a, done))
    key, path = (m, n), []
    while key[1] > 0:
        prev = best[key][1]
        path.append((pos_start(prev[0]), pos_end(key[0])))
        key = prev
    return path[::-1]


def narrate(ep, pairs, tts):
    """Return (voice track, [(start, end, zh, en), ...])."""
    lines = [zh for zh, _ in pairs]
    for ext in ("wav", "mp3", "m4a"):
        rec = HERE / "voice" / f"ep{ep:02d}.{ext}"
        if rec.exists():
            print(f"  using recording {rec.name}")
            y = decode(rec)
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
            voice = np.concatenate([np.zeros(int(LEAD_IN * SR), np.float32), y])
            spans = align_recording(y, lines)
            return voice, [(LEAD_IN + a, LEAD_IN + b + 0.25, l, en)
                           for (a, b), (l, en) in zip(spans, pairs)]

    segs, cues, t = [np.zeros(int(LEAD_IN * SR), np.float32)], [], LEAD_IN
    for i, line in enumerate(lines):
        y = trim(tts(line, lines[i - 1] if i else "", lines[i + 1] if i + 1 < len(lines) else ""))
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
        d = len(y) / SR
        cues.append((t, t + d + 0.25, line, pairs[i][1]))
        gap = pause_after(line)
        segs += [y, np.zeros(int(gap * SR), np.float32)]
        t += d + gap
    return np.concatenate(segs), cues


# ---------------------------------------------------------------- mixing
def load_bgm(ep, total):
    """Use music/epNN.* or music/bgm.* if present: trimmed, looped if short,
    with fade-in/out. Returns stereo float32 or None."""
    for name in (f"ep{ep:02d}", "bgm"):
        for f in sorted((HERE / "music").glob(name + ".*")):
            raw = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(f), "-map", "0:a:0", "-ac", "2",
                 "-ar", str(SR), "-f", "f32le", "-"], check=True, capture_output=True).stdout
            y = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
            print(f"  music: {f.name}")
            reps = int(np.ceil(total / len(y)))
            y = np.tile(y, (reps, 1))[:total].copy()
            n_in, n_out = int(1.5 * SR), int(3.5 * SR)
            y[:n_in] *= np.linspace(0, 1, n_in)[:, None]
            y[-n_out:] *= np.linspace(1, 0, n_out)[:, None] ** 2
            return y
    return None


def mix(voice, ep):
    total = len(voice) + int(TAIL * SR)
    v = np.zeros(total, np.float32)
    v[: len(voice)] = voice
    bgm = load_bgm(ep, total)
    if bgm is None:
        bgm = music.compose(total / SR, variant=ep - 1, seed=ep * 11)[:total]
    if len(bgm) < total:  # guard against off-by-one from float rounding
        bgm = np.pad(bgm, ((0, total - len(bgm)), (0, 0)))
    # match loudness of supplied tracks to the synthesized default
    bgm = bgm / (np.sqrt(np.mean(bgm ** 2)) + 1e-9) * 0.2
    # duck music under narration (smoothed voice envelope, moving average)
    k = int(0.4 * SR)
    c = np.concatenate([[0], np.cumsum(np.abs(v), dtype=np.float64)])
    env = np.empty(total)
    lo = np.clip(np.arange(total) - k // 2, 0, total)
    hi = np.clip(np.arange(total) + k // 2, 0, total)
    env[:] = (c[hi] - c[lo]) / k
    active = np.clip(env / 0.05, 0, 1)
    gain = 0.20 - 0.12 * active  # music ~-9 dB vs speech when alone, ~-17 dB under voice
    out = bgm * gain[:, None] + v[:, None] * 0.95
    return out.astype(np.float32)


# ---------------------------------------------------------------- subtitles
def wrap(text, width=15, stretch=17):
    """Wrap Chinese text, breaking after punctuation whenever possible.

    A line may stretch to `stretch` chars to end on punctuation; otherwise
    it is hard-cut at `width`.
    """
    out, rest = [], text
    while len(rest) > width:
        cuts = [i + 1 for i, ch in enumerate(rest[:stretch]) if ch in "，、：；。？！"]
        cuts = [c for c in cuts if c < len(rest)]
        cut = max(cuts) if cuts else width
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        if out and len(rest) <= 2:
            out[-1] += rest
        else:
            out.append(rest)
    return out


def wrap_en(text, width=46):
    """Wrap English into the fewest lines, with line lengths balanced."""
    import textwrap
    n = len(textwrap.wrap(text, width))
    for w in range(20, width + 1):
        lines = textwrap.wrap(text, w)
        if len(lines) == n:
            return lines
    return [text]


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
Style: Sub,WenQuanYi Zen Hei,56,&H00243040,&H00243040,&H30E6F2F8,&H00000000,0,0,0,0,100,100,1,0,3,16,0,2,40,40,440,1
Style: En,Bitstream Charter,38,&H00304A5C,&H00304A5C,&H30E6F2F8,&H00000000,0,1,0,0,100,100,0.5,0,3,12,0,2,70,70,440,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = []
    for s, e, text, en in cues:
        # ASS has no line-spacing control, so each wrapped line is its own
        # event stacked upward from the bottom margin with a roomy gap.
        # English sits at the bottom, Chinese stacked above it.
        en_lines = wrap_en(en)
        for k, line in enumerate(en_lines):
            mv = SUB_MARGIN_V + (len(en_lines) - 1 - k) * EN_LINE_STEP
            rows.append(f"Dialogue: 0,{ts(s)},{ts(e)},En,,0,0,{mv},,{{\\fad(180,180)}}{line}")
        base = SUB_MARGIN_V + len(en_lines) * EN_LINE_STEP + EN_GAP
        lines = wrap(text)
        for k, line in enumerate(lines):
            mv = base + (len(lines) - 1 - k) * SUB_LINE_STEP
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
    return "\n".join(f"{i}\n{t(s)} --> {t(e)}\n{text}\n{en}\n" for i, (s, e, text, en) in enumerate(cues, 1))


def main():
    eps = [int(a) for a in sys.argv[1:]] or sorted(EPISODES)
    tts = get_tts()
    for ep in eps:
        info = EPISODES[ep]
        print(f"第{ep}集 {info['title']}")
        # Each short opens with the episode title read aloud. The cover image
        # already shows it, so that cue gets no subtitle.
        num = "一二三四五六七八九十"[ep - 1]
        intro = (f"第{num}集。" + info["title"].rstrip("？") + ("？" if info["title"].endswith("？") else "。"), "")
        voice, cues = narrate(ep, [intro] + info["lines"], tts)
        render(ep, info, mix(voice, ep), cues[1:])


if __name__ == "__main__":
    main()
