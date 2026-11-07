# -*- coding: utf-8 -*-
"""Synthesize soft, light orchestral background music (original, royalty-free).

Layers: string-ensemble pad, cello bass, harp arpeggios and a gentle
flute-like melody, all through a hall reverb. Every piece is generated
from scratch with numpy, so there are no licensing concerns.
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100

# Chord progressions as scale degrees in a major key (semitones from tonic).
PROGRESSIONS = [
    [[0, 4, 7], [-3, 0, 4], [-7, -3, 0], [-5, -1, 2]],          # I  vi IV V
    [[0, 4, 7], [-5, -1, 2], [-3, 0, 4], [-7, -3, 0]],          # I  V  vi IV
    [[-7, -3, 0], [0, 4, 7], [-5, -1, 2], [-3, 0, 4]],          # IV I  V  vi
    [[-3, 0, 4], [-7, -3, 0], [0, 4, 7], [-5, -1, 2]],          # vi IV I  V
]
KEYS = [53, 55, 50, 52]  # MIDI tonic: F3, G3, D3, E3


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def envelope(n, attack, release):
    env = np.ones(n)
    a = min(int(attack * SR), n // 2)
    r = min(int(release * SR), n // 2)
    if a:
        env[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if r:
        env[-r:] = np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return env


def lowpass(x, cutoff):
    return sosfilt(butter(2, cutoff, fs=SR, output="sos"), x)


def string_voice(freq, dur, rng):
    """Detuned band-limited saw ensemble with vibrato -> soft strings."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for _ in range(5):
        det = freq * (1 + rng.uniform(-0.004, 0.004))
        vib = 1 + 0.0025 * np.sin(2 * np.pi * rng.uniform(4.5, 5.5) * t + rng.uniform(0, 6))
        phase = 2 * np.pi * np.cumsum(det * vib) / SR
        for h in range(1, 9):
            out += np.sin(h * phase + rng.uniform(0, 6)) / h
    return out / 5


def harp_note(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * freq * t)
            + 0.35 * np.sin(4 * np.pi * freq * t) * np.exp(-t * 3)
            + 0.12 * np.sin(6 * np.pi * freq * t) * np.exp(-t * 5))
    return tone * np.exp(-t * 1.6) * envelope(n, 0.004, 0.05)


def flute_note(freq, dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * t) * np.clip(t / 0.6, 0, 1)
    phase = 2 * np.pi * np.cumsum(freq * vib) / SR
    tone = np.sin(phase) + 0.18 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    breath = lowpass(rng.standard_normal(n), 3000) * 0.03
    return (tone + breath) * envelope(n, 0.35, 0.6)


def reverb(x, seconds=3.2, wet=0.38, seed=7):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    ir = lowpass(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    y = fftconvolve(x, ir)[: len(x)]
    return (1 - wet) * x + wet * y * 0.6


def place(buf, sig, start):
    s = int(start * SR)
    e = min(len(buf), s + len(sig))
    if s < len(buf):
        buf[s:e] += sig[: e - s]


def compose(duration, variant=0, seed=1):
    rng = np.random.default_rng(seed)
    prog = PROGRESSIONS[variant % len(PROGRESSIONS)]
    key = KEYS[variant % len(KEYS)]
    chord_len = 4.0  # seconds per chord (~60 bpm, 4 beats)
    total = int((duration + 4) * SR)
    strings = np.zeros(total)
    bass = np.zeros(total)
    harp = np.zeros(total)
    flute = np.zeros(total)

    n_chords = int(np.ceil(duration / chord_len)) + 1
    for c in range(n_chords):
        start = c * chord_len
        chord = prog[c % len(prog)]
        dur = chord_len + 1.2  # overlap for legato
        # string pad (voiced around key + octave)
        for iv in chord:
            sig = string_voice(midi_hz(key + 12 + iv), dur, rng)
            place(strings, sig * envelope(len(sig), 1.2, 1.2), start)
        # cello bass, root an octave below
        root = chord[0] if chord[0] >= -5 else chord[0] + 12
        sig = string_voice(midi_hz(key - 12 + root), dur, rng)
        place(bass, sig * envelope(len(sig), 0.8, 1.2), start)
        # harp arpeggio: 8 eighth notes up and down
        if c >= 1:
            notes = [chord[0], chord[1], chord[2], chord[0] + 12,
                     chord[2] + 12, chord[0] + 12, chord[2], chord[1]]
            for k, iv in enumerate(notes):
                place(harp, harp_note(midi_hz(key + 12 + iv), 2.2), start + k * 0.5)
        # flute melody from the 3rd chord on, mostly chord tones
        if c >= 2 and c % 2 == 0:
            tones = [chord[2] + 12, chord[1] + 12, chord[0] + 12]
            pattern = rng.choice(3, size=3)
            lens = [1.5, 1.0, 1.5]
            t0 = start
            for p, ln in zip(pattern, lens):
                place(flute, flute_note(midi_hz(key + 12 + tones[p]), ln + 0.3, rng), t0)
                t0 += ln

    strings = lowpass(strings, 2200)
    bass = lowpass(bass, 600)
    mix = 0.30 * strings / (np.max(np.abs(strings)) + 1e-9) \
        + 0.22 * bass / (np.max(np.abs(bass)) + 1e-9) \
        + 0.22 * harp / (np.max(np.abs(harp)) + 1e-9) \
        + 0.16 * flute / (np.max(np.abs(flute)) + 1e-9)
    mix = reverb(mix)
    mix = mix[: int(duration * SR)]
    mix *= envelope(len(mix), 2.5, 3.5)
    mix /= np.max(np.abs(mix)) + 1e-9
    stereo = np.stack([mix, np.roll(mix, int(0.012 * SR))], axis=1) * 0.9
    return stereo.astype(np.float32)


if __name__ == "__main__":
    import sys
    import soundfile as sf
    sf.write(sys.argv[1], compose(float(sys.argv[2]) if len(sys.argv) > 2 else 30), SR)
