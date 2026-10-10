# -*- coding: utf-8 -*-
"""Lo-fi cassette 'Happy Birthday' loop -> assets/hbd.mp3"""
import numpy as np, lameenc, math, os

SR = 32000
BPM = 92.0
SPB = 60.0 / BPM          # seconds per beat
BEATS = 36                # 9 bars
DUR = BEATS * SPB
N = int(DUR * SR)
buf = np.zeros(N + SR, dtype=np.float64)   # +1s tail for wrap

def add(sig, beat):
    i = int(beat * SPB * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[:j - i]

def wrap_tail():
    """wrap overhang back to the start so the loop is seamless"""
    if len(buf) > N:
        buf[:len(buf) - N] += buf[N:]
        return buf[:N]
    return buf

rng = np.random.default_rng(7)

# ---------- helpers ----------
def env(n, a, d, s, r, sus=None):
    """ADSR in samples"""
    sus = s if sus is None else sus
    tot = a + d + sus + r
    if tot < 1: tot = 1
    e = np.zeros(int(tot))
    a, d, sus, r = int(a), int(d), int(sus), int(r)
    if a: e[:a] = np.linspace(0, 1, a)
    if d: e[a:a + d] = np.linspace(1, 0.55, d)
    if sus: e[a + d:a + d + sus] = 0.55
    if r: e[a + d + sus:a + d + sus + r] = np.linspace(0.55, 0, r)
    out = np.zeros(n)
    m = min(n, len(e))
    out[:m] = e[:m]
    if n > len(e):
        out[len(e):] = 0
    return out

def lp(x, fc, taps=161):
    """FIR lowpass via windowed sinc"""
    fc = max(60.0, min(fc, SR / 2 - 500))
    t = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * fc * t / SR) * np.hanning(taps)
    h /= h.sum()
    return np.convolve(x, h, mode='same')

def hp(x, fc, taps=161):
    return x - lp(x, fc, taps)

def osc(freq, n, kind='sine', det=0.0, phase=0.0):
    t = np.arange(n) / SR
    f = freq * (1 + det)
    if kind == 'sine':
        return np.sin(2 * np.pi * f * t + phase)
    if kind == 'tri':
        return 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t + phase))
    if kind == 'saw':
        x = (f * t + phase / (2 * np.pi)) % 1.0
        return 2 * x - 1
    if kind == 'sq':
        return np.sign(np.sin(2 * np.pi * f * t + phase))
    return np.sin(2 * np.pi * f * t)

# ---------- instruments ----------
def kick():
    n = int(0.34 * SR)
    t = np.arange(n) / SR
    f = 118 * np.exp(-t * 26) + 44
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 8.5)
    s += 0.25 * np.exp(-t * 60) * rng.normal(0, 1, n) * np.exp(-t * 30)
    return lp(s, 2600) * 0.95

def snare():
    n = int(0.24 * SR)
    t = np.arange(n) / SR
    nz = rng.normal(0, 1, n)
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 26) * 0.35
    s = hp(nz * np.exp(-t * 19), 900) * 0.55 + body
    return lp(s, 7200) * 0.5

def hat(open_=False):
    n = int((0.16 if open_ else 0.055) * SR)
    t = np.arange(n) / SR
    nz = rng.normal(0, 1, n)
    s = hp(nz, 7000) * np.exp(-t * (34 if open_ else 90))
    return s * 0.16

def bass(freq, dur, gain=0.5):
    n = int(dur * SR)
    s = osc(freq, n, 'tri') * 0.7 + osc(freq, n, 'sine') * 0.9
    s = lp(s, 420)
    s *= env(n, 0.008 * SR, 0.06 * SR, n - 0.16 * SR, 0.09 * SR)
    return s * gain

def rhodes(freq, dur, gain=0.18):
    """electric-piano-ish: fundamental + bell partials"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for mult, amp, dec in ((1, 1.0, 3.2), (2, 0.42, 5.0), (3, 0.16, 7.0), (4.02, 0.09, 9.0)):
        s += amp * np.sin(2 * np.pi * freq * mult * t + 0.6) * np.exp(-t * dec)
    s *= env(n, 0.004 * SR, 0.10 * SR, n - 0.30 * SR, 0.20 * SR)
    s = lp(s, 2600)
    return s * gain

def lead(freq, dur, gain=0.30):
    """warm tape-ish lead"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    s = osc(freq * vib, n, 'tri') * 0.55 + osc(freq * vib, n, 'sine', det=0.003) * 0.7
    s += 0.10 * osc(freq * 2 * vib, n, 'sine')
    s = lp(s, 2300)
    s *= env(n, 0.030 * SR, 0.09 * SR, n - 0.34 * SR, 0.22 * SR)
    return s * gain

# ---------- notes / chords ----------
NOTE = {}
names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
for oct_ in range(1, 7):
    for i, nm in enumerate(names):
        NOTE[nm + str(oct_)] = 440.0 * 2 ** (((oct_ - 4) * 12 + i - 9) / 12.0)

CHORDS = {
    'Cmaj7': ['C3','E3','G3','B3'],
    'Fmaj7': ['F2','A2','C3','E3'],
    'G7':    ['G2','B2','D3','F3'],
    'Am7':   ['A2','C3','E3','G3'],
    'Dm7':   ['D3','F3','A3','C4'],
    'C7':    ['C3','E3','G3','A#3'],
    'Gm7':   ['G2','A#2','D3','F3'],
}
BASS_ROOT = {'Cmaj7':'C2','Fmaj7':'F1','G7':'G1','Am7':'A1','Dm7':'D2','C7':'C2','Gm7':'G1'}

# chord timeline over 36 beats
PROG = [
    (0,  'Cmaj7'), (6,  'Fmaj7'), (8,  'Cmaj7'),
    (12, 'Cmaj7'), (16, 'G7'),    (18, 'Fmaj7'),
    (20, 'Cmaj7'), (22, 'Fmaj7'), (24, 'Dm7'),
    (26, 'C7'),    (28, 'Fmaj7'), (30, 'G7'), (32, 'Fmaj7'), (34, 'Cmaj7'),
]
def chord_at(b):
    cur = PROG[0][1]
    for t, c in PROG:
        if t <= b: cur = c
        else: break
    return cur

# melody: (beat, note, dur_beats)
MELODY = [
    (7.0,'C4',0.5), (7.5,'C4',0.5), (8.0,'D4',1.0), (9.0,'C4',1.0), (10.0,'F4',1.0), (11.0,'E4',2.0),
    (13.0,'C4',0.5),(13.5,'C4',0.5),(14.0,'D4',1.0),(15.0,'C4',1.0),(16.0,'G4',1.0),(17.0,'F4',2.0),
    (19.0,'C4',0.5),(19.5,'C4',0.5),(20.0,'C5',1.0),(21.0,'A4',1.0),(22.0,'F4',1.0),(23.0,'E4',1.0),(24.0,'D4',2.0),
    (26.0,'A#4',0.5),(26.5,'A#4',0.5),(27.0,'A4',1.0),(28.0,'F4',1.0),(29.0,'G4',1.0),(30.0,'F4',2.0),
]

# ---------- arrange ----------
# drums (lo-fi, swung eighths)
for bar in range(9):
    b0 = bar * 4
    add(kick(), b0)
    add(kick(), b0 + 2.5)
    add(snare(), b0 + 1)
    add(snare(), b0 + 3)
    for e in range(8):
        t = b0 + e * 0.5 + (0.06 if e % 2 else 0.0)   # swing
        add(hat(open_=(e == 7)), t)

# bass + rhodes
for bar in range(9):
    for off in (0.0, 2.5):
        b = bar * 4 + off
        ch = chord_at(b)
        add(bass(NOTE[BASS_ROOT[ch]], SPB * 1.6, 0.55), b)
    for off in (0.0, 1.5, 2.5):
        b = bar * 4 + off
        ch = chord_at(b)
        dur = SPB * (1.1 if off else 1.4)
        for nt in CHORDS[ch]:
            add(rhodes(NOTE[nt], dur, 0.085), b + 0.02 * rng.random())

# melody
for b, nt, d in MELODY:
    add(lead(NOTE[nt], SPB * d * 0.98, 0.30), b + 0.01)

# ---------- texture: tape hiss + vinyl crackle ----------
hiss = rng.normal(0, 1, N) * 0.010
hiss = lp(hiss, 6500)
buf[:N] += hiss

crackle = np.zeros(N)
for _ in range(int(N / SR * 46)):
    i = rng.integers(0, N - 400)
    crackle[i:i + 120] += rng.normal(0, 1, 120) * np.exp(-np.arange(120) / 14.0) * rng.uniform(0.05, 0.5)
buf[:N] += crackle

# wow & flutter (tape speed wobble) -- tiny pitch drift applied as resample
def wow(x):
    n = len(x)
    t = np.arange(n) / SR
    drift = 1 + 0.0016 * np.sin(2 * np.pi * 0.72 * t) + 0.0007 * np.sin(2 * np.pi * 4.3 * t + 1.1)
    idx = np.clip(np.cumsum(drift) - 1, 0, n - 1)
    i0 = idx.astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    fr = idx - i0
    return x[i0] * (1 - fr) + x[i1] * fr

y = wrap_tail()
y = wow(y)

# tape saturation + gentle bus compression + master lowpass (cassette bandwidth)
y = np.tanh(y * 1.35) * 0.72
y = lp(y, 8200)
# soft knee compressor
thr, ratio = 0.45, 2.6
a = np.abs(y)
over = a > thr
y[over] = np.sign(y[over]) * (thr + (a[over] - thr) / ratio)

# tiny crossfade at the loop point
xf = int(0.030 * SR)
y[:xf] *= np.linspace(0, 1, xf)
y[-xf:] *= np.linspace(1, 0, xf)
y[:xf] += np.concatenate([np.zeros(0), ]) if False else 0

y = y / max(1e-6, np.abs(y).max()) * 0.89

pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
enc = lameenc.Encoder()
enc.set_bit_rate(112)
enc.set_in_sample_rate(SR)
enc.set_channels(1)
enc.set_quality(2)
mp3 = enc.encode(pcm.tobytes())
mp3 += enc.flush()
os.makedirs('assets', exist_ok=True)
open('assets/hbd.mp3', 'wb').write(mp3)
print('wrote assets/hbd.mp3  %.2fs  %d bytes  peak %.2f' % (len(pcm) / SR, len(mp3), np.abs(y).max()))
