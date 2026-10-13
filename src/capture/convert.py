import json
import sys

import numpy as np

sys.path.insert(0, '..')
from sample import plot_strokes

DW = '../../dataset/deepwriting_training.npz'
LINES = '../../personal_data/lines.jsonl'
OUT = '../../personal_data/personal.npz'
DT = 38     

dw = np.load(DW, allow_pickle=True)
mean, std = dw['mean'], dw['std']

ppc, wpc = [], []
for s, t in zip(dw['strokes'], dw['texts']):
    n = len(str(t))
    
    # less than 10 is very noisy
    if n < 10 or len(s) > 400:
        continue

    xy = np.cumsum(s[:, :2] * std[:2] + mean[:2], axis=0)
    ppc.append(len(s) / n)
    wpc.append((xy[:, 0].max() - xy[:, 0].min()) / n)

TARGET_PPC, TARGET_WPC = np.median(ppc), np.median(wpc)
print(f"DeepWriting (<=400 pts): {TARGET_PPC:.1f} points/char "
      f"(25-75%: {np.percentile(ppc, 25):.1f}-{np.percentile(ppc, 75):.1f}), "
      f"{TARGET_WPC:.5f} width/char")

recs = [json.loads(l) for l in open(LINES) if l.strip()]
# removing writing that comes from a mouse and not pen
recs = [r for r in recs if r.get('pointer_type')=='pen']


def resample(stroke, dt):
    a = np.array(stroke, dtype=float)            # columns: x, y, pressure, t
    # guard against repeated timestamps
    t = np.maximum.accumulate(a[:, 3])
    # short strokes like dots would fuck up the intervals so first and last point are kept
    if len(a) < 2 or t[-1] - t[0] < dt:
        return a[[0, -1], :3] if len(a) > 1 else a[:, :3]

    tn = np.append(np.arange(t[0], t[-1], dt), t[-1])
    return np.stack([np.interp(tn, t, a[:, k]) for k in range(3)], axis=1)


if DT is None:
    for dt in [20, 25, 30, 35, 40, 50]:
        vals = [sum(len(resample(s, dt)) for s in r['strokes']) / len(r['text'])
                for r in recs]
        print(f"dt={dt:>2} ms -> {np.median(vals):.1f} points/char")
    print("set DT to the value closest to the DeepWriting number, then rerun")
    sys.exit()


user_wpc = []
for r in recs:
    xs = np.concatenate([resample(s, DT)[:, 0] for s in r['strokes']])
    user_wpc.append((xs.max() - xs.min()) / len(r['text']))

SCALE = TARGET_WPC / np.median(user_wpc)
print(f"scale factor: {SCALE:.6f}")


def obj(lst):
    a = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        a[i] = x
    return a

S, T, P = [], [], []
for r in recs:
    pts, pen, pres = [], [], []
    for s in r['strokes']:
        a = resample(s, DT)
        for i, p in enumerate(a):
            pts.append(p[:2] * SCALE)
            pres.append(p[2])
            pen.append(1.0 if i == len(a) - 1 else 0.0)

    pts = np.array(pts)
    off = np.diff(pts, axis=0, prepend=pts[:1])           # first offset = 0
    off = (off - mean[:2]) / std[:2]

    S.append(np.concatenate([off, np.array(pen)[:, None]], axis=1).astype(np.float32))
    T.append(r['text'])
    P.append(np.array(pres, dtype=np.float32))

np.savez(OUT, strokes=obj(S), texts=np.array(T), pressure=obj(P),
         mean=mean, std=std, alphabet=dw['alphabet'])


allp = np.concatenate(S)
print(f"{len(S)} lines, {len(allp)} points")
print("offset mean (want ~0):", allp[:, :2].mean(0))
print("offset std  (want ~1):", allp[:, :2].std(0))
print(f"pen fraction: {allp[:, 2].mean():.3f} (DeepWriting 0.042)")
print("points/char:", np.median([len(s) / len(t) for s, t in zip(S, T)]))

for i in range(min(3, len(S))):
    plot_strokes(S[i], mean, std, save_path=f'../../outputs/personal_check_{i}.png')
print("plots saved to outputs/personal_check_*.png")
