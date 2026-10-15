import random
from tkinter import CHAR

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

import config
import vocab
from layout import generate_best, wrap
from model import SynthesisModel
from sample import catmull_rom, smooth_stroke
from train import get_device

# A4 at 96 px per inch
PAGE_W, PAGE_H = 794, 1123
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 70, 100, 90
CHAR_PX = 18                      # average width of one character
TARGET_WPC = 0.00963              # DeepWriting width per character (from convert.py)
LINE_GAP = 3*CHAR_PX                     # distance between lines
SCALE = CHAR_PX / TARGET_WPC
W_MAX, W_MIN = 2.0, 1.2           # pen width in px (slow = thick, fast = thin)
SPACE_PX = 1.1 * CHAR_PX
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.,-()/ ")
REPL = {'’': "'", '‘': "'", '“': '', '”': '', '"': '', ';': ',',
        ':': '.', '!': '.', '?': '.', '–': '-', '—': '-', '\t': ' '}

# for each character check with REPL and use replacement if there is one
def clean(text):
    text = ''.join(REPL.get(c, c) for c in text)
    # after replacement, if any character still not allowed exclude them
    dropped = sorted({c for c in text if c not in ALLOWED and c != '\n'})
    return ''.join(c for c in text if c in ALLOWED or c == '\n'), dropped

def load(ckpt='../checkpoints/personal_best.pt', data='../personal_data/personal.npz'):
    device = get_device()
    d = np.load(data, allow_pickle=True)
    c2i = vocab.build_char_to_idx(vocab.build_vocab(d['alphabet']))
    model = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, d, c2i, device

# model output to page coordinates
def to_strokes(pts, mean, std):
    xy = np.cumsum(pts[:, :2] * std[:2] + mean[:2], axis=0) * SCALE
    xy[:, 0] -= xy[:, 0].min()                  # line starts at x = 0
    xy[:, 1] -= np.median(xy[:, 1])             # baseline at y = 0
    pen, strokes, start = pts[:, 2], [], 0

    for j in range(len(pen)):
        if pen[j] == 1 or j == len(pen) - 1:
            strokes.append(xy[start:j + 1])
            start = j + 1

    return strokes

def generate_lines(text, model, d, c2i, device, prime_idx, tries=6, temp=0.25,max_chars=24):
    text, dropped = clean(text)
    prime_s, prime_t = d['strokes'][prime_idx], str(d['texts'][prime_idx])
    mean, std = d['mean'], d['std']

    # paragraphs are split as new lines
    lines = []
    for para in text.split('\n'):
        lines.extend(wrap(para,max_chars) if para.strip() else [''])

    items, failed = [], []
    for line in lines:
        # empty line becomes None so a blank row is stores as blank
        if not line:
            items.append(None)                  
            continue

        pts = generate_best(model, prime_s, prime_t, line, c2i, device, tries=tries, temp=temp)
        # if best is not generated temp is lowered and retried
        if pts is None:                         
            pts = generate_best(model, prime_s, prime_t, line, c2i, device, tries=tries, temp=temp * 0.7)
        # if retry failed then it is recorded as a failed line and it leaves a blank row
        if pts is None:
            failed.append(line)
            items.append(None)
            continue

        items.append(to_strokes(pts, mean, std))
    
    return items, dropped, failed


def pack(items):
    usable = PAGE_W - 2 * MARGIN_X
    rows, cur, x = [], [], 0.0
    for s in items:
        # paragraph break
        if s is None:                          
            if cur:
                rows.append(cur)

            # blank row
            rows.append([])                    
            cur, x = [], 0.0
            continue
        w = max(st[:, 0].max() for st in s)
        if w > usable:                         
            s = [st * (usable / w) for st in s]
            w = usable
        if cur and x + SPACE_PX + w > usable:  
            rows.append(cur)
            cur, x = [], 0.0
        off = x + (SPACE_PX if cur else 0.0)
        cur.append((s, off))
        x = off + w
    if cur:
        rows.append(cur)
    return rows


def place_row(row, r, rng):
    ang = np.radians(rng.uniform(-0.4, 0.4))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    origin = np.array([MARGIN_X, MARGIN_TOP + r * LINE_GAP + rng.uniform(-2, 2)])
    out = []
    for strokes, off in row:
        dy = rng.uniform(-1.5, 1.5)            
        for s in strokes:
            out.append((s + [off, dy]) @ R.T + origin)
    return out

def to_segments(strokes):
    segs, lens = [], []
    for s in strokes:
        # dot
        if len(s) == 1:                                      
            segs.append(np.array([s[0], s[0] + [0.8, 0]]))
            lens.append(0.0)
            continue

        c = catmull_rom(smooth_stroke(s, 1), 8)
        segs.extend(np.stack([c[:-1], c[1:]], axis=1))
        lens.extend(np.linalg.norm(np.diff(c, axis=0), axis=1))

    return segs, lens


def render(items, out_prefix, seed=0, dpi=200):
    rng = random.Random(seed)
    rows = pack(items)
    per_page = (PAGE_H - MARGIN_TOP - MARGIN_BOTTOM) // LINE_GAP

    pages, all_lens = {}, []
    for r, row in enumerate(rows):
        if not row:
            continue

        segs, lens = to_segments(place_row(row, r % per_page, rng))
        pages.setdefault(r // per_page, []).append((segs, lens))
        all_lens.extend(lens)
    
    if not all_lens:
        return []

    # fast
    ref = np.percentile([l for l in all_lens if l > 0], 90)

    paths = []
    for pg in range(max(len(rows) - 1, 0) // per_page + 1):
        fig = plt.figure(figsize=(PAGE_W / 96, PAGE_H / 96))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, PAGE_W) 
        ax.set_ylim(PAGE_H, 0); ax.axis('off')

        for segs, lens in pages.get(pg, []):
            w = W_MAX - (W_MAX - W_MIN) * np.clip(np.array(lens) / ref, 0, 1)
            ax.add_collection(LineCollection(segs, linewidths=w * 0.75, colors='black', capstyle='round'))
        
        for ext in ('png', 'svg'):
            path = f'{out_prefix}_{pg + 1}.{ext}'
            fig.savefig(path, dpi=dpi)
            paths.append(path)
        plt.close(fig)

    return paths


if __name__ == '__main__':
    TEXT = config.TEXT
    model, d, c2i, device = load()
    prime_ids = [i for i, t in enumerate(d['texts']) if 8 <= len(str(t)) <= 14]
    items, dropped, failed = generate_lines(TEXT, model, d, c2i, device, prime_ids[0],max_chars=14)
    print('characters dropped:', dropped, '| lines that failed:', failed)
    print(render(items, '../outputs/page'))
