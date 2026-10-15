import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import vocab
from layout import generate_best, wrap
from model import SynthesisModel
from sample import catmull_rom
from train import get_device

TEXT = config.TEXT 
LINE_H = config.LINE_H               # vertical spacing between lines, in raw units
TRIES = config.TRIES

# setup
device = get_device()
d = np.load('../personal_data/personal.npz', allow_pickle=True)
mean, std = d['mean'], d['std']
char_to_idx = vocab.build_char_to_idx(vocab.build_vocab(d['alphabet']))

model = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
model.load_state_dict(torch.load('../checkpoints/personal_best.pt', map_location=device))
model.eval()

# indexes of priming candidates
prime_ids = [i for i, t in enumerate(d['texts']) if 8 <= len(str(t)) <= 14]

lines = wrap(TEXT)
print(lines)

fig, ax = plt.subplots(figsize=(12, 1.2 * len(lines) + 1))
for i, line in enumerate(lines):
    # different prime for each line
    p = random.choice(prime_ids)                       
    pts = generate_best(model, d['strokes'][p], str(d['texts'][p]), line,
                        char_to_idx, device, tries=TRIES)
    if pts is None:
        print(f"FAILED: {line!r}")
        continue

    # model output to page coordinates
    raw = pts[:, :2] * std[:2] + mean[:2]              # undo normalization
    xy = np.cumsum(raw, axis=0)                        # offsets -> positions
    # coordinates before were continuing from prime so jumping that to follow the page
    xy[:, 0] -= xy[:, 0].min()                         # left align every line
    xy[:, 1] += i * LINE_H - np.median(xy[:, 1])       # put the line on its own row

    pen = pts[:, 2]
    start = 0
    for j in range(len(pen)):
        # pen 1 means stroke ends
        if pen[j] == 1 or j == len(pen) - 1:
            s = xy[start:j + 1]
            # if its a dot then its normally drawn as a dot
            if len(s) == 1:
                ax.plot(s[0, 0], s[0, 1], 'k.', markersize=2)
            # else catmull rom to smooth it out
            else:
                c = catmull_rom(s, 8)
                ax.plot(c[:, 0], c[:, 1], 'k', linewidth=1.2, solid_capstyle='round')
            start = j + 1
    print(f"done: {line!r}")

ax.invert_yaxis()
ax.set_aspect('equal')
ax.axis('off')
fig.savefig('../outputs/layout_test.png', dpi=150, bbox_inches='tight')
print("saved ../outputs/layout_test.png")
