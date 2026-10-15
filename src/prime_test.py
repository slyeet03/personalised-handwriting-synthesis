from test import draw

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import vocab
from model import SynthesisModel
from prime import sample_primed
from sample import sample
from train import get_device

PRIME_IDX = 0                       
TEXT = config.TEXT

# setup
device = get_device()
d = np.load('../personal_data/personal.npz', allow_pickle=True)
mean, std = d['mean'], d['std']
char_to_idx = vocab.build_char_to_idx(vocab.build_vocab(d['alphabet']))
prime_strokes, prime_text = d['strokes'][PRIME_IDX], str(d['texts'][PRIME_IDX])
print("priming with:", repr(prime_text))

model = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
model.load_state_dict(torch.load('../checkpoints/personal_best.pt', map_location=device))
model.eval()

# plot config
fig, axes = plt.subplots(4, 2, figsize=(12, 8))

for row in axes:
    # generate with normal sample for unprimed data
    pts, _ = sample(model, TEXT, char_to_idx, device, greedy=False, temp=0.3)
    draw(row[0], pts, mean, std)
    row[0].set_title("unprimed", fontsize=8)

    # generate with primed data
    pts, _ = sample_primed(model, prime_strokes, prime_text, TEXT, char_to_idx, device, temp=0.3)
    draw(row[1], pts, mean, std)
    row[1].set_title("primed", fontsize=8)

fig.tight_layout()
fig.savefig('../outputs/prime_compare.png', dpi=130)

