import matplotlib.pyplot as plt
import numpy as np
import torch

import vocab
from model import SynthesisModel
from sample import catmull_rom, sample

CKPT = '../checkpoints/personal_best.pt'
DATA = '../personal_data/personal.npz'
TEXT = "transition function"
TEMPS = [0.3]        
N = 8                     # samples per temperature
OUT = '../outputs'


def draw(ax, points, mean, std):
    raw = points[:, :2] * std[:2] + mean[:2]
    xy = np.cumsum(raw, axis=0)
    pen = points[:, 2]
    start = 0
    for i in range(len(pen)):
        if pen[i] == 1 or i == len(pen) - 1:
            s = xy[start:i+1]
            if len(s) == 1:
                ax.plot(s[0, 0], s[0, 1], 'k.', markersize=2)
            else:
                c = catmull_rom(s, 8)
                ax.plot(c[:, 0], c[:, 1], 'k', linewidth=1)
            start = i + 1
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

def main():
    device = torch.device('cpu')                    
    data = np.load(DATA, allow_pickle=True)
    mean, std = data['mean'], data['std']
    char_to_idx = vocab.build_char_to_idx(vocab.build_vocab(data['alphabet']))

    model = SynthesisModel().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    for temp in TEMPS:
        fig, axes = plt.subplots(4, 2, figsize=(12, 8))
        fig_a, axes_a = plt.subplots(4, 2, figsize=(12, 8))

        for ax, ax_a in zip(axes.flat, axes_a.flat):
            points, phis = sample(model, TEXT, char_to_idx, device,
                                  greedy=False, temp=temp)
            draw(ax, points, mean, std)
            ax.set_title(f"{len(points)} steps", fontsize=8)

            phis_n = phis / (phis.max(axis=1, keepdims=True) + 1e-8)
            ax_a.imshow(phis_n.T, aspect='auto')
            ax_a.set_title(f"{len(points)} steps", fontsize=8)

        fig.suptitle(f'"{TEXT}"  |  temp={temp}')
        fig_a.suptitle(f'attention  |  temp={temp}')
        fig.tight_layout(); fig_a.tight_layout()
        fig.savefig(f'{OUT}/grid_temp{temp}.png', dpi=130)
        fig_a.savefig(f'{OUT}/attn_temp{temp}.png', dpi=130)

    plt.show()


if __name__ == '__main__':
    main()
