import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import Dataset, collate_fn, split_indices
from layout import generate_best, wrap
from model import SynthesisModel
from prime import sample_primed
from train import evaluate, get_device

PERSONAL = '../personal_data/personal.npz'
BASE = os.path.join(config.CKPT_DIR, 'base_pretrained.pt')
FT = os.path.join(config.CKPT_DIR, 'personal_best.pt')
OUT = '../outputs/metrics.json'

CHUNK = 14          # characters per generated chunk, as in the app
N_CHUNKS = 40       # chunks used for the generation statistics
N_BEST = 20         # chunks used for the best-of-N failure rate
TEMP = 0.25
TRIES = 6

np.random.seed(0)
torch.manual_seed(0)


def load_model(path, device):
    m = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    m.eval()
    return m


def loader(ds, idx=None):
    sub = ds if idx is None else Subset(ds, idx)
    return DataLoader(sub, batch_size=16, shuffle=False, collate_fn=collate_fn)


def stats(seqs, nchars):
    allp = np.concatenate(seqs)
    return {
        'points_per_char': float(np.median([len(s) / n for s, n in zip(seqs, nchars)])),
        'pen_lift_fraction': float(allp[:, 2].mean()),
        'x_offset_std': float(allp[:, 0].std()),
        'y_offset_std': float(allp[:, 1].std()),
    }


def main():
    device = get_device()
    ds = Dataset(PERSONAL)
    train_idx, val_idx, test_idx = split_indices(len(ds))
    print(f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test lines")

    base, ft = load_model(BASE, device), load_model(FT, device)
    results = {
        'lines': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
        'parameters': sum(p.numel() for p in ft.parameters()),
    }

    dw = Dataset(config.VAL_PATH, max_len=config.MAX_LEN)
    results['nll'] = {}
    for name, m in [('base', base), ('fine_tuned', ft)]:
        results['nll'][name] = {
            'personal_test': evaluate(m, loader(ds, test_idx), device),
            'deepwriting_val': evaluate(m, loader(dw), device),   # how much general ability is kept
        }
    print(json.dumps(results['nll'], indent=2))

    def raw(i):                         # dataset index -> row in the npz (filtered lines are skipped)
        return ds.indices[i]

    texts_all, strokes_all = ds.dict['texts'], ds.dict['strokes']
    test_texts = [str(texts_all[raw(i)]) for i in test_idx]
    test_strokes = [strokes_all[raw(i)] for i in test_idx]

    prime_ids = [raw(i) for i in train_idx if 8 <= len(str(texts_all[raw(i)])) <= 14]
    p = prime_ids[0]                    # the prime comes from training lines only
    prime_s, prime_t = strokes_all[p], str(texts_all[p])
    c2i = ds.char_to_idx

    chunks = [c for t in test_texts for c in wrap(t, CHUNK)][:N_CHUNKS]
    gen, secs, fin = [], [], []
    for c in chunks:
        t0 = time.time()
        pts, ok = sample_primed(ft, prime_s, prime_t, c, c2i, device, temp=TEMP)
        secs.append(time.time() - t0)
        fin.append(ok)
        gen.append(pts)

    results['real_test_lines'] = stats(test_strokes, [len(t) for t in test_texts])
    results['generated_primed'] = stats(gen, [len(c) for c in chunks])
    results['finished_first_try'] = float(np.mean(fin))
    results['seconds_per_chunk'] = float(np.median(secs))

    sub = chunks[:N_BEST]
    fails = sum(generate_best(ft, prime_s, prime_t, c, c2i, device,
                              tries=TRIES, temp=TEMP) is None for c in sub)
    results['best_of_n_failure_rate'] = fails / max(len(sub), 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print('saved', OUT)


if __name__ == '__main__':
    main()
