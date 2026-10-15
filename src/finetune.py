import os
import random

import matplotlib

matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import Dataset, collate_fn
from model import SynthesisModel
from sample import plot_smooth, plot_strokes, sample
from train import compute_loss, evaluate, get_device

BASE = os.path.join(config.CKPT_DIR, 'base_pretrained.pt')
BEST = os.path.join(config.CKPT_DIR, 'personal_best.pt')

def main():
    device = get_device()

    ds = Dataset(config.PERSONAL)                      
    mean, std = ds.dict['mean'], ds.dict['std']

    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)

    # 10 percent goes to validation
    n_val = max(10, len(idx) // 10)
    val_ds, train_ds = Subset(ds, idx[:n_val]), Subset(ds, idx[n_val:])
    print(f"{len(train_ds)} train lines, {len(val_ds)} validation lines")

    train_loader = DataLoader(train_ds, batch_size=config.FT_BATCH, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.FT_BATCH, shuffle=False, collate_fn=collate_fn)

    model = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
    ckpt = torch.load(BASE, map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=config.FT_LR)

    os.makedirs(config.OUT_DIR, exist_ok=True)

    # run the model once and save the picture
    def snap(tag):
        pts, _ = sample(model, config.TEXT, ds.char_to_idx, device,
                        greedy=False, temp=0.3)
        plot_smooth(pts, mean, std, save_path=f'{config.OUT_DIR}/ft_{tag}.png')
        model.train()

    best = evaluate(model, val_loader, device)
    print(f"base model on your validation lines: {best:.4f}")
    snap('epoch0')
    bad = 0

    for epoch in range(1, config.FT_EPOCHS + 1):
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            loss = compute_loss(model, batch, device)
            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            if not torch.isfinite(gn):
                continue
            opt.step()
            run += loss.item()
            n += 1

        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{config.FT_EPOCHS} | train {run / max(n, 1):.4f} | val {val:.4f}")

        # early stopping
        if val < best:
            best, bad = val, 0
            torch.save(model.state_dict(), BEST)
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print("validation stopped improving, stopping early")
                break

        if epoch % 5 == 0:
            snap(f'epoch{epoch}')

    print(f"best validation loss: {best:.4f}")


if __name__ == '__main__':
    main()
