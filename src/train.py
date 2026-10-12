import os

import matplotlib

matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader

import config
from dataset import BucketSampler, Dataset, collate_fn
from mdn import mdn_loss
from model import SynthesisModel
from sample import plot_strokes, sample

RESUME_PATH = os.path.join(config.CKPT_DIR,'resume.pt')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_loss(model, batch, device):
    strokes, texts, mask, text_mask = [b.to(device) for b in batch]
    char_onehot = torch.nn.functional.one_hot(texts, num_classes=config.VOCAB_SIZE).float()

    # shifting the target and input as input is one point then output is next point and the series go on so the start of input is p0 and for target it should be p1
    inputs  = strokes[:, :-1]
    targets = strokes[:, 1:]
    mask    = mask[:, 1:].float()

    pi, mu_x, mu_y, sx, sy, rho, e = model(inputs, char_onehot, text_mask)

    # 0 -> del x
    # 1 -> del y
    # 2 -> pen state
    return mdn_loss(pi, mu_x, mu_y, sx, sy, rho, e,
                    targets[..., 0], targets[..., 1], targets[..., 2], mask)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = [compute_loss(model, batch, device).item() for batch in loader]
    model.train()
    return sum(losses) / len(losses)


def main():
    device = get_device()
    print("device:", device)

    train_ds = Dataset(config.TRAIN_PATH, max_len=config.MAX_LEN)

    mean, std = train_ds.dict['mean'], train_ds.dict['std']
    char_to_idx = train_ds.char_to_idx
    os.makedirs(config.OUT_DIR, exist_ok=True)

    train_loader = DataLoader(train_ds,
                          batch_sampler=BucketSampler(train_ds.lengths, config.BATCH_SIZE),
                          collate_fn=collate_fn)    
    val_loader = DataLoader(Dataset(config.VAL_PATH, max_len=config.MAX_LEN), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SynthesisModel(vocab_size=config.VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    best_path = os.path.join(config.CKPT_DIR, 'best.pt')

    start_epoch = 0
    best_val = float('inf')

    if os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device)

        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if 'opt' in ckpt:
                optimizer.load_state_dict(ckpt['opt'])
            start_epoch = ckpt.get('epoch', 121) + 1
            best_val = ckpt.get('best_val', -2.55)
            print(f"resumed from epoch {start_epoch}, best_val {best_val:.4f}")
        else:
            model.load_state_dict(ckpt)
            start_epoch = 122
            best_val = -2.55
            print("old-format resume.pt: loaded weights only (fresh optimizer)")

        for g in optimizer.param_groups:
            g['lr'] = config.LR
    
    skipped = 0                                     

    for epoch in range(start_epoch, config.EPOCHS): 
        running = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            loss = compute_loss(model, batch, device)

            if not torch.isfinite(loss):
                skipped += 1
                print(f"non-finite loss at epoch {epoch+1} step {step}, skipped ({skipped})")
                if skipped > 20: return
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)   
            if not torch.isfinite(grad_norm):
                skipped += 1
                print(f"non-finite gradient at epoch {epoch+1} step {step}, skipped ({skipped})")
                if skipped > 20: return
                continue

            optimizer.step()

            running += loss.item()
            n += 1

        train_loss = running / max(n, 1)            
        msg = f"epoch {epoch+1}/{config.EPOCHS} | train loss {train_loss:.4f}"

        val_loss = evaluate(model, val_loader, device)
        msg += f" | val loss {val_loss:.4f}"
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        print(msg)

        
        torch.save({'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val': best_val},
                   RESUME_PATH)

        if (epoch + 1) % config.SAMPLE_EVERY == 0:
            points, _ = sample(model, config.SAMPLE_TEXT, char_to_idx, device, greedy=False, temp=0.3)
            plot_strokes(points, mean, std, save_path=f'{config.OUT_DIR}/epoch_{epoch+1}.png')
            model.train()

    torch.save(model.state_dict(), os.path.join(config.CKPT_DIR, 'last.pt'))

if __name__ == '__main__':
    main()
