import numpy as np
import torch
import torch.nn.functional as F

import config
import vocab


# pick one point from the output
def draw_point(out, temp):
    pi, mu_x, mu_y, sx, sy, rho, e = [o[0] for o in out]

    # icreasing the chance of picking the best gaussian bump using temp
    p = pi ** (1.0 / temp)
    p = p / p.sum()
    k = torch.multinomial(p, 1).item()
    
    # shrinking the spread so lower temp gives less scatter
    s_x, s_y = sx[k].item() * temp, sy[k].item() * temp
    r = rho[k].item()

    z1, z2 = np.random.randn(2)
    dx = mu_x[k].item() + s_x * z1
    dy = mu_y[k].item() + s_y * (r * z1 + np.sqrt(1 - r ** 2) * z2)

    pen = float(np.random.rand() < e[0].item())
    return [dx, dy, pen]


def sample_primed(model, prime_strokes, prime_text, text, char_to_idx, device,
                  temp=0.3, max_steps=None):
    model.eval()
    finished=False
    # joining the prime text with the generated text with a space
    full = prime_text + ' ' + text
    idx = torch.tensor([vocab.encode_text(full, char_to_idx)], device=device)
    U = idx.shape[1]
    
    # adding one extra 0 in the end to let the cursor know when to end the sentence
    onehot = F.pad(F.one_hot(idx, config.VOCAB_SIZE).float(), (0, 0, 0, 1))   # extra empty end slot
    text_mask = torch.ones(1, U + 1, dtype=torch.bool, device=device)

    state = model.init_state(1, device)
    prime = torch.tensor(prime_strokes, dtype=torch.float32, device=device)

    pts = []
    max_steps = max_steps or 30 * len(text)
    with torch.no_grad():
        # priming->we put real handwriting for one timestep then update the model parameters according to the given writing
        for t in range(len(prime)):
            h, state, phi = model.step(prime[t].unsqueeze(0), state, onehot, text_mask)
        

        # generation->sample from where the prime left off
        stop_at = None
        for _ in range(max_steps):
            x = draw_point(model.mdn(h), temp)
            pts.append(x)
            h, state, phi = model.step(torch.tensor([x], dtype=torch.float32, device=device),
                                       state, onehot, text_mask)
            if stop_at is None and len(pts) > 20 and phi[0].argmax().item() == U:
                stop_at = len(pts)
            if stop_at is not None and (x[2] == 1.0 or len(pts) - stop_at >= 12):
                finished = True
                break
    
    # only generated points are returned not the prime
    return np.array(pts), finished
