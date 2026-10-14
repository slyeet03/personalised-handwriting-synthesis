import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import vocab


def sample(model, text, char_to_idx, device, max_steps=1500, greedy=False, temp=1.0):
    model.eval()
    idx = torch.tensor([vocab.encode_text(text, char_to_idx)], device=device)
    char_onehot = F.one_hot(idx, 70).float()
    U = idx.shape[1]
    char_onehot = F.pad(char_onehot, (0, 0, 0, 1))
    text_mask = torch.ones(1, U+1, dtype=torch.bool, device=device)

    state = model.init_state(1, device)
    x = torch.zeros(1, 3, device=device)
    points, phis = [], []

    with torch.no_grad():
        for _ in range(max_steps):
            h, state, phi = model.step(x, state, char_onehot, text_mask)
            pi, mu_x, mu_y, sx, sy, rho, e = model.mdn(h)

            p = pi[0] ** (1.0 / temp)
            p = p / p.sum()
            # picking a random output but the choice being weighted
            k = torch.multinomial(pi, 1).item()

            # gaussian bump's center spread and correlation
            m_x, m_y = mu_x[0, k].item(), mu_y[0, k].item() # center
            s_x, s_y, r = sx[0, k].item()*temp, sy[0, k].item()*temp, rho[0, k].item() #spread

            # picking actual point inside that bump
            if greedy:
                dx = m_x  
                dy = m_y
                pen = float(e>0.5)
            else:
                z1, z2 = np.random.randn(2) # two random point
                dx = m_x + s_x * z1 # center+spread*noise
                dy = m_y + s_y * (r * z1 + np.sqrt(1 - r**2) * z2) # center+spread*noise partially taken from z1 and partly fresh
                
                # pen lifts when the model says pen lifts 
                pen = float(np.random.rand() < e[0, 0].item())

            # save all the shit and feed the sampled point back in as next input
            points.append([dx, dy, pen])
            phis.append(phi[0].cpu().numpy())
            x = torch.tensor([[dx, dy, pen]], dtype=torch.float32, device=device)

            if len(points) > 20 and phi[0].argmax().item() == U:
                break

    return np.array(points), np.array(phis)

def plot_strokes(points, mean, std, save_path=None):
    # reverse engineer this shit to get raw points
    raw = points[:, :2] * std[:2] + mean[:2]
    # turning offsets to actual position on the page
    xy = np.cumsum(raw, axis=0)
    pen = points[:, 2]

    plt.figure(figsize=(10,3))

    start = 0
    for i in range(len(pen)):
        if pen[i] == 1 or i == len(pen) - 1:
            plt.plot(xy[start:i+1, 0], xy[start:i+1, 1], 'k', linewidth=1)
            start = i + 1

    plt.gca().invert_yaxis()      
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()



def catmull_rom(p, n=8):
    p = np.asarray(p, dtype=float)
    if len(p) < 3:
        return p
    P = np.vstack([p[0], p, p[-1]])              
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1:])
    return np.vstack(out)


def plot_smooth(points, mean, std, save_path=None, n=8):
    raw = points[:, :2] * std[:2] + mean[:2]
    xy = np.cumsum(raw, axis=0)
    pen = points[:, 2]

    plt.figure(figsize=(10, 3))
    start = 0
    for i in range(len(pen)):
        if pen[i] == 1 or i == len(pen) - 1:
            s = xy[start:i + 1]
            if len(s) == 1:
                plt.plot(s[0, 0], s[0, 1], 'k.', markersize=2)      
            else:
                c = catmull_rom(s, n)
                plt.plot(c[:, 0], c[:, 1], 'k', linewidth=1.2, solid_capstyle='round')
            start = i + 1
    plt.gca().invert_yaxis()
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
