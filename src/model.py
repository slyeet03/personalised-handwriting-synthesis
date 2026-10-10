import torch
import torch.nn as nn

from attention import Attention
from mdn import MDNHead


class SynthesisModel(nn.Module):
    def __init__(self, vocab_size=70, hidden_size=256, K=10, M=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.K = K

        self.lstm1 = nn.LSTMCell(3 + vocab_size, hidden_size)
        self.attention = Attention(hidden_size, K)
        self.lstm2 = nn.LSTMCell(3 + hidden_size + vocab_size, hidden_size)
        self.lstm3 = nn.LSTMCell(3 + hidden_size + vocab_size, hidden_size)
        self.mdn = MDNHead(3 * hidden_size, M)

    def init_state(self, batch, device):
        H = self.hidden_size
        z = lambda n: torch.zeros(batch, n, device=device)
        
        return dict(h1=z(H), c1=z(H), h2=z(H), c2=z(H), h3=z(H), c3=z(H), kappa=z(self.K), window=z(self.vocab_size))

    def step(self, x, s, char_onehot, text_mask):
        h1, c1 = self.lstm1(torch.cat([x, s['window']], 1), (s['h1'], s['c1']))
        window, kappa, phi = self.attention(h1, s['kappa'], char_onehot, text_mask)
        h2, c2 = self.lstm2(torch.cat([x, h1, window], 1), (s['h2'], s['c2']))
        h3, c3 = self.lstm3(torch.cat([x, h2, window], 1), (s['h3'], s['c3']))

        new_s = dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, kappa=kappa, window=window)
    
        return torch.cat([h1, h2, h3], 1), new_s, phi

    def forward(self, strokes, char_onehot, text_mask):
        state = self.init_state(strokes.shape[0], strokes.device)
        outs = []

        for t in range(strokes.shape[1]):
            out, state, _ = self.step(strokes[:, t], state, char_onehot, text_mask)
            outs.append(out)
    
        return self.mdn(torch.stack(outs, dim=1))
