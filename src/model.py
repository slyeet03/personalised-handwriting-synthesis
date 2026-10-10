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

    def forward(self, strokes, char_onehot, text_mask):
        batch, T, _ = strokes.shape
        dev = strokes.device
        H = self.hidden_size

        h1 = torch.zeros(batch, H, device=dev)
        c1 = torch.zeros(batch, H, device=dev)
        
        h2 = torch.zeros(batch, H, device=dev)
        c2 = torch.zeros(batch, H, device=dev)

        h3 = torch.zeros(batch, H, device=dev)
        c3 = torch.zeros(batch, H, device=dev)

        kappa = torch.zeros(batch, self.K, device=dev)
        window = torch.zeros(batch, self.vocab_size, device=dev)

        outputs = []
        for t in range(T):
            x = strokes[:, t]                                        

            h1, c1 = self.lstm1(torch.cat([x, window], dim=1), (h1, c1))
            window, kappa, phi = self.attention(h1, kappa, char_onehot, text_mask)
            h2, c2 = self.lstm2(torch.cat([x, h1, window], dim=1), (h2, c2))
            h3, c3 = self.lstm3(torch.cat([x, h2, window], dim=1), (h3, c3))

            outputs.append(torch.cat([h1, h2, h3], dim=1))           

        outputs = torch.stack(outputs, dim=1)                        
        return self.mdn(outputs)
