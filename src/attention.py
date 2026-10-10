import torch
import torch.nn as nn

# tracks on which character the pen is currently on
# it outputs an array of probabilities for each character + previous position of the pen to be used in the function again + phi for better plots

class Attention(nn.Module):
    def __init__(self, hidden_size, K):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(hidden_size, 3 * K)

    def forward(self, h, kappa_prev, char_onehot, text_mask):
        K = self.K
        raw = self.linear(h)                     

        # brightness
        alpha = torch.exp(raw[:, 0*K : 1*K])
        # sharpness
        beta  = torch.exp(raw[:, 1*K : 2*K])    
        # position
        kappa = kappa_prev + torch.exp(raw[:, 2*K : 3*K])

        U = char_onehot.shape[1]
        u = torch.arange(U, device=h.device).float().view(1, 1, U)   

        alpha_ = alpha.unsqueeze(-1)             # (batch, K, 1)
        beta_  = beta.unsqueeze(-1)
        kappa_ = kappa.unsqueeze(-1)

        phi = (alpha_ * torch.exp(-beta_ * (kappa_ - u) ** 2)).sum(dim=1)   # (batch, U)
        # padding doesnt fuck up
        phi = phi * text_mask                    

        window = torch.bmm(phi.unsqueeze(1), char_onehot).squeeze(1)       # (batch, vocab)

        return window, kappa, phi
