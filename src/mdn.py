import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# mdn head basically takes the hidden state vector of some size and 
# transform them into a different vector which represents where the 
# pen should go next

class MDNHead:
    # M being how many different plausible next stroke at once we want
    def __init__(self, hidden_size, M):
        super().__init__()
        self.M = M

        # pi, mu_x, mu_y, sigma_x, sigma_y, rho, e
        # pi -> probability of each component
        # mu_x, ,mu_y -> coordinate offsets
        # sigma_x, sigma_y -> spread
        # rho -> correlation 
        # e -> pen lift probability
        total_output_size = self.M * 6 + 1
        self.linear = nn.Linear(hidden_size, total_output_size)

    def forward(self, hidden_state):
        # hidden_state shape: (batch, seq_len, hidden_size)
        raw = self.linear(hidden_state)  # shape: (batch, seq_len, M*6 + 1)

        M = self.M
        pi_raw      = raw[..., 0*M : 1*M]
        mu_x        = raw[..., 1*M : 2*M]
        mu_y        = raw[..., 2*M : 3*M]
        sigma_x_raw = raw[..., 3*M : 4*M]
        sigma_y_raw = raw[..., 4*M : 5*M]
        rho_raw     = raw[..., 5*M : 6*M]
        e_raw       = raw[..., 6*M : 6*M + 1]

        pi      = F.softmax(pi_raw, dim=-1)
        sigma_x = torch.exp(sigma_x_raw)
        sigma_y = torch.exp(sigma_y_raw)
        rho     = torch.tanh(rho_raw)
        e       = torch.sigmoid(e_raw)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho, e
        

def mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho, e, true_dx, true_dy, true_pen, mask):
    # adding one more dimension to true_dx so it can be subtracted from mu_x which has three dimensions cuz mdn
    true_dx = true_dx.unsqueeze(-1)   # (batch, seq_len, 1)
    true_dy = true_dy.unsqueeze(-1)

    # (t-y)
    dx = true_dx - mu_x               # (batch, seq_len, M)
    dy = true_dy - mu_y               

    # stats and prob normal distribution shit
    Z = (dx / sigma_x)**2 + (dy / sigma_y)**2 - (2 * rho * dx * dy) / (sigma_x * sigma_y)

    one_minus_rho2 = 1 - rho**2

    # normalizing constant
    denom = 2 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho2)
    N = torch.exp(-Z / (2 * one_minus_rho2)) / denom

    mixture_density = torch.sum(pi * N, dim=-1)   # (batch, seq_len)
    loss_position = -torch.log(mixture_density + 1e-8)

    # e had 3 dimensions so reducing it to 2 to do further calculations
    e = e.squeeze(-1)                             # (batch, seq_len)
    # binary cross entropy loss for the pen up down
    loss_pen = -(true_pen * torch.log(e + 1e-8) +
                 (1 - true_pen) * torch.log(1 - e + 1e-8))

    loss_per_timestep = loss_position + loss_pen
    # neutralising the padding so the additional 0 doesn't fuck up the loss
    masked_loss = loss_per_timestep * mask
    final_loss = masked_loss.sum() / mask.sum()

    return final_loss
