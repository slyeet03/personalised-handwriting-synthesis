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
        
 
