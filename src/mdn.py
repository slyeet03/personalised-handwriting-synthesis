import torch
import torch.nn as nn

# mdn head basically takes the hidden state vector of some size and 
# transform them into a different vector which represents where the 
# pen should go next

class MDNHead:
    # M being how many different plausible next stroke at once we want
    def __init__(self, hidden_size, M):
        total_output_size = M*6+1
        
        self.linear = nn.Linear(hidden_size,total_output_size)
        self.M = M
        
 