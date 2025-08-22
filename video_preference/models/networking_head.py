import torch
import torch.nn as nn


class NetworkingHead(nn.Module):
    """
    A simple linear layer as networking head for NetLLM.
    """
    def __init__(self, input_dim, out_dim, fut_window=None):
        super().__init__()
        self.input_dim = input_dim
        self.fut_window = fut_window
        self.hidden_size_1 = 256
        self.hidden_size_2 = 32
        self.dropout = 0.1
        self.networking_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size_1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size_2, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, input_logits):
        last_one = input_logits.shape[1]
        needed_logits = input_logits[:, last_one-1, :] # 取最新的时间序列
        needed_logits = torch.unsqueeze(needed_logits, dim=1)
        prediction = self.networking_head(needed_logits)
        return prediction
    
    
    def teacher_forcing(self, input_logits):
        size = input_logits.shape[1]
        needed_logits = input_logits[:, size-self.fut_window-1:size-1, :]
        prediction = self.networking_head(needed_logits)
        return prediction
    