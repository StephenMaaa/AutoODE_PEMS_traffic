import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
device = "cuda"

class FC(nn.Module): 
    def __init__(self, input_dim, input_len, hidden_dim, output_dim): 
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(input_len * input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim) 
        )
    
    def forward(self, x, output_length): 
#         x = x.permute(0, 2, 1).float().to(device)
        x = x.reshape(x.shape[0], -1) 

        outputs = []
        for i in range(output_length): 
            out = self.fc(x)
            xx = torch.cat([x[:, self.input_dim:], out], dim = 1)  
            outputs.append(out)
#             print(out.shape)
        outputs = torch.cat(outputs, dim = 1)
        outputs = outputs.reshape((x.shape[0], 3, 12, -1))
        return outputs