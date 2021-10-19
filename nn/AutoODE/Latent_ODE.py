from torchdiffeq import odeint
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
device = "cuda"

class Latent_ODE(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, rhidden = 20, aug = False, aug_dim = 2):
        super(Latent_ODE, self).__init__()
        self.aug = aug
        self.aug_dim = aug_dim
        if self.aug:
            self.rec = RecognitionRNN(latent_dim, obs_dim+aug_dim, rhidden)
        else:
            self.rec = RecognitionRNN(latent_dim, obs_dim, rhidden)
    
        self.func = LatentODEfunc(latent_dim, nhidden)
        self.dec = LatentODEDecoder(latent_dim, obs_dim, nhidden)
        
    def forward(self, x, output_length):
        time_steps = torch.arange(0, output_length, 0.01).float().to(device)[:output_length] 
        batch_size = x.size(0) 
        input_length = x.size(2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((batch_size, input_length, -1))
        if self.aug:
            aug_ten = torch.zeros(x.shape[0], x.shape[1], self.aug_dim).float().to(device)
            x = torch.cat([x, aug_ten], dim = -1)
#         print(xx.shape)
#         print(torch.flip(xx, [1]).shape)
        z0 = self.rec.forward(torch.flip(x, [1]))
        pred_z = odeint(self.func, z0, time_steps).permute(1, 0, 2)
        out = self.dec(pred_z)
#         print(out.shape)
        out = out.reshape((batch_size, output_length, 3, -1))
        out = out.permute((0, 2, 1, 3))
        return out  
    
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, latent_dim)
        )
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.model(x)
        return out
    
class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.model = nn.GRU(obs_dim, nhidden, batch_first = True)
        self.linear = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        #h0 = torch.zeros(1, x.shape[0], self.nhidden).to(device)
        output, hn = self.model(x)#, h0
        return self.linear(hn[0])
    
class LatentODEDecoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(LatentODEDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, obs_dim)
        )
        
    def forward(self, z):
        out = self.model(z)
        return out 