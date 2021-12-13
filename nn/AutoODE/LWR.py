import torch 
import torch.nn as nn
import numpy as np
from DNN import Seq2Seq
device = torch.device("cuda")

class LWR_Residual(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(LWR_Residual, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
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
    
    def forward(self, x, t): 
#         x = x.permute(0, 2, 1).float().to(device)
#         x = x.reshape(x.shape[0], -1) 
        # x [120] t [3550]
        x_len = x.shape[0]
        xx, tt = np.meshgrid(x.cpu().data.numpy(), t.cpu().data.numpy())
        X = torch.tensor(np.vstack((np.ravel(xx), np.ravel(tt))).T).float().to(device)
        outputs = []

        for i in range(t.shape[0]): 
            out = self.fc(X[i * x_len : i * x_len + x_len, :].reshape(-1).float())
            outputs.append(out)
        outputs = torch.stack(outputs)
        outputs = outputs.reshape((t.shape[0], 3, x_len))
        
        return outputs

import torch.nn.functional as F

class PiecewiseLinearModel(nn.Module):
    def __init__(self, n_breaks, num_regions):
        super(PiecewiseLinearModel, self).__init__()
        self.breaks = nn.Parameter(torch.rand((num_regions, 1, n_breaks)))
        self.linear = nn.Linear(n_breaks + 1, 1)
    def forward(self, xx): 
        if len(xx.shape) < 3: 
            xx = xx.unsqueeze(-1)
        out = torch.cat([xx, F.relu(xx - self.breaks)],2)
        
        return self.linear(out).squeeze(-1) 

# class LWR_residual_adjoint(nn.Module): 
#     def __init__(self, nx, dx, dt, kj, vf, x_size, tskip, plm = True, plm_vf = True, initial='random', boundary='zeros',
#                 fix_vf=False, parstep=1): 
#         super(LWR_residual_adjoint, self).__init__()
#         self.LWR_model = LWR(nx, dx, dt, kj, vf, tskip, plm = True, plm_vf = False, 
#                           initial={}, boundary={}, fix_vf=False, parstep=10).to(device)
#         self.residual_model = LWR_Residual(x_size * 2 - 2, 512, x_size * 3 - 3)
    
#     def forward(self, xi, initial, boundary, tsteps): 
#         x = torch.linspace(0, 1, len(xi) - 1).to(device)
#         tt = torch.linspace(0, 1, 24 - 1).to(device) 
#         pred = self.LWR_model(xi, initial, boundary, tsteps) 
#         residual = self.residual_model(x, tt) 
#         pred[:, :, 1:] += residual
        
#         return pred

class LWR_seq2seq(nn.Module): 
    def __init__(self, LWR): 
        super(LWR_seq2seq, self).__init__()
        self.LWR_model = LWR.to(device) 
        for p in self.LWR_model.parameters(): 
            p.requires_grad = False 
#         LWR_w(nx, dx, dt, kj, vf, tskip, plm = True, plm_vf = False, 
#                           initial={}, boundary={}, fix_vf=False, parstep=10).to(device)
        self.residual_model = Seq2Seq(input_dim = 231, hidden_dim = 512, output_dim = 231, num_layers = 1).to(device)
    
    def forward(self, xi, x, initial, boundary, tsteps): 
        with torch.no_grad():
            pred1 = self.LWR_model(xi, initial[:, 0], boundary[:, 0], tsteps) # first 12
            pred2 = self.LWR_model(xi, initial[:, 1], boundary[:, 1], tsteps) # last 12
        res = x[:, :, 1:13, 1:] - pred1.detach().clone() 

        residual = self.residual_model(res.float(), 12) # last 12

        pred = pred2.detach().clone()
        pred += residual # residual[:, :, 1:] 
        
        return pred.double() 

class LWR_batch_version(nn.Module):
    def __init__(self, nx, dx, dt, kj, vf, tskip, plm = True, plm_vf = True, initial='random', boundary='zeros',
                fix_vf=False, parstep=1): 
        super(LWR_batch_version, self).__init__()
        self.nx = nx
        #xi are the locations along distance where data is available
        self.xi = [] 
        self.tskip = tskip 
#         self.y_max = y_max 
#         self.y_min = y_min 
        self.cmps=["k","q","u"]  #density, flow, velocity
#         self.initial={}
#         self.boundary={}
        
        self.initial = {} # shape: [batch, 3, # sensors]
        self.boundary = {} # shape: [batch, seq_len, 3]
            
        #factor by which parameter resolution is reduced with respect to nx
        self.parstep=parstep
        self.flag = plm 
        
        # use piecewise linear function for kj
        if plm == False: 
            self.kappa = torch.nn.Parameter(torch.tensor(kj[::self.parstep]), requires_grad=True)
#             print(self.kappa, self.kappa.shape)
        else: 
            self.plm = PiecewiseLinearModel(n_breaks = 2, num_regions = self.nx).to(device)
        
        #characteristic velocity vf
        if not fix_vf:
            self.vf=torch.nn.Parameter(torch.tensor(vf[::self.parstep]), requires_grad=True) 
        else:
            self.vf=torch.tensor(vf[::self.parstep])
        
        self.dx = torch.tensor(dx)
        self.dt = torch.tensor(dt)
        
    def forward(self, xi, initial, boundary, tsteps): 
        self.xi = xi 
        nt=tsteps 
        self.initial = torch.nn.Parameter(initial).to(device)
        self.boundary = torch.nn.Parameter(boundary).to(device) 

        batch_size = self.initial.shape[0]
 
        self.k = [self.initial[:, 0]]
        self.q = [self.initial[:, 1]]
        self.u = [self.initial[:, 2]]
        
        #initial values at output points
        self.ki=[self.initial[:, 0][:, self.xi[0]]]
        self.ui=[self.initial[:, 1][:, self.xi[0]]]
        self.qi=[self.initial[:, 2][:, self.xi[0]]]

        for n in range(1, nt):
            #This corresponds to the upwind scheme according to Gaddam et al. (2015).
            nk=torch.zeros((batch_size, self.nx), requires_grad=True).double().to(device)
            nu=torch.zeros((batch_size, self.nx), requires_grad=True).double().to(device)
            nq=torch.zeros((batch_size, self.nx), requires_grad=True).double().to(device)
            
            #new values for 3 variables stored in one tensor per time step
            nk[:, 0] = self.boundary[:, 0, n]
            nq[:, 0] = self.boundary[:, 1, n]
            nu[:, 0] = self.boundary[:, 2, n] 
            
            #full-resolution tensor hkappa and hv1, from down-sampled versions
            idx=torch.arange(self.nx) / self.parstep
            if self.flag: 
                hkappa = self.kappa[:, n - 1] 
            else: 
                hkappa=self.kappa[idx.long()]
            hvf=self.vf[idx.long()] #.repeat(batch_size, 1)

            nk[:, 1:] = self.k[n-1][:, 1:] - 0.0016 * self.dt / self.dx * (self.q[n-1][:, 1:]-self.q[n-1][:, :-1]) 
#             idx = nk < 0. 
#             nk[idx] = 0.01 
            nu[:, 1:] = hvf[1:] * (1 - nk[:, 1:] / hkappa[1:])
#             idx = nu < 0.
#             nu[idx] = 2  
            nq[:, 1:] = nk[:, 1:] * nu[:, 1:] 
#             idx = nq < 0.
#             nq[idx] = 0.2  
            
            ### 
            # Method of lines + RK4 method 
            # dt * f(t[n], y[n]) 
#             k1_k = - (self.q[n-1][2:] - self.q[n-1][:-2]) / (2 * self.dx) # central difference 
#             k1_k = - (self.q[n-1][:, 1:] - self.q[n-1][:, :-1]) / self.dx # finite difference 
#             nk_1 = torch.cat((nk[:, 0].unsqueeze(1), self.k[n-1][:, 1:] + 0.0016 * k1_k / 2 * self.dt), dim = 1) 
#             idx = nk_1 < 0. 
#             nk_1[idx] = 0.01 
#             nu_1 = hvf[1:] * (1 - nk_1[:, 1:] / hkappa[1:]) 
#             idx = nu_1 < 0.
#             nu_1[idx] = 2 
#             nq_1 = nk_1[:, 1:] * nu_1
#             nq_1 = torch.cat((nq[:, 0].unsqueeze(1), nq_1), dim = 1) 
#             idx = nq_1 < 0.
#             nq_1[idx] = 0.2 
            
#             # dt * f(t[n] + dt/2, y[n] + k1/2)  
#             k2_k = - (nq_1[:, 1:] - nq_1[:, :-1]) / self.dx 
#             nk_2 = torch.cat((nk[:, 0].unsqueeze(1), self.k[n-1][:, 1:] + 0.0016 * k2_k / 2 * self.dt), dim = 1) 
#             idx = nk_2 < 0. 
#             nk_2[idx] = 0.01 
#             nu_2 = hvf[1:] * (1 - nk_2[:, 1:] / hkappa[1:]) 
#             idx = nu_2 < 0.
#             nu_2[idx] = 2 
#             nq_2 = nk_2[:, 1:] * nu_2 
#             nq_2 = torch.cat((nq[:, 0].unsqueeze(1), nq_2), dim = 1) 
#             idx = nq_2 < 0.
#             nq_2[idx] = 0.2 
            
#             k3_k = - (nq_2[:, 1:] - nq_2[:, :-1]) / self.dx 
#             nk_3 = torch.cat((nk[:, 0].unsqueeze(1), self.k[n-1][:, 1:] + 0.0016 * k3_k * self.dt), dim = 1) 
#             idx = nk_3 < 0. 
#             nk_3[idx] = 0.01 
#             nu_3 = hvf[1:] * (1 - nk_3[:, 1:] / hkappa[1:]) 
#             idx = nu_3 < 0.
#             nu_3[idx] = 2 
#             nq_3 = nk_3[:, 1:] * nu_3 
#             nq_3 = torch.cat((nq[:, 0].unsqueeze(1), nq_3), dim = 1) 
#             idx = nq_3 < 0.
#             nq_3[idx] = 0.2 

#             # dt * f(t[n] + dt, y[n] + k3) 
#             k4_k = - (nq_3[:, 1:] - nq_3[:, :-1]) / self.dx
#             nk[:, 1:] += self.k[n-1][:, 1:] + 0.0016 * 1/6 * (k1_k + 2 * k2_k + 2 * k3_k + k4_k) * self.dt 
#             idx = nk < 0. 
#             nk[idx] = 0.01 
#             nu[:, 1:] += hvf[1:] * (1 - nk[:, 1:] / hkappa[1:]) 
#             idx = nu < 0.
#             nu[idx] = 2 
#             nq[:, 1:] += nk[:, 1:] * nu[:, 1:] 
#             idx = nq < 0.
#             nq[idx] = 0.2 

            self.k.append(nk)
            self.u.append(nu)
            self.q.append(nq) 
            
            #only output every tskip timesteps
            if (n % self.tskip) == 0: 
                self.ki.append(nk[:, self.xi[0]])
                self.ui.append(nu[:, self.xi[0]])
                self.qi.append(nq[:, self.xi[0]])

        pred = torch.stack([torch.stack(self.ki, dim = 1),torch.stack(self.qi, dim = 1),torch.stack(self.ui, dim = 1)], dim = 1) 
        
        return pred[:, :, 1:, 1:] 