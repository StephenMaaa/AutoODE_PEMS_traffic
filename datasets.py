import torch
import torch.nn as nn
import numpy as np
from torch.utils import data 
device = "cuda" 

class Dataset(data.Dataset): 
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        X = self.data[idx][:, :12, :]
        y = self.data[idx][:, 12:, :]
        return X, y

class LWRDataset(data.Dataset): 
    def __init__(self, data, initial, boundary):
        self.data = data
        self.initial = initial
        self.boundary = boundary
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        X = self.data[idx][:, :12, :]
        y = self.data[idx][:, 12:, :]
        initial_i = self.initial[idx]
        boundary_i = self.boundary[idx]
        return X, initial_i, boundary_i, y

def train(model, train_loader, optimizer, criterion): 
    preds = []
    trues = []
    mse = [] 
    
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X, 12) 
        
        loss = 0
        loss = criterion(pred, y)
        mse.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        trues.append(y.cpu().data.numpy())
        preds.append(pred.cpu().data.numpy())
    
    preds = np.concatenate(preds, axis = 0)
    trues = np.concatenate(trues, axis = 0)
    
    return preds, trues, np.mean(mse)

def val(model, val_loader, best_loss, criterion, name): 
    preds = []
    trues = []
    mse = []
    
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X, 12)
            
            loss = 0
            loss = criterion(pred, y)
            mse.append(loss.item()) 
            
            trues.append(y.cpu().data.numpy())
            preds.append(pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
    
    val_loss = np.mean(mse)
    if val_loss <= best_loss: 
        best_loss = val_loss
#             torch.save(model.state_dict(), 'model.pt')
        torch.save(model, name + ".pth")

    return preds, trues, val_loss, best_loss