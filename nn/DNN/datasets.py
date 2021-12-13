import torch
import torch.nn as nn
import numpy as np
import os
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

class Dataset_res(data.Dataset): 
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        X = self.data[idx][:, :11, :]
        y = self.data[idx][:, 11:, :]
        return X, y 

class LWRDataset(data.Dataset): 
    def __init__(self, xi, data, initial, boundary): 
        self.xi = xi
        self.data = data
        self.initial = initial
        self.boundary = boundary
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        x = self.xi[idx] 
        y = self.data[idx][:, 1:, 1:]
        initial_i = self.initial[idx]
        boundary_i = self.boundary[idx]
        return x, initial_i, boundary_i, y 

class LWRDataset_res(data.Dataset): 
    def __init__(self, xi, data, initial, boundary): 
        self.xi = xi
        self.data = data 
        self.initial = initial
        self.boundary = boundary
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        xi = self.xi[idx] 
        x = self.data[idx] 
        y = self.data[idx][:, -12:, 1:]
        initial_i = self.initial[idx]
        boundary_i = self.boundary[idx]
        return xi, initial_i, boundary_i, x, y 

def train_epoch(model, train_loader, optimizer, criterion): 
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
    
    #     trues.append(y.cpu().data.numpy())
    #     preds.append(pred.cpu().data.numpy())
    
    # preds = np.concatenate(preds, axis = 0)
    # trues = np.concatenate(trues, axis = 0)
    
    return preds, trues, np.mean(mse) 

def eval_epoch(model, val_loader, criterion): 
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
            
        #     trues.append(y.cpu().data.numpy())
        #     preds.append(pred.cpu().data.numpy())

        # preds = np.concatenate(preds, axis = 0)
        # trues = np.concatenate(trues, axis = 0)

    return preds, trues, np.mean(mse)

def train_LWR(model, train_loader, optimizer, criterion, steps): 
    preds = []
    trues = []
    mse = [] 
    model.train()
    for xi, initial, boundary, y in train_loader: 
        xi, initial, boundary, y = xi.to(device), initial.to(device), boundary.to(device), y.to(device)
        pred = model(xi.long(), initial, boundary, steps) 
        
        loss = 0
        loss = criterion(pred, y)
        mse.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #     trues.append(y.cpu().data.numpy())
    #     preds.append(pred.cpu().data.numpy())
    
    # preds = np.concatenate(preds, axis = 0)
    # trues = np.concatenate(trues, axis = 0)
    
    return preds, trues, np.mean(mse) 

def train_hybrid_LWR(model, train_loader, optimizer, criterion, steps): 
    preds = []
    trues = []
    mse = [] 
    model.train()
    for xi, initial, boundary, x, y in train_loader: 
        xi, initial, boundary, x, y = xi.to(device), initial.to(device), boundary.to(device), x.to(device), y.to(device) 
        pred = model(xi, x, initial, boundary, steps) 
        loss = 0
        loss = criterion(pred, y)
        mse.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #     trues.append(y.cpu().data.numpy())
    #     preds.append(pred.cpu().data.numpy())
    
    # preds = np.concatenate(preds, axis = 0)
    # trues = np.concatenate(trues, axis = 0)
    
    return preds, trues, np.mean(mse) 

def eval_LWR(model, val_loader, criterion, steps): 
    preds = []
    trues = []
    mse = []
    
    model.eval()
    with torch.no_grad():
        for xi, initial, boundary, y in val_loader: 
            xi, initial, boundary, y = xi.to(device), initial.to(device), boundary.to(device), y.to(device) 
            pred = model(xi.long(), initial, boundary, steps) 
            loss = 0
            loss = criterion(pred, y)
            mse.append(loss.item()) 

        #     trues.append(y.cpu().data.numpy())
        #     preds.append(pred.cpu().data.numpy())

        # preds = np.concatenate(preds, axis = 0)
        # trues = np.concatenate(trues, axis = 0) 
    
    return preds, trues, np.mean(mse) 

def eval_hybrid_LWR(model, val_loader, criterion, steps): 
    preds = []
    trues = []
    mse = []
    
    model.eval()
    with torch.no_grad():
        for xi, initial, boundary, x, y in val_loader: 
            xi, initial, boundary, x, y = xi.to(device), initial.to(device), boundary.to(device), x.to(device), y.to(device) 
            pred = model(xi, x, initial, boundary, steps) 
            loss = 0
            loss = criterion(pred, y)
            mse.append(loss.item()) 
        
        #     trues.append(y.cpu().data.numpy())
        #     preds.append(pred.cpu().data.numpy())

        # preds = np.concatenate(preds, axis = 0)
        # trues = np.concatenate(trues, axis = 0) 
    
    return preds, trues, np.mean(mse) 


def eval_epoch_true(model, val_loader, criterion, std, mean): 
    preds = []
    trues = []
    mse = [] 
    model.eval()
    with torch.no_grad(): 
        for X, y in val_loader: 
            X, y = X.to(device), y.to(device)
            pred = model(X, 12) 
            pred[:, 0, :, :] = (pred[:, 0, :, :] * std[0] + mean[0])
            pred[:, 1, :, :] = (pred[:, 1, :, :] * std[1] + mean[1])
            pred[:, 2, :, :] = (pred[:, 2, :, :] * std[2] + mean[2])
                             
            y[:, 0, :, :] = (y[:, 0, :, :] * std[0] + mean[0])
            y[:, 1, :, :] = (y[:, 1, :, :] * std[1] + mean[1])
            y[:, 2, :, :] = (y[:, 2, :, :] * std[2] + mean[2])
            loss = 0
            loss = criterion(pred, y)
            mse.append(loss.item()) 
            
            trues.append(y.cpu().data.numpy())
            preds.append(pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)

    return preds, trues, np.sqrt(np.mean(mse)) 

def test_LWR(model, test_loader, criterion, steps): 
    preds = []
    trues = []
    mse = []
    
    model.eval()
    with torch.no_grad():
        for xi, initial, boundary, y in test_loader: 
            xi, initial, boundary, y = xi.to(device), initial.to(device), boundary.to(device), y.to(device) 
            pred = model(xi.long(), initial, boundary, steps) 
            loss = 0
            loss = criterion(pred, y)
            mse.append(loss.item()) 
            print(loss.item()) 
            trues.append(y.cpu().data.numpy())
            preds.append(pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)

    return preds, trues, np.mean(mse)

def test_hybrid_LWR(model, val_loader, criterion, test_sensors, steps): 
    preds = []
    trues = []
    mse = [] 
    model.eval()
    with torch.no_grad():
        for xi, initial, boundary, x, y in val_loader: 
            xi, initial, boundary, x, y = xi.to(device), initial.to(device), boundary.to(device), x.to(device), y.to(device) 
            pred = model(xi.long(), x, initial, boundary, steps)  
            loss = 0
            loss = criterion(pred[:, :, :, test_sensors], y[:, :, :, test_sensors])
            mse.append(loss.item()) 
            print(loss.item()) 
            trues.append(y.cpu().data.numpy())
            preds.append(pred.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0) 
    
    return preds, trues, np.mean(mse) 