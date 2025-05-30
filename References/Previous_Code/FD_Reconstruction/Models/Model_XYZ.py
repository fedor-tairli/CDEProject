# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import numpy as np
import hexagdly
import matplotlib.pyplot as plt
import os

import sys
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# add code path into the path
sys.path.append(Paths.code_path)
import ManageData as MD





def Loss(Pred,Truth,normState = 'Net'):
    # Break The code here
    assert Pred != None , 'TestingCodeBreak'
    # Check that shapes match
    assert Pred.shape[1:] == torch.Size([3]), 'Pred shape is {}'.format(Pred.shape)      # Already Summed
    assert Truth.shape[1:] == torch.Size([3]), 'Truth shape is {}'.format(Truth.shape) # Already Summed
    assert Pred.dtype == Truth.dtype, 'Pred and Truth dtypes do not match'    
    assert Pred.shape == Truth.shape , 'Shapes dont Match'
    assert not torch.any(torch.isnan(Pred)), 'Pred has NaNs'
    assert not torch.any(torch.isnan(Truth)), 'Truth has NaNs'

    
    # print(Pred.shape)
    # print(Truth.shape)
    # Check normalisation state
    criterion = nn.MSELoss()
    loss = criterion(Pred,Truth)
    return loss



def validate(model,dataloader,Loss,device = 'cuda'):
    model.eval()
    val_loss = 0
    val_loss_Phi = 0
    val_loss_Theta = 0

    with torch.no_grad():
        for Main, Truth in dataloader:
            Main = Main.to(device)
            Truth = Truth.to(device)
            Y_pred,normState = model(Main,normStateIn = 'Net',normStateOut = normState)
            loss = Loss(Y_pred,Truth,normState)
            val_loss += loss.item()
            
            
    return val_loss/len(dataloader)



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Main, Truth):
        self.Main = Main
        self.Truth = Truth
    def __len__(self):
        return len(self.Main)
    def __getitem__(self, idx):
        return self.Main[idx,:,:,:], self.Truth[idx,:]
    




class XYZEstimator(nn.Module):
    def __init__(self,Dtype=torch.float16):
        super(XYZEstimator, self).__init__()
        # Info 
        self.Name = 'XYZEstimator'
        self.Description = '''
        Literally just what chat gave me, lets see if this works
        '''
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(22*20*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3) # Pred XYZ
        
    def forward(self, x,normStateIn,normStateOut):
        assert normStateIn  == 'Net', 'AngleEstimator only works in Net Normalisation'
        assert normStateOut == 'Net', 'AngleEstimator only works in Net Normalisation'

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output between -1 and 1
        
        return x,normStateOut
