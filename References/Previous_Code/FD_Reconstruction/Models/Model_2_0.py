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



# Absolute Loss
def Loss(Pred,Truth,normState = 'Net'):
    # Check that shapes match
    assert Pred.shape[1:] == torch.Size([2]), 'Pred shape is {}'.format(Pred.shape)
    assert Truth.shape[1:] == torch.Size([2]), 'Truth shape is {}'.format(Truth.shape)
    assert Pred.dtype == Truth.dtype, 'Pred and Truth dtypes do not match'    
    assert Pred.shape == Truth.shape , 'Shapes dont Match'
    assert torch.isfinite(Pred).sum().item() == Pred.numel(), 'Pred has NaNs or Infs'
    assert torch.isfinite(Truth).sum().item() == Truth.numel(), 'Truth has NaNs or Infs'
    assert normState == 'Net', 'normState must be Net'
    
    # Calculate MSE Loss
    criterion = nn.L1Loss()
    PhiLoss   = criterion(Pred[:,0],Truth[:,0])
    # PhiLoss = torch.tensor(0.0).to(Pred.device)
    ThetaLoss = criterion(Pred[:,1],Truth[:,1])
    loss = PhiLoss+ThetaLoss
    
    return loss, PhiLoss, ThetaLoss

    



def validate(model,dataloader,Loss,device = 'cuda',normState = 'Net'):
    model.eval()
    val_loss = 0
    val_loss_Phi = 0
    val_loss_Theta = 0

    with torch.no_grad():
        for Main, Truth in dataloader:
            Main = Main.to(device)
            Truth = Truth.to(device)
            Y_pred,normState = model(Main,normStateIn = 'Net',normStateOut = normState)
            loss,lossPhi,lossTheta = Loss(Y_pred,Truth,normState)
            val_loss += loss.item()
            val_loss_Phi += lossPhi.item()
            val_loss_Theta += lossTheta.item()

            # val_loss,val_loss_Phi,val_loss_Theta += Loss(Y_pred,Truth,normState).item()
    return val_loss/len(dataloader),val_loss_Phi/len(dataloader),val_loss_Theta/len(dataloader)



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Main, Truth):
        self.Main = Main
        self.Truth = Truth
    def __len__(self):
        return len(self.Main)
    def __getitem__(self, idx):
        return self.Main[idx,:,:,:], self.Truth[idx,:]
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x




class Model_2_0(nn.Module):

    def __init__(self,in_channels = 2,N_filters = 64,NDenseNodes = 256,DropOut_Rate = 0.5,Dtype=torch.float32):
        assert N_filters % 4 ==0 , 'N_filters must be a multiple of 4'
        assert NDenseNodes % 2 ==0 , 'NDenseNodes must be a multiple of 2'

        super(Model_2_0, self).__init__()
        # Info
        self.Name = 'Model_2_0'
        self.Description = '''
        Try using Hexagonal Convolutions instead of regular convolutions here, see if it imporoves the general case prediction
        Initialisation of weights is not implimented automatically, dtype is not implemented automatically
        Will have to adjust the size to not be 20x20, but 20x22, as padding is not implemented automatically
        '''
        # Layers
        # Initial Scan by these layers (Add more later, maybe)
        self.Scan1 = hexagdly.Conv2d(in_channels=in_channels, out_channels=N_filters, kernel_size=2, stride=1) # Outputs 20x22
        
        self.conv1 = hexagdly.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=1, stride=1) # Outputs 20x22
        self.conv2 = hexagdly.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=1, stride=1) # Outputs 20x22
        self.conv3 = hexagdly.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=1, stride=1) # Outputs 20x22
        self.conv4 = hexagdly.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=1, stride=1) # Outputs 20x22
        self.conv5 = hexagdly.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=1, stride=1) # Outputs 20x22

        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*22*N_filters,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(20*22*N_filters,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        self.FinalActivation  = Identity()
        
        self.ConvDropout = nn.Dropout2d(DropOut_Rate)
        self.DenseDropout = nn.Dropout(DropOut_Rate)


        # # Initialize weights
        # nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')

        nn.init.kaiming_normal_(self.PhiDense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.PhiDense2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.PhiDense3.weight, nonlinearity='leaky_relu')

        nn.init.kaiming_normal_(self.ThetaDense1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.ThetaDense2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.ThetaDense3.weight, nonlinearity='leaky_relu')


    def forward(self, Main,normStateIn = 'Net',normStateOut = 'Net'):
        if normStateIn == 'Val': # Need to normalise before we go
            Main[:,0,:,:] = MD.PixTime_to_net(Main[:,0,:,:])
            Main[:,1,:,:] = MD.PixSig_to_net(Main[:,1,:,:])
        # Rotate by 90 degrees to get the correct hexagdly orientation
        Main = torch.rot90(Main,k=1,dims=(2,3))

        # CANCEL THE PAD
        # Hexagdly doesnt pad, need to pad Main with zeros
        # Main = F.pad(Main,(2,2,1,1),'constant',0)

        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Conv
        out = self.ConvActivation(self.conv1(out))
        out = self.ConvActivation(self.conv2(out))
        out = self.ConvActivation(self.conv3(out))
        out = self.ConvActivation(self.conv4(out))
        out = self.ConvActivation(self.conv5(out))
        
        # Flatten
        out = out.view(out.shape[0],-1)
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)

        # UnNormalise
        if normStateOut == 'Val':
            Phi = MD.Phi_to_val(Phi)
            Theta = MD.Theta_to_val(Theta)

        return torch.cat((Phi,Theta),dim=1),normStateOut
    
