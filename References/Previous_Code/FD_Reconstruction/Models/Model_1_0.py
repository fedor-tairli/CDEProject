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



# MSE Loss
# def Loss(Pred,Truth,normState = 'Net'):
#     # Check that shapes match
#     assert Pred.shape[1:] == torch.Size([2]), 'Pred shape is {}'.format(Pred.shape)
#     assert Truth.shape[1:] == torch.Size([2]), 'Truth shape is {}'.format(Truth.shape)
#     assert Pred.dtype == Truth.dtype, 'Pred and Truth dtypes do not match'    
#     assert Pred.shape == Truth.shape , 'Shapes dont Match'
#     assert torch.isfinite(Pred).sum().item() == Pred.numel(), 'Pred has NaNs or Infs'
#     assert torch.isfinite(Truth).sum().item() == Truth.numel(), 'Truth has NaNs or Infs'
#     assert normState == 'Net', 'normState must be Net'
    
#     # Calculate MSE Loss
#     criterion = nn.MSELoss()
#     PhiLoss   = criterion(Pred[:,0],Truth[:,0])
#     # PhiLoss = torch.tensor(0.0).to(Pred.device)
#     ThetaLoss = criterion(Pred[:,1],Truth[:,1])
#     loss = PhiLoss+ThetaLoss
    
#     return loss, PhiLoss, ThetaLoss


# # Cos of Deviation Loss
# def Loss(Pred,Truth,normState = 'Net'):
#     # Check that shapes match
#     assert Pred.shape[1:] == torch.Size([2]), 'Pred shape is {}'.format(Pred.shape)
#     assert Truth.shape[1:] == torch.Size([2]), 'Truth shape is {}'.format(Truth.shape)
#     assert Pred.dtype == Truth.dtype, 'Pred and Truth dtypes do not match'    
#     assert Pred.shape == Truth.shape , 'Shapes dont Match'
#     assert torch.isfinite(Pred).sum().item() == Pred.numel(), 'Pred has NaNs or Infs'
#     assert torch.isfinite(Truth).sum().item() == Truth.numel(), 'Truth has NaNs or Infs'
#     if normState == 'Net':
#         PhiPred = MD.Phi_to_val(Pred[:,0])
#         PhiTruth = MD.Phi_to_val(Truth[:,0])

#         ThetaPred = MD.Theta_to_val(Pred[:,1])
#         ThetaTruth = MD.Theta_to_val(Truth[:,1])
    
#     PhiLoss   = torch.mean(1.0-torch.cos(PhiPred-PhiTruth))
#     ThetaLoss = torch.mean(1.0-torch.cos(ThetaPred-ThetaTruth))

#     loss = ThetaLoss+PhiLoss

#     return loss, PhiLoss, ThetaLoss


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

class Model_1_0(nn.Module):

    def __init__(self,in_channels = 2,N_filters = 32,NDenseNodes = 128,Dtype=torch.float16):
        assert N_filters % 4 ==0 , 'N_filters must be a multiple of 4'
        assert NDenseNodes % 2 ==0 , 'NDenseNodes must be a multiple of 2'

        super(Model_1_0, self).__init__()
        # Info
        self.Name = 'Model_1_0'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta
        Introduce the normalisation state variable to keep track of the normalisation state and adjust on the fly if needed
        '''
        # Layers
        # Initial Scan by these layers (Add more later, maybe)
        self.Scan1 = nn.Conv2d(in_channels=in_channels, out_channels=N_filters, kernel_size=5, stride=1, padding=(1,2),dtype=Dtype) # Outputs 20x20
        
        self.conv1 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv2 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv3 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20

        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        self.FinalActivation  = Identity()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        

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


        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Conv
        out = self.ConvActivation(self.conv1(out))
        out = self.ConvActivation(self.conv2(out))
        out = self.ConvActivation(self.conv3(out))
        # Flatten
        out = out.view(out.shape[0],-1)
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        # Phi = Phi*0
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)

        # UnNormalise
        if normStateOut == 'Val':
            Phi = MD.Phi_to_val(Phi)
            Theta = MD.Theta_to_val(Theta)

        return torch.cat((Phi,Theta),dim=1),normStateOut



class Model_1_1(nn.Module):

    def __init__(self,in_channels = 2,N_filters = 32,NDenseNodes = 128,Dtype=torch.float16):
        assert N_filters % 4 ==0 , 'N_filters must be a multiple of 4'
        assert NDenseNodes % 2 ==0 , 'NDenseNodes must be a multiple of 2'

        super(Model_1_1, self).__init__()
        # Info
        self.Name = 'Model_1_1'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta
        Use simplified model as the one above produces GIGALOSS
        Introduce the normalisation state variable to keep track of the normalisation state and adjust on the fly if needed
        '''
        # Layers
        # Initial Scan by these layers (Add more later, maybe)
        self.Scan1 = nn.Conv2d(in_channels=in_channels, out_channels=N_filters, kernel_size=5, stride=1, padding=(1,2),dtype=Dtype) # Outputs 20x20
        
        # Dense Layers for prediction
        self.Dense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Dense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        
        

    def forward(self, Main,normStateIn = 'Net',normStateOut = 'Net'):
        if normStateIn == 'Val': # Need to normalise before we go
            Main[:,0,:,:] = MD.PixTime_to_net(Main[:,0,:,:])
            Main[:,1,:,:] = MD.PixSig_to_net(Main[:,1,:,:])


        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Flatten
        out = out.view(out.shape[0],-1)
        # Predict
        out = self.DenseActivation(self.Dense1(out))
        out = self.DenseActivation(self.Dense2(out))
        out = self.Dense3(out)
        
        # UnNormalise
        if normStateOut == 'Val':
            out[:,0] = MD.Phi_to_val(out[:,0])
            out[:,1] = MD.Theta_to_val(out[:,1])

        return out,normStateOut


class Model_1_2(nn.Module):

    def __init__(self,in_channels = 2,N_filters = 32,NDenseNodes = 128,Dtype=torch.float16):
        assert N_filters % 4 ==0 , 'N_filters must be a multiple of 4'
        assert NDenseNodes % 2 ==0 , 'NDenseNodes must be a multiple of 2'

        super(Model_1_2, self).__init__()
        # Info
        self.Name = 'Model_1_2'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta
        Introduce the normalisation state variable to keep track of the normalisation state and adjust on the fly if needed

        Looks like the above worked, Increasing the number of layers and filters
        Seems like this architecture is a disaster, something makes it unstable, too lazy to figure out.
        '''
        # Layers
        # Initial Scan by these layers (Add more later, maybe)
        self.Scan3 = nn.Conv2d(in_channels=in_channels  , out_channels=N_filters, kernel_size=3, stride=1, padding=(0,1),dtype=Dtype) # Outputs 20x20
        self.Scan5 = nn.Conv2d(in_channels=in_channels  , out_channels=N_filters, kernel_size=5, stride=1, padding=(1,2),dtype=Dtype) # Outputs 20x20

        self.conv5_1 = nn.Conv2d(in_channels=N_filters*2, out_channels=N_filters, kernel_size=5, stride=1, padding=(2,2),dtype=Dtype) # Outputs 20x20
        self.conv5_2 = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=5, stride=1, padding=(2,2),dtype=Dtype) # Outputs 20x20
        self.conv5_3 = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=5, stride=1, padding=(2,2),dtype=Dtype) # Outputs 20x20

        self.conv3_1 = nn.Conv2d(in_channels=N_filters*2, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv3_2 = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv3_3 = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20

        self.sum1    = nn.Conv2d(in_channels=N_filters*2, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.sum2    = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.sum3    = nn.Conv2d(in_channels=N_filters  , out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20


        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        self.FinalActivation  = Identity()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.Scan3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.Scan5.weight, nonlinearity='leaky_relu')


        nn.init.kaiming_normal_(self.conv3_1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3_2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3_3.weight, nonlinearity='leaky_relu')

        nn.init.kaiming_normal_(self.conv5_1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5_2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5_3.weight, nonlinearity='leaky_relu')

        nn.init.kaiming_normal_(self.sum1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.sum2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.sum3.weight, nonlinearity='leaky_relu')

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


        # Scan
        out3 = self.ConvActivation(self.Scan3(Main))
        out5 = self.ConvActivation(self.Scan5(Main))
        # Concatenate

        out = torch.cat((out3,out5),dim=1)
        
        # Conv
        out3 = self.ConvActivation(self.conv3_1(out))
        out3 = self.ConvActivation(self.conv3_2(out3))
        out3 = self.ConvActivation(self.conv3_3(out3))

        out5 = self.ConvActivation(self.conv5_1(out))
        out5 = self.ConvActivation(self.conv5_2(out5))
        out5 = self.ConvActivation(self.conv5_3(out5))

        # Summarise
        out = torch.cat((out3,out5),dim=1)
        out = self.ConvActivation(self.sum1(out))
        out = self.ConvActivation(self.sum2(out))
        out = self.ConvActivation(self.sum3(out))

        # Flatten
        out = out.view(out.shape[0],-1)
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        # Phi = Phi*0
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)

        # UnNormalise
        if normStateOut == 'Val':
            Phi = MD.Phi_to_val(Phi)
            Theta = MD.Theta_to_val(Theta)

        return torch.cat((Phi,Theta),dim=1),normStateOut




class Model_1_3(nn.Module):

    def __init__(self,in_channels = 2,N_filters = 64,NDenseNodes = 256,DropOut_Rate = 0.5,Dtype=torch.float16):
        assert N_filters % 4 ==0 , 'N_filters must be a multiple of 4'
        assert NDenseNodes % 2 ==0 , 'NDenseNodes must be a multiple of 2'

        super(Model_1_3, self).__init__()
        # Info
        self.Name = 'Model_1_3'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta
        Introduction of Dropout Regularisation in order to allow for larger conv and dense layer sizes
        Also Add depth to the network with 2 more conv layers
        '''
        # Layers
        # Initial Scan by these layers (Add more later, maybe)
        self.Scan1 = nn.Conv2d(in_channels=in_channels, out_channels=N_filters, kernel_size=5, stride=1, padding=(1,2),dtype=Dtype) # Outputs 20x20
        
        self.conv1 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv2 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv3 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv4 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20
        self.conv5 = nn.Conv2d(in_channels=N_filters, out_channels=N_filters, kernel_size=3, stride=1, padding=(1,1),dtype=Dtype) # Outputs 20x20

        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_filters,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        self.FinalActivation  = Identity()
        
        self.ConvDropout = nn.Dropout2d(DropOut_Rate)
        self.DenseDropout = nn.Dropout(DropOut_Rate)


        # Initialize weights
        nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')

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


        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Conv
        out = self.ConvActivation(self.conv1(out))
        out = self.ConvDropout(out)
        out = self.ConvActivation(self.conv2(out))
        out = self.ConvDropout(out)
        out = self.ConvActivation(self.conv3(out))
        out = self.ConvDropout(out)
        out = self.ConvActivation(self.conv4(out))
        out = self.ConvDropout(out)
        out = self.ConvActivation(self.conv5(out))
        out = self.ConvDropout(out)

        # Flatten
        out = out.view(out.shape[0],-1)
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseDropout(Phi)
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.DenseDropout(Phi)
        Phi = self.PhiDense3(Phi)
        
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseDropout(Theta)
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.DenseDropout(Theta)
        Theta = self.ThetaDense3(Theta)

        # UnNormalise
        if normStateOut == 'Val':
            Phi = MD.Phi_to_val(Phi)
            Theta = MD.Theta_to_val(Theta)

        return torch.cat((Phi,Theta),dim=1),normStateOut
    
