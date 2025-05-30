# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
# from   torch_geometric.nn import GCNConv, TAGConv,GATConv
# from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
# from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['SDPTheta','SDPPhi'],ReturnTensor = True):

    '''
    Takes Truth,Pred in form -> [Theta,Phi] 
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
    losses['Total'] = sum(losses.values())
    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses


def validate(model,Dataset,Loss,device,BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _,BatchMains,BatchAux,BatchTruth,_, BatchGraph in Dataset:
            Predictions = model(BatchMains,BatchAux) if BatchGraph is None else model(BatchMains,BatchAux,BatchGraph)
            Preds .append(Predictions.to('cpu'))
            Truths.append(BatchTruth .to('cpu'))

        Preds  = torch.cat(Preds ,dim=0)
        Truths = torch.cat(Truths,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss(Preds,Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)
    

def metric(model,Dataset,device,keys = ['SDPTheta','SDPPhi'],BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the 68% containment range of the angular deviation
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _,BatchMains,BatchAux,BatchTruth,_,BatchGraph in Dataset:
            Predictions = model(BatchMains,BatchAux) if BatchGraph is None else model(BatchMains,BatchAux,BatchGraph)
            Preds .append(Predictions.to('cpu'))
            Truths.append(BatchTruth .to('cpu'))

    Preds  = torch.cat(Preds ,dim=0)
    Truths = torch.cat(Truths,dim=0)
    Preds  = Dataset.Unnormalise_Truth(Preds )
    Truths = Dataset.Unnormalise_Truth(Truths)

    Units = Dataset.Truth_Units
    metrics = {}
    for i,key in enumerate(keys):
        if Units[i] == 'rad':
            AngDiv = torch.atan2(torch.sin(Preds[:,i]-Truths[:,i]),torch.cos(Preds[:,i]-Truths[:,i]))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)
        if Units[i] == 'deg':
            AngDiv = torch.atan2(torch.sin(torch.deg2rad(Preds[:,i]-Truths[:,i])),torch.cos(torch.deg2rad(Preds[:,i]-Truths[:,i])))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)*180/torch.pi
    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return metrics

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x



# Define the model
class Model_SDP_Conv_0(nn.Module):
    Name = 'Model_SDP_Conv_0'
    Description = '''
    Simple Conv2D Model for SDP Reconstruction
    3x 5x5 and 3x 3x3 Conv2d Layers followed by 3x 3x3 Conv2d Layers into 6x Fully Connected Layers
    '''
    def __init__(self, in_channels=1, in_channels_aux=1, out_channels=1, N_kernels = 16, N_dense_nodes=128, init_type = None,dtype = torch.float32):
        # Input is 20x22 grid
        super(Model_SDP_Conv_0,self).__init__()
        # in_Channels should be a tuple of the number of channels for each Main, Only have 1 main here
        in_channels = in_channels[0]
        # 5x5 Conv2d
        self.Conv5_1 = nn.Conv2d(in_channels,N_kernels,kernel_size=5,stride=1,padding=(2,1))
        self.Conv5_2 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))
        self.Conv5_3 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2)) # Padding here reduces the size to 20x20

        self.Conv3_1 = nn.Conv2d(in_channels,N_kernels,kernel_size=3,stride=1,padding=(1,0))
        self.Conv3_2 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=3,stride=1,padding=(1,1))
        self.Conv3_3 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=3,stride=1,padding=(1,1)) # Padding here reduces the size to 20x20

        self.ConvT_4 = nn.Conv2d(2*N_kernels,N_kernels,kernel_size=5,stride=1,padding=0) # Padding here reduces the size to 16x16
        self.ConvT_5 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=5,stride=1,padding=0) # Padding here reduces the size to 12x12
        self.ConvT_6 = nn.Conv2d(  N_kernels,N_kernels,kernel_size=5,stride=1,padding=0) # Padding here reduces the size to 8x8


            



        # Fully Connected Layers
        self.FC1 = nn.Linear(8*8*N_kernels,N_dense_nodes)
        self.FC2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        
        
        # Add Aux Layers Here
        self.FC4 = nn.Linear(N_dense_nodes+in_channels_aux,N_dense_nodes)
        self.FC5 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.FC6 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # Output Layer
        self.Phi   = nn.Linear(N_dense_nodes,1)
        self.Theta = nn.Linear(N_dense_nodes,1)

        # Activation Functions
        self.Conv_Activation = nn.LeakyReLU()
        self.FC_Activation   = nn.LeakyReLU()
        self.Out_Activation  = nn.Tanh()


        # Initialise the weights
        if init_type == 'kaiming':
            nn.init.kaiming_normal_(self.Conv5_1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Conv5_2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Conv5_3.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Conv3_1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Conv3_2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Conv3_3.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ConvT_4.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ConvT_5.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ConvT_6.weight, nonlinearity='leaky_relu')

            nn.init.kaiming_normal_(self.FC1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.FC2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.FC4.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.FC5.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.FC6.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Phi.weight, nonlinearity='tanh')
            nn.init.kaiming_normal_(self.Theta.weight, nonlinearity='tanh')


    def forward(self,Mains,Aux):
        # Assertions
        assert len(Mains) == 1, f'Expected 1 Main, got {len(Mains)}'
        assert Mains[0].shape[1:] == (1,20,22), f'Expected Main Shape (N,1,20,22), got {Mains[0].shape}'

        Main = Mains[0]
        Main = Main.to(self.Conv5_1.weight.device)
        # Convolutional Layers
        X5 = self.Conv_Activation(  self.Conv5_1(Main)  )
        X5 = self.Conv_Activation(  self.Conv5_2( X5 )  )
        X5 = self.Conv_Activation(  self.Conv5_3( X5 )  )

        X3 = self.Conv_Activation(  self.Conv3_1(Main)  )
        X3 = self.Conv_Activation(  self.Conv3_2( X3 )  )
        X3 = self.Conv_Activation(  self.Conv3_3( X3 )  )

        XT = torch.cat([X5,X3],dim=1)
        XT = self.Conv_Activation(  self.ConvT_4( XT )  )
        XT = self.Conv_Activation(  self.ConvT_5( XT )  )
        XT = self.Conv_Activation(  self.ConvT_6( XT )  )

        # Flatten
        XT = XT.view(XT.shape[0],-1)

        # Fully Connected Layers
        XT = self.FC_Activation(  self.FC1( XT )  )
        XT = self.FC_Activation(  self.FC2( XT )  )

        # Add Aux Layers Here
        XT = torch.cat([XT,Aux],dim=1)
        XT = self.FC_Activation(  self.FC4( XT )  )
        XT = self.FC_Activation(  self.FC5( XT )  )
        XT = self.FC_Activation(  self.FC6( XT )  )

        # Output Layer
        Phi   = self.Out_Activation(  self.Phi  ( XT )  )
        Theta = self.Out_Activation(  self.Theta( XT )  )

        return torch.cat([Theta,Phi],dim=1)


class Model_SDP_Conv_1(nn.Module):
    Name = 'Model_SDP_Conv_1'
    Description = '''
    Copy of what i was doing in FDReconstruction before. 
    Simple, Several Conv2D Layers followed by Fully Connected Layers
    '''
    def __init__(self, in_channels=(2,), in_channels_aux=1, out_channels=2, N_kernels = 64, N_dense_nodes=256, init_type = None,DropOut_rate=0.5,dtype = torch.float32): 
        super(Model_SDP_Conv_1,self).__init__()
        # Layers
        print(f'Expecting {in_channels[0]} channels')
        self.Scan1 = nn.Conv2d(in_channels=in_channels[0],out_channels=N_kernels,kernel_size=5,stride=1,padding=(2,1),dtype = dtype) # Should Return 20x20

        self.conv1 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv2 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv3 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv4 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv5 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20

        
        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.PhiDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.PhiDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.ThetaDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.ThetaDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ConvActivation = nn.LeakyReLU()
        self.FCActivation   = nn.LeakyReLU()
        self.OutActivation  = nn.Tanh()

        self.ConvDropout = nn.Dropout2d(DropOut_rate)
        self.DenseDropout   = nn.Dropout(DropOut_rate)

        # Initialise weights
        if init_type == 'kaiming':
            print('Initialising kaiming weights')
            nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense3.weight, nonlinearity='tanh')
            nn.init.kaiming_normal_(self.ThetaDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense3.weight, nonlinearity='tanh')


    def forward(self,Mains,Aux):
        # Does Not use Aux and Only Takes one Main
        # Assertions
        assert len(Mains) == 1, f'Expected 1 Main, got {len(Mains)}'
        assert Mains[0].shape[2:] == (20,22), f'Expected Main Shape (N,C,20,22), got {Mains[0].shape}'

        Main = Mains[0]
        Main = Main.to(self.Scan1.weight.device)
        
        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Convolutional Layers
        out = self.ConvDropout(self.ConvActivation(self.conv1(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv2(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv3(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv4(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv5(out)))
        
        # Flatten
        out = out.view(out.shape[0],-1)
        
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense1(out)))
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense2(Theta)))
        Theta = self.OutActivation(self.ThetaDense3(Theta))

        Phi = self.DenseDropout(self.FCActivation(self.PhiDense1(out)))
        Phi = self.DenseDropout(self.FCActivation(self.PhiDense2(Phi)))
        Phi = self.OutActivation(self.PhiDense3(Phi))
        
        return torch.cat([Theta,Phi],dim=1)

        
class Model_SDP_Conv_2(nn.Module):
    Name = 'Model_SDP_Conv_2'
    Description = '''
    Copy of what i was doing in FDReconstruction before. 
    Simple, Several Conv2D Layers followed by Fully Connected Layers
    Deeper than the _1 model
    '''
    def __init__(self, in_channels=(2,), in_channels_aux=1, out_channels=2, N_kernels = 64, N_dense_nodes=256, init_type = None,DropOut_rate=0.5,dtype = torch.float32): 
        super(Model_SDP_Conv_2,self).__init__()
        # Layers
        print(f'Expecting {in_channels[0]} channels')
        self.Scan1 = nn.Conv2d(in_channels=in_channels[0],out_channels=N_kernels,kernel_size=5,stride=1,padding=(2,1),dtype = dtype) # Should Return 20x20
        self.Scan2 = nn.Conv2d(in_channels=in_channels[0],out_channels=N_kernels,kernel_size=5,stride=1,padding=(2,1),dtype = dtype) # Should Return 20x20

        self.conv1 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv2 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv3 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv4 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv5 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20

        self.conv6 = nn.Conv2d(in_channels=N_kernels*2,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv7 = nn.Conv2d(in_channels=N_kernels  ,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv8 = nn.Conv2d(in_channels=N_kernels  ,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        
        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.PhiDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.PhiDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.ThetaDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.ThetaDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ConvActivation = nn.LeakyReLU()
        self.FCActivation   = nn.LeakyReLU()
        self.OutActivation  = nn.Tanh()

        self.ConvDropout = nn.Dropout2d(DropOut_rate)
        self.DenseDropout   = nn.Dropout(DropOut_rate)

        # Initialise weights
        if init_type == 'kaiming':
            print('Initialising kaiming weights')
            nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Scan2.weight, nonlinearity='leaky_relu')

            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')
            
            nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv8.weight, nonlinearity='leaky_relu')

            nn.init.kaiming_normal_(self.PhiDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense3.weight, nonlinearity='tanh')
            nn.init.kaiming_normal_(self.ThetaDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense3.weight, nonlinearity='tanh')


    def forward(self,Mains,Aux):
        # Does Not use Aux and Only Takes one Main
        # Assertions
        assert len(Mains) == 1, f'Expected 1 Main, got {len(Mains)}'
        assert Mains[0].shape[2:] == (20,22), f'Expected Main Shape (N,C,20,22), got {Mains[0].shape}'

        Main = Mains[0]
        Main = Main.to(self.Scan1.weight.device)
        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Convolutional Layers
        out = self.ConvDropout(self.ConvActivation(self.conv1(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv2(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv3(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv4(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv5(out)))
        
        out = torch.cat([out,self.ConvActivation(self.Scan2(Main))],dim=1)
        out = self.ConvDropout(self.ConvActivation(self.conv6(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv7(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv8(out)))
        
        # Flatten
        out = out.view(out.shape[0],-1)
        
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense1(out)))
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense2(Theta)))
        Theta = self.OutActivation(self.ThetaDense3(Theta))

        Phi = self.DenseDropout(self.FCActivation(self.PhiDense1(out)))
        Phi = self.DenseDropout(self.FCActivation(self.PhiDense2(Phi)))
        Phi = self.OutActivation(self.PhiDense3(Phi))
        
        return torch.cat([Theta,Phi],dim=1)


class Model_SDP_Conv_3(nn.Module):
    Name = 'Model_SDP_Conv_3'
    Description = '''
    Copy of what i was doing in FDReconstruction before. 
    Simple, Several Conv2D Layers followed by Fully Connected Layers
    Deeper than the _1 model
    Added a few more Conv Layers and Batch Normalisation
    
    '''
    def __init__(self, in_channels=(2,), in_channels_aux=1, out_channels=2, N_kernels = 64, N_dense_nodes=256, init_type = None,DropOut_rate=0.5,dtype = torch.float32): 
        super(Model_SDP_Conv_3,self).__init__()
        # Layers
        print(f'Expecting {in_channels[0]} channels')
        self.Scan1 = nn.Conv2d(in_channels=in_channels[0],out_channels=N_kernels,kernel_size=5,stride=1,padding=(2,1),dtype = dtype) # Should Return 20x20
        self.Scan2 = nn.Conv2d(in_channels=in_channels[0],out_channels=N_kernels,kernel_size=5,stride=1,padding=(2,1),dtype = dtype) # Should Return 20x20
        # self.BNScan= nn.BatchNorm2d(N_kernels)

        self.conv1 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv2 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv3 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.BN1   = nn.BatchNorm2d(N_kernels)

        self.conv4 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv5 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv6 = nn.Conv2d(in_channels=N_kernels,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype)
        self.BN2   = nn.BatchNorm2d(N_kernels)

        self.conv7 = nn.Conv2d(in_channels=N_kernels*2,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv8 = nn.Conv2d(in_channels=N_kernels  ,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.conv9 = nn.Conv2d(in_channels=N_kernels  ,out_channels=N_kernels,kernel_size=3,stride=1,padding=(1,1),dtype = dtype) # Should Return 20x20
        self.BN3   = nn.BatchNorm2d(N_kernels)


        # Dense Layers for prediction
        self.PhiDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.PhiDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.PhiDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ThetaDense1 = nn.Linear(20*20*N_kernels,N_dense_nodes,dtype=dtype)
        self.ThetaDense2 = nn.Linear(N_dense_nodes,N_dense_nodes,dtype=dtype)
        self.ThetaDense3 = nn.Linear(N_dense_nodes,1,dtype=dtype)

        self.ConvActivation = nn.LeakyReLU()
        self.FCActivation   = nn.LeakyReLU()
        self.OutActivation  = nn.Tanh()

        self.ConvDropout    = nn.Dropout2d(DropOut_rate)
        self.DenseDropout   = nn.Dropout(DropOut_rate)

        # Initialise weights
        if init_type == 'kaiming':
            print('Initialising kaiming weights')
            nn.init.kaiming_normal_(self.Scan1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.Scan2.weight, nonlinearity='leaky_relu')
            
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='leaky_relu')
            
            nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv8.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.conv9.weight, nonlinearity='leaky_relu')

            nn.init.kaiming_normal_(self.PhiDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.PhiDense3.weight, nonlinearity='tanh')

            nn.init.kaiming_normal_(self.ThetaDense1.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense2.weight, nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.ThetaDense3.weight, nonlinearity='tanh')


    def forward(self,Mains,Aux):
        # Does Not use Aux and Only Takes one Main
        # Assertions
        assert len(Mains) == 1, f'Expected 1 Main, got {len(Mains)}'
        assert Mains[0].shape[2:] == (20,22), f'Expected Main Shape (N,C,20,22), got {Mains[0].shape}'

        Main = Mains[0]
        Main = Main.to(self.Scan1.weight.device)
        # Scan
        out = self.ConvActivation(self.Scan1(Main))
        # Convolutional Layers
        out = self.ConvDropout(self.ConvActivation(self.conv1(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv2(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv3(out)))
        out = self.BN1(out)
        out = self.ConvDropout(self.ConvActivation(self.conv4(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv5(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv6(out)))
        out = self.BN2(out)

        out = torch.cat([out,self.ConvActivation(self.Scan2(Main))],dim=1)
        out = self.ConvDropout(self.ConvActivation(self.conv7(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv8(out)))
        out = self.ConvDropout(self.ConvActivation(self.conv9(out)))
        out = self.BN3(out)
        
        # Flatten
        out = out.view(out.shape[0],-1)
        
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense1(out)))
        Theta = self.DenseDropout(self.FCActivation(self.ThetaDense2(Theta)))
        Theta = self.OutActivation(self.ThetaDense3(Theta))

        Phi = self.DenseDropout(self.FCActivation(self.PhiDense1(out)))
        Phi = self.DenseDropout(self.FCActivation(self.PhiDense2(Phi)))
        Phi = self.OutActivation(self.PhiDense3(Phi))
        
        return torch.cat([Theta,Phi],dim=1)

