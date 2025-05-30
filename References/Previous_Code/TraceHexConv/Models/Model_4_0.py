##############################################################
#                        Simple Model                        #
#               Reduced for growing , No Hexag               #
#            Need to learn Xmax, complexity 0                #
##############################################################




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

# Define the custom Datastructures and loss and validation functions, etc

def Loss(y_pred, y_true,coeffs):

    E_loss = coeffs[0]*torch.nn.functional.mse_loss(y_pred[0],y_true[0])
    C_loss = coeffs[1]*torch.nn.functional.mse_loss(y_pred[1],y_true[1])
    A_loss = coeffs[2]*torch.nn.functional.mse_loss(y_pred[2],y_true[2])
    X_loss = coeffs[3]*torch.nn.functional.mse_loss(y_pred[3],y_true[3])

    T_Loss = E_loss + C_loss + A_loss+ X_loss

    return T_Loss, E_loss, C_loss, A_loss, X_loss

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.targets[0])
    def __getitem__(self, index):
        
        D_main = self.data[0][index]
        if D_main.shape[0]==1: D_main = D_main.squeeze(0)
        
        D_aux = self.data[1][index]
        if len(D_aux.shape)==2: D_aux = D_aux.unsqueeze(0)
        
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        y4 = self.targets[3][index]
        
        return D_main,D_aux, y1, y2, y3, y4



def validate(model, dataloader_val, Loss_function, model_Coefficients,device='cuda'):
    model.eval()
    val_T_loss = 0 
    val_E_loss = 0 
    val_C_loss = 0 
    val_A_loss = 0
    val_X_loss = 0

    with torch.no_grad():
        for batchD_main,batchD_aux, batchlogE,batchCore,batchAxis,batchXmax in dataloader_val:
            batchD_main = batchD_main.to(device)
            batchD_aux = batchD_aux.to(device)
            batchlogE = batchlogE.to(device)
            batchCore = batchCore.to(device)
            batchAxis = batchAxis.to(device)
            batchXmax = batchXmax.to(device)

            predictions = model(batchD_main,batchD_aux)
            
            T_loss,E_loss,C_loss,A_loss,X_loss = Loss_function(predictions,(batchlogE,batchCore,batchAxis,batchXmax),coeffs = model_Coefficients)
            
            val_T_loss += T_loss.item()
            val_E_loss += E_loss.item()
            val_C_loss += C_loss.item()
            val_A_loss += A_loss.item()
            val_X_loss += X_loss.item()
            # break
    val_T_loss /= len(dataloader_val)
    val_E_loss /= len(dataloader_val)
    val_C_loss /= len(dataloader_val)
    val_A_loss /= len(dataloader_val)
    val_X_loss /= len(dataloader_val)

    return val_T_loss, val_E_loss, val_C_loss, val_A_loss, val_X_loss

def Training_Track(model,dataloader,device = 'cuda'):
    return 'Nothing To Track'

def SlapAGraph(model,dataloader,EpochN,device = 'cuda'):
    
    model.eval()
    TotalSize = len(dataloader.dataset)

    Result_Pred_E = np.zeros((TotalSize,1))
    Result_Pred_C = np.zeros((TotalSize,2))
    Result_Pred_A = np.zeros((TotalSize,3))
    Result_Pred_X = np.zeros((TotalSize,1))

    Result_True_E = np.zeros((TotalSize,1))
    Result_True_C = np.zeros((TotalSize,2))
    Result_True_A = np.zeros((TotalSize,3))
    Result_True_X = np.zeros((TotalSize,1))

    filled = 0
    with torch.no_grad():
        for batchD_main,batchD_aux, batchlogE,batchCore,batchAxis,batchXmax in dataloader:

            batchD_main = batchD_main.to(device)
            batchD_aux = batchD_aux.to(device)
            CurrentBatchSize = batchD_main.shape[0]
            predictions = model(batchD_main,batchD_aux,Unnorm = True)
            truths      = model.Unnormalise(batchlogE,batchCore,batchAxis,batchXmax)

            Result_Pred_E[filled:filled+CurrentBatchSize] = predictions[0].cpu().numpy()
            Result_Pred_C[filled:filled+CurrentBatchSize] = predictions[1].cpu().numpy()
            Result_Pred_A[filled:filled+CurrentBatchSize] = predictions[2].cpu().numpy()
            Result_Pred_X[filled:filled+CurrentBatchSize] = predictions[3].cpu().numpy()

            Result_True_E[filled:filled+CurrentBatchSize] = truths[0].cpu().numpy()
            Result_True_C[filled:filled+CurrentBatchSize] = truths[1].cpu().numpy()
            Result_True_A[filled:filled+CurrentBatchSize] = truths[2].cpu().numpy()
            Result_True_X[filled:filled+CurrentBatchSize] = truths[3].cpu().numpy()



            filled += CurrentBatchSize
    PredList = (Result_Pred_E,Result_Pred_C,Result_Pred_A,Result_Pred_X)
    TrueList = (Result_True_E,Result_True_C,Result_True_A,Result_True_X)

    Dir   = '/remote/tychodata/ftairli/work/Projects/Temp/Graphs/'
    for k,(Val,Size) in enumerate(zip(['Energy','Core','Axis','Xmax'],[1,2,3,1])):
        subdir = Dir + Val + '/'
        if not os.path.exists(subdir): os.makedirs(subdir, exist_ok=True)
        title = model.Name + ' '+ Val + ' EpochN ' + str(EpochN)
        
        true_values_all = TrueList[k]
        pred_values_all = PredList[k]
        

        fig,ax = plt.subplots(1,Size,figsize=(Size*5,5))
        if Size==1: ax = [ax]

        for i in range(Size):
            true_values = true_values_all[:, i]
            pred_values = pred_values_all[:, i]

            if true_values.size == 0 or np.isnan(true_values).all():
                continue  # Skip this iteration if true_values is empty or all NaN

            ax[i].scatter(true_values, pred_values,marker = '.')
            min_val, max_val = min(true_values), max(true_values)
            ax[i].plot([min_val, max_val], [min_val, max_val], 'r')

        fig.suptitle(title)
        fig.savefig(subdir+title+'.png')

            # break







# Define custom Model stuff

class Recurrent_Block(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, num_layers=1, dropout_rate=0, num_features=10):
        super(Recurrent_Block, self).__init__()

        # Bidirectional LSTM layer
        self.bidirectional_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)

        # Linear layer to transform output to desired number of features
        self.fc = nn.Linear(hidden_dim, num_features)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.bidirectional_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # input shape: (batch_size, sequence_length, num_channels, width, height)

        batch_size, sequence_length, num_channels, width, height = x.shape

        
        # rearrange input to shape: (batch_size*width*height, sequence_length, num_channels)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, sequence_length, num_channels)

        # pass data through Bidirectional LSTM layer
        bidir_lstm_out, _ = self.bidirectional_lstm(x)  # output shape: (batch_size*width*height, sequence_length, hidden_dim*2)

        # pass data through LSTM layers
        lstm_out, _ = self.lstm(bidir_lstm_out)  # output shape: (batch_size*width*height, sequence_length, hidden_dim)

        # apply linear layer to every time step
        features = self.fc(lstm_out[:, -1, :])  # output shape: (batch_size*width*height, num_features)

        # reshape features to original width and height, shape: (batch_size, height, width, num_features)
        features = features.view(batch_size, -1, width, height)

        return features


# Define Custom Task Block one for each task
class Task_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Task_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=0,groups=in_channels//2) # (N,12,11,11) -> (N,12,9,9)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=0) # (N,12,9,9) -> (N,12,7,7)
        

        self.FC1 = nn.Linear(7*7*in_channels//4 , 64)
        self.FC2 = nn.Linear(64, 32)
        self.FC3 = nn.Linear(32, out_channels)

        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.contiguous().reshape(x.size(0), -1)
        # print(x.shape)
        x = self.FC1(x)
        # print(x.shape)
        x = self.FC2(x)
        # print(x.shape)
        x = self.FC3(x)
        # print(x.shape)
        
        return x



class Model_4_0(nn.Module):

    def __init__(self):
        super(Model_4_0, self).__init__()

        # Info
        self.Name = 'Model_4_0'
        self.Description ='''
        Model with like no complexity, will be grown to Model_4_..
        Try using convolutions instead of RNNs, for testing.
        No Hexagonal Convolution
        No Residual Blocks
        '''

        # History
        self.T_Loss_history = []
        self.E_Loss_history = []
        self.X_Loss_history = []
        self.C_Loss_history = []
        self.A_Loss_history = []

        self.T_Loss_history_val = []
        self.E_Loss_history_val = []
        self.X_Loss_history_val = []
        self.C_Loss_history_val = []
        self.A_Loss_history_val = []

        self.E_MSE_history = []
        self.C_MSE_history = []
        self.A_MSE_history = []
        self.X_MSE_history = []

        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,1,1,1]
        # self.LossCoefficients = [1,0,0,1]
        

        # Layers
        self.conv1 = nn.Conv3d(3, 1, kernel_size=(10, 3, 3), stride=(10, 1, 1), padding=(0, 1, 1))  # (N, 3, 120, 11, 11) -> (N, 1, 12, 11, 11)
        
        
        self.Energy = Task_Block(in_channels=13, out_channels=1)
        self.Core   = Task_Block(in_channels=13, out_channels=2)
        self.Axis   = Task_Block(in_channels=13, out_channels=3)
        self.Xmax   = Task_Block(in_channels=13, out_channels=1)
    
    def Unnormalise(E,C,A,X):
        # Unnormalising Data
        E_n = E=19
        C_n = C*750
        A_n = A
        X_n = X*66.80484050442804 +750
        return(E_n,C_n,A_n,X_n)

    def forward(self, traces, arrival,Unnorm=False):
        out = self.conv1(traces.permute(0,2,1,3,4))
        # Concatenate 'arrival' along the channel dimension
        out = torch.cat((out, arrival.unsqueeze(1)), dim=2).permute(0,2,1,3,4)
        
        Energy = self.Energy(out)
        Core   = self.Core(out)
        Axis   = self.Axis(out)
        Xmax   = self.Xmax(out)
        
        if Unnorm:
            return self.Unnormalise(Energy,Core,Axis,Xmax)
        else:
            return Energy,Core,Axis,Xmax


class Model_4_1(nn.Module):

    def __init__(self):
        super(Model_4_1, self).__init__()

        # Info
        self.Name = 'Model_4_1'
        self.Description ='''
        Model with like no complexity, will be grown to Model_4_..
        Funk Convolutions in time domain, small RNN back pls
        No Hexagonal Convolution
        No Residual Blocks
        '''

        # History
        self.T_Loss_history = []
        self.E_Loss_history = []
        self.X_Loss_history = []
        self.C_Loss_history = []
        self.A_Loss_history = []

        self.T_Loss_history_val = []
        self.E_Loss_history_val = []
        self.X_Loss_history_val = []
        self.C_Loss_history_val = []
        self.A_Loss_history_val = []

        self.E_MSE_history = []
        self.C_MSE_history = []
        self.A_MSE_history = []
        self.X_MSE_history = []

        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,0,0,0]
        # self.LossCoefficients = [1,0,0,1]
        
        self.Nfeat = 11

        self.RNN = Recurrent_Block(input_dim=3, hidden_dim=self.Nfeat, num_layers=1, dropout_rate=0, num_features=self.Nfeat)
        # Layers

        self.conv1 = nn.Conv2d(in_channels = self.Nfeat+1, out_channels = self.Nfeat+1, kernel_size = 3, stride = 1, padding = 0,groups = self.Nfeat+1) # (N,12,11,11) -> (N,12,9,9)
        self.conv2 = nn.Conv2d(in_channels = self.Nfeat+1, out_channels = self.Nfeat+1, kernel_size = 3, stride = 1, padding = 0,groups = self.Nfeat+1) # (N,12,9,9) -> (N,12,7,7)

        
        self.Energy = Task_Block(in_channels=self.Nfeat+1, out_channels=1)
        self.Core   = Task_Block(in_channels=self.Nfeat+1, out_channels=2)
        self.Axis   = Task_Block(in_channels=self.Nfeat+1, out_channels=3)
        self.Xmax   = Task_Block(in_channels=self.Nfeat+1, out_channels=1)
    
    def Unnormalise(self,E,C,A,X):
        # Unnormalising Data
        E_n = E+19
        C_n = C*750
        A_n = A
        X_n = X*66.80484050442804 +750
        return(E_n,C_n,A_n,X_n)

    def forward(self, traces, arrival,Unnorm=False):
        out = self.RNN(traces)
        out = torch.cat((out, arrival), dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        # Concatenate 'arrival' along the channel dimension

        Energy = self.Energy(out)
        Core   = self.Core(out)
        Axis   = self.Axis(out)
        Xmax   = self.Xmax(out)
        
        if Unnorm:
            return self.Unnormalise(Energy,Core,Axis,Xmax)
        else:
            return Energy,Core,Axis,Xmax



class Model_4_2(nn.Module):

    def __init__(self):
        super(Model_4_2, self).__init__()

        # Info
        self.Name = 'Model_4_2'
        self.Description ='''
        Try RNN with only TaskBlocks 
        '''

        # History
        self.T_Loss_history = []
        self.E_Loss_history = []
        self.X_Loss_history = []
        self.C_Loss_history = []
        self.A_Loss_history = []

        self.T_Loss_history_val = []
        self.E_Loss_history_val = []
        self.X_Loss_history_val = []
        self.C_Loss_history_val = []
        self.A_Loss_history_val = []

        self.E_MSE_history = []
        self.C_MSE_history = []
        self.A_MSE_history = []
        self.X_MSE_history = []

        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,1,1,1]
        # self.LossCoefficients = [1,0,0,1]
        
        self.Nfeat = 11

        self.RNN = Recurrent_Block(input_dim=3, hidden_dim=self.Nfeat, num_layers=1, dropout_rate=0, num_features=self.Nfeat)
        # Layers


        
        self.Energy = Task_Block(in_channels=self.Nfeat+1, out_channels=1)
        self.Core   = Task_Block(in_channels=self.Nfeat+1, out_channels=2)
        self.Axis   = Task_Block(in_channels=self.Nfeat+1, out_channels=3)
        self.Xmax   = Task_Block(in_channels=self.Nfeat+1, out_channels=1)
    
    def Unnormalise(self,E,C,A,X):
        # Unnormalising Data
        E_n = E+19
        C_n = C*750
        A_n = A
        X_n = X*66.80484050442804 +750
        return(E_n,C_n,A_n,X_n)

    def forward(self, traces, arrival,Unnorm=False):
        out = self.RNN(traces)
        out = torch.cat((out, arrival), dim=1)
        # Concatenate 'arrival' along the channel dimension

        Energy = self.Energy(out)
        Core   = self.Core(out)
        Axis   = self.Axis(out)
        Xmax   = self.Xmax(out)
        
        if Unnorm:
            return self.Unnormalise(Energy,Core,Axis,Xmax)
        else:
            return Energy,Core,Axis,Xmax

