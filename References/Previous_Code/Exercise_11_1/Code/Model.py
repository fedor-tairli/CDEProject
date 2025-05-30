##############################################################
#                 Here we define the models                  #
#            We iterate with simple version numbers          #
##############################################################

# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Input shape is (200000,9,9,2)

class Model_1_0(nn.Module):

    def __init__(self):
        super(Model_1_0, self).__init__()


        # Info
        self.Name = 'Model_1_0'
        self.Description = 'Start Simple, Small 2D_Conv layers, into flatten, then split off  to dense into 3 outputs'
        self.LossCoefficients = [1,1,1] # Equal Ratio of errors, need to adjust. 
        
        # History
        self.T_Loss_hist = []
        self.E_Loss_hist = []
        self.C_Loss_hist = []
        self.A_Loss_hist = []
        self.T_Loss_hist_val = []
        self.E_Loss_hist_val = []
        self.C_Loss_hist_val = []
        self.A_Loss_hist_val = []


        # Layers
        # super(Model_1_0, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # 32,9,9
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16,9,9
        self.fc1 = nn.Linear(16*9*9, 64)

        # 3 Objectives Layers

        self.Efc1 = nn.Linear(64, 32)
        self.Efc2 = nn.Linear(32, 1)

        self.Cfc1 = nn.Linear(64, 32)
        self.Cfc2 = nn.Linear(32, 2)

        self.Afc1 = nn.Linear(64, 32)
        self.Afc2 = nn.Linear(32, 3)

    def forward(self, x):
        # Convolutional Layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.reshape(-1, 16*9*9)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        # 3 Objectives Layers

        E = F.relu(self.Efc1(x))
        E = self.Efc2(E)

        C = F.relu(self.Cfc1(x))
        C = self.Cfc2(C)

        A = F.relu(self.Afc1(x))
        A = self.Afc2(A)

        return (E,C,A)
     

class Model_1_1(nn.Module):
    def __init__(self):
        super(Model_1_1, self).__init__()


        # Info
        self.Name = 'Model_1_1'
        self.Description = 'Change up the Loss Coefficients'
        self.LossCoefficients = [1,1*10**(-4),1] # E,C,A,Equal Ratio of errors, need to adjust. 
        
        # History
        self.T_Loss_hist = []
        self.E_Loss_hist = []
        self.C_Loss_hist = []
        self.A_Loss_hist = []
        self.T_Loss_hist_val = []
        self.E_Loss_hist_val = []
        self.C_Loss_hist_val = []
        self.A_Loss_hist_val = []


        # Layers
        # super(Model_1_1, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # 32,9,9
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16,9,9
        self.fc1 = nn.Linear(16*9*9, 64)

        # 3 Objectives Layers

        self.Efc1 = nn.Linear(64, 32)
        self.Efc2 = nn.Linear(32, 1)

        self.Cfc1 = nn.Linear(64, 32)
        self.Cfc2 = nn.Linear(32, 2)

        self.Afc1 = nn.Linear(64, 32)
        self.Afc2 = nn.Linear(32, 3)

    def forward(self, x):
        # Convolutional Layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.reshape(-1, 16*9*9)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        # 3 Objectives Layers

        E = F.relu(self.Efc1(x))
        E = self.Efc2(E)

        C = F.relu(self.Cfc1(x))
        C = self.Cfc2(C)

        A = F.relu(self.Afc1(x))
        A = self.Afc2(A)

        return (E,C,A)


class Model_1_2(nn.Module):
    def __init__(self):
        super(Model_1_2, self).__init__()


        # Info
        self.Name = 'Model_1_2'
        self.Description = '''
        Change up the Loss Coefficients,
        Energy Was being buggy
        tried using tanh instead of relu for energy
        didnt work.
        Set the coefficients to 1,0,0 to only train energy
        
        '''
        self.LossCoefficients = [1,1,1] # E,C,A,Equal Ratio of errors, need to adjust. 
        
        # History
        self.T_Loss_hist = []
        self.E_Loss_hist = []
        self.C_Loss_hist = []
        self.A_Loss_hist = []
        self.T_Loss_hist_val = []
        self.E_Loss_hist_val = []
        self.C_Loss_hist_val = []
        self.A_Loss_hist_val = []


        # Layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # 32,9,9
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16,9,9
        self.fc1 = nn.Linear(16*9*9, 64)

        # 3 Objectives Layers

        self.Efc1 = nn.Linear(64, 32)
        self.Efc2 = nn.Linear(32, 1)

        self.Cfc1 = nn.Linear(64, 32)
        self.Cfc2 = nn.Linear(32, 2)

        self.Afc1 = nn.Linear(64, 32)
        self.Afc2 = nn.Linear(32, 3)

    def forward(self, x):
        # Convolutional Layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.reshape(-1, 16*9*9)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        
        # 3 Objectives Layers

        E = F.relu(self.Efc1(x))
        # print(E)
        E = self.Efc2(E)
        # print(E)
        C = F.relu(self.Cfc1(x))
        C = self.Cfc2(C)

        A = F.relu(self.Afc1(x))
        A = self.Afc2(A)

        return (E,C,A)
    
        

class Model_2_1(nn.Module):
    def __init__(self):
        super(Model_2_1, self).__init__()


        # Info
        self.Name = 'Model_2_1'
        self.Description = '''
        The model will now be expanded to larger convolution and FC layers sizes
        Use max pool in beteween conv layers        
        Dont Go into Dense before the 3 objectives
        Eacg will have its own conv layer
        '''
        self.LossCoefficients = [1,1,1] # E,C,A,Equal Ratio of errors, need to adjust. 
        
        # History
        self.T_Loss_hist = []
        self.E_Loss_hist = []
        self.C_Loss_hist = []
        self.A_Loss_hist = []
        self.T_Loss_hist_val = []
        self.E_Loss_hist_val = []
        self.C_Loss_hist_val = []
        self.A_Loss_hist_val = []


        # Layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=0) # 64,9,9
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 32,7,7
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        

        # 3 Objectives Layers
        self.Econv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Econv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Efc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Efc2 = nn.Linear(32, 1)                              # 1

        self.Cconv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Cconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Cfc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Cfc2 = nn.Linear(32, 2)                              # 2

        self.Aconv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Aconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Afc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Afc2 = nn.Linear(32, 3)                              # 3

    def forward(self, x):
        # Convolutional Layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 3 Objectives Layers
        E = F.relu(self.Econv1(x))
        E = F.relu(self.Econv2(E))
        E = E.reshape(-1, 32*7*7)
        E = F.relu(self.Efc1(E))
        E = self.Efc2(E)

        C = F.relu(self.Cconv1(x))
        C = F.relu(self.Cconv2(C))
        C = C.reshape(-1, 32*7*7)
        C = F.relu(self.Cfc1(C))
        C = self.Cfc2(C)

        A = F.relu(self.Aconv1(x))
        A = F.relu(self.Aconv2(A))
        A = A.reshape(-1, 32*7*7)
        A = F.relu(self.Afc1(A))
        A = self.Afc2(A)


        return (E,C,A)
    
class Model_2_2(nn.Module):
    def __init__(self):
        super(Model_2_2, self).__init__()


        # Info
        self.Name = 'Model_2_2'
        self.Description = '''
        Going to adjust the coefficients to the Jonas' values
        Also introduce LR Reduction
        And Early Stopping
        '''
        self.LossCoefficients = [50,900,1500] # E,C,A,Equal Ratio of errors, need to adjust. 
        
        # History
        self.T_Loss_hist = []
        self.E_Loss_hist = []
        self.C_Loss_hist = []
        self.A_Loss_hist = []
        self.T_Loss_hist_val = []
        self.E_Loss_hist_val = []
        self.C_Loss_hist_val = []
        self.A_Loss_hist_val = []


        # Layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=0) # 64,9,9
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 32,7,7
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        

        # 3 Objectives Layers
        self.Econv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Econv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Efc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Efc2 = nn.Linear(32, 1)                              # 1

        self.Cconv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Cconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Cfc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Cfc2 = nn.Linear(32, 2)                              # 2

        self.Aconv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Aconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32,7,7
        self.Afc1 = nn.Linear(32*7*7, 32)                         # 32
        self.Afc2 = nn.Linear(32, 3)                              # 3

    def forward(self, x):
        # Convolutional Layers
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 3 Objectives Layers
        E = F.relu(self.Econv1(x))
        E = F.relu(self.Econv2(E))
        E = E.reshape(-1, 32*7*7)
        E = F.relu(self.Efc1(E))
        E = self.Efc2(E)

        C = F.relu(self.Cconv1(x))
        C = F.relu(self.Cconv2(C))
        C = C.reshape(-1, 32*7*7)
        C = F.relu(self.Cfc1(C))
        C = self.Cfc2(C)

        A = F.relu(self.Aconv1(x))
        A = F.relu(self.Aconv2(A))
        A = A.reshape(-1, 32*7*7)
        A = F.relu(self.Afc1(A))
        A = self.Afc2(A)


        return (E,C,A)
    


def Loss(y_pred, y_true,coeffs):
    # Loss is the sum of mean square error for each output scaled by coeffs
    E_loss = coeffs[0]*torch.mean((y_pred[0].squeeze() - y_true[0])**2)
    C_loss = coeffs[1]*torch.mean((y_pred[1] - y_true[1])**2)
    A_loss = coeffs[2]*torch.mean((y_pred[2] - y_true[2])**2)
    

    T_Loss = E_loss + C_loss + A_loss
    
    return T_Loss, E_loss, C_loss, A_loss

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[index,...]
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        return X, y1, y2, y3

def validate(model, dataloader_val, Loss_function, model_Coefficients):
    model.eval()
    val_T_loss = 0 
    val_E_loss = 0 
    val_C_loss = 0 
    val_A_loss = 0 
    with torch.no_grad():
        for batchX, batchlogE,batchcore,batchaxis in dataloader_val:
        
            predictions = model(batchX)
            
            T_loss,E_loss,C_loss,A_loss = Loss_function(predictions,(batchlogE,batchcore,batchaxis),coeffs = model_Coefficients)
            
            val_T_loss += T_loss.item()
            val_E_loss += E_loss.item()
            val_C_loss += C_loss.item()
            val_A_loss += A_loss.item()
            # break
    val_T_loss /= len(dataloader_val)
    val_E_loss /= len(dataloader_val)
    val_C_loss /= len(dataloader_val)
    val_A_loss /= len(dataloader_val)
    return val_T_loss, val_E_loss, val_C_loss, val_A_loss