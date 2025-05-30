##############################################################
#                 Here we train the models                   #
#            Using the models define in Model.py             #
##############################################################

# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# import model

from Model import Model_2_2 as SelectModel
from Model import Loss as Loss_function
from Model import MyDataset
from Model import validate

# Importing the dataset


X_train    = torch.load(Paths.data_path+'X_train.pt')
logE_train = torch.load(Paths.data_path+'logE_train.pt')
core_train = torch.load(Paths.data_path+'core_train.pt')
axis_train = torch.load(Paths.data_path+'axis_train.pt')

X_train,    X_val    = torch.split(X_train,[160000,20000],dim=0)
logE_train, logE_val = torch.split(logE_train,[160000,20000],dim=0)
core_train, core_val = torch.split(core_train,[160000,20000],dim=0)
axis_train, axis_val = torch.split(axis_train,[160000,20000],dim=0)


# Begin Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SelectModel().to(device)
model_Coefficients = model.LossCoefficients
# print(model_Coefficients)

# Optimiser
LR = 0.001
optimiser = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=10, factor=0.1)

# Variables

BatchSize = 128
epochs    = 1000

# Setup dataset 

X_train_tensor =  X_train.to(device)
logE_train_tensor = logE_train.to(device)
core_train_tensor = core_train.to(device)
axis_train_tensor = axis_train.to(device)

X_val_tensor =  X_val.to(device)
logE_val_tensor = logE_val.to(device)
core_val_tensor = core_val.to(device)
axis_val_tensor = axis_val.to(device)


targets_train     = (logE_train_tensor,core_train_tensor,axis_train_tensor)
targets_val       = (logE_val_tensor,core_val_tensor,axis_val_tensor)

train_dataset = MyDataset(X_train_tensor,targets_train)
val_dataset   = MyDataset(X_val_tensor  ,targets_val)

dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)



for epoch in range(epochs):
    
    model.train()
    epoch_T_loss = 0
    epoch_E_loss = 0
    epoch_C_loss = 0
    epoch_A_loss = 0

    for batchX, batchlogE,batchcore,batchaxis in dataloader_train:
        
        optimiser.zero_grad()
        predictions = model(batchX)
        
        T_loss,E_loss,C_loss,A_loss = Loss_function(predictions,(batchlogE,batchcore,batchaxis),model_Coefficients)
        T_loss.backward()
        optimiser.step()

        epoch_T_loss += T_loss.item()
        epoch_E_loss += E_loss.item()
        epoch_C_loss += C_loss.item()
        epoch_A_loss += A_loss.item()
        # break

    epoch_T_loss /= len(dataloader_train)
    epoch_E_loss /= len(dataloader_train)
    epoch_C_loss /= len(dataloader_train)
    epoch_A_loss /= len(dataloader_train)

    model.T_Loss_hist.append(epoch_T_loss)
    model.E_Loss_hist.append(epoch_E_loss)
    model.C_Loss_hist.append(epoch_C_loss)
    model.A_Loss_hist.append(epoch_A_loss)

    # Validation 

    T_Loss_val,E_Loss_val,C_Loss_val,A_Loss_val = validate(model,dataloader_val,Loss_function ,model_Coefficients)
    model.T_Loss_hist_val.append(T_Loss_val)
    model.E_Loss_hist_val.append(E_Loss_val)
    model.C_Loss_hist_val.append(C_Loss_val)
    model.A_Loss_hist_val.append(A_Loss_val)

    scheduler.step(T_Loss_val)


    print('Epoch: ',('     '+str(epoch))[-3:],'Loss: ',str(epoch_T_loss)[:9],'E: ',str(epoch_E_loss)[:9],'C: ',str(epoch_C_loss)[:9],'A: ',str(epoch_A_loss)[:9])
    print('                  ',                        str(T_Loss_val  )[:9],'E: ',str(E_Loss_val  )[:9],'C: ',str(C_Loss_val  )[:9],'A: ',str(A_Loss_val  )[:9])
    
    # break
    

torch.save(model,Paths.models_path+model.Name+'.pt')


