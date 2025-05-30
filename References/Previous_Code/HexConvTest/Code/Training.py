##############################################################
#                 Here we train the models_path                   #
#            Using the models_path define in Model.py             #
##############################################################

# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

from matplotlib import pyplot as plt


# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.models_path)
# import model

from Model_2_0 import Model_2_0 as SelectModel
from Model_2_0 import Loss_2 as Loss_function
from Model_2_0 import MyDataset
from Model_2_0 import validate

# Importing the dataset


X_train    = torch.load(Paths.NormData+'X_train.pt')
logE_train = torch.load(Paths.NormData+'Y_E_train.pt')
core_train = torch.load(Paths.NormData+'Y_Core_train.pt')
axis_train = torch.load(Paths.NormData+'Y_Axis_train.pt')
Xmax_train = torch.load(Paths.NormData+'Y_Xmax_train.pt')

X_val    = torch.load(Paths.NormData+'X_val.pt')
logE_val = torch.load(Paths.NormData+'Y_E_val.pt')
core_val = torch.load(Paths.NormData+'Y_Core_val.pt')
axis_val = torch.load(Paths.NormData+'Y_Axis_val.pt')
Xmax_val = torch.load(Paths.NormData+'Y_Xmax_val.pt')



# Begin Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = SelectModel().to(device)
model_Coefficients = model.LossCoefficients
# print(model_Coefficients)

# Optimiser
LR = 0.001 
optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) ### Regularisation L2


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, factor=0.1)

# Variables
# print(X_train.shape)
BatchSize = int(X_train.shape[0]/512)
epochs    = 10

# Setup dataset 

X_train_tensor =  X_train.to(device)
logE_train_tensor = logE_train.to(device)
core_train_tensor = core_train.to(device)
axis_train_tensor = axis_train.to(device)
Xmax_train_tensor = Xmax_train.to(device)

if True: # Reduce the Data Size for debug
    X_train_tensor =  X_train_tensor[:100000]
    logE_train_tensor = logE_train_tensor[:100000]
    core_train_tensor = core_train_tensor[:100000]
    axis_train_tensor = axis_train_tensor[:100000]
    Xmax_train_tensor = Xmax_train_tensor[:100000]


X_val_tensor =  X_val.to(device)
logE_val_tensor = logE_val.to(device)
core_val_tensor = core_val.to(device)
axis_val_tensor = axis_val.to(device)
X_max_val_tensor = Xmax_val.to(device)


targets_train     = (logE_train_tensor,core_train_tensor,axis_train_tensor,Xmax_train_tensor)
targets_val       = (logE_val_tensor,core_val_tensor,axis_val_tensor,X_max_val_tensor)

train_dataset = MyDataset(X_train_tensor,targets_train)
val_dataset   = MyDataset(X_val_tensor  ,targets_val)


dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)



print('Training model: ',model.Name)
# plt.figure()

for epoch in range(epochs):
    
    model.train()
    epoch_T_loss = 0
    epoch_E_loss = 0
    epoch_C_loss = 0
    epoch_A_loss = 0
    epoch_X_loss = 0
    batchN = 0                               # Current batch number
    batchT = len(dataloader_train)           # Total N batches
    for batchX, batchlogE,batchcore,batchaxis,batchXmax in dataloader_train:
        print(f'\rCurrent batch : {batchN} out of {batchT}',end='')
        batchN += 1

        optimiser.zero_grad()
        predictions = model(batchX)
        
        T_loss,E_loss,C_loss,A_loss,X_Loss = Loss_function(predictions,(batchlogE,batchcore,batchaxis,batchXmax),model_Coefficients)

        # L1 Regularisation
        # l1_lambda = 0.001
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # T_loss = T_loss + l1_lambda * l1_norm


        # # Plot Some things
        # batchXmax_p = predictions[3]
        # plt.cla()
        # # plt.scatter(batchXmax.cpu().detach().numpy(),batchXmax_p.cpu().detach().numpy())
        # plt.hist(batchXmax.cpu().detach().numpy(),bins=100,range=(-3,3),alpha=0.5)
        # plt.hist(batchXmax_p.cpu().detach().numpy(),bins=100,range=(-3,3),alpha=0.5)
        # # plt.plot([-3,3],[-3,3])
        # plt.title(f'Epoch {epoch}, batch {batchN}')
        # # plt.xlim(-3,3)
        # # plt.ylim(-3,3)
        # plt.show(block=False)
        # plt.pause(0.001)
        

        T_loss.backward()

        # # Print Parameters
        # for name, param in model.named_parameters():
        #     if 'Xmax' in name and 'FC' in name:
        #         if param.requires_grad:
        #             print(name)
        #             print(param.data)
        #             print(param.grad)
        #             print('---------------------------------------------------------------------------')

        optimiser.step()

        epoch_T_loss += T_loss.item()
        epoch_E_loss += E_loss.item()
        epoch_C_loss += C_loss.item()
        epoch_A_loss += A_loss.item()
        epoch_X_loss += X_Loss.item()
        # break

    epoch_T_loss /= len(dataloader_train)
    epoch_E_loss /= len(dataloader_train)
    epoch_C_loss /= len(dataloader_train)
    epoch_A_loss /= len(dataloader_train)
    epoch_X_loss /= len(dataloader_train)

    model.T_Loss_history.append(epoch_T_loss)
    model.E_Loss_history.append(epoch_E_loss)
    model.C_Loss_history.append(epoch_C_loss)
    model.A_Loss_history.append(epoch_A_loss)
    model.X_Loss_history.append(epoch_X_loss)

    # Validation 

    T_Loss_val,E_Loss_val,C_Loss_val,A_Loss_val,X_Loss_val = validate(model,dataloader_val,Loss_function ,model_Coefficients)
    model.T_Loss_history_val.append(T_Loss_val)
    model.E_Loss_history_val.append(E_Loss_val)
    model.C_Loss_history_val.append(C_Loss_val)
    model.A_Loss_history_val.append(A_Loss_val)
    model.X_Loss_history_val.append(X_Loss_val)

    scheduler.step(T_Loss_val)

    track     = str( 'Epoch: '+(' '+str(epoch+1))[-3:]+'/'+'{:<4}'.format(str(epochs))+'Loss: '+str(epoch_T_loss)[:9]+' E: '+str(epoch_E_loss)[:9]+' C: '+str(epoch_C_loss)[:9]+' A: '+str(epoch_A_loss)[:9]+' X: '+str(epoch_X_loss)[:9])
    val_track = str(                                                                          str(T_Loss_val  )[:9]+' E: '+str(E_Loss_val  )[:9]+' C: '+str(C_Loss_val  )[:9]+' A: '+str(A_Loss_val  )[:9]+' X: '+str(X_Loss_val  )[:9]).rjust(len(track))

    print()
    print(track)
    print(val_track)
    # print('Including L1 Norm: ',l1_norm.item())
    # break
    

torch.save(model,Paths.models_path+model.Name+'.pt')


