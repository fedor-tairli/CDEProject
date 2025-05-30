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


# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.models_path)
# import model

from XmaxOnlyModel import Model_X_0 as SelectModel
from XmaxOnlyModel import Loss as Loss_function
from XmaxOnlyModel import MyDataset
from XmaxOnlyModel import validate

# Importing the dataset


X_train    = torch.load(Paths.NormData+'X_train.pt')
Xmax_train = torch.load(Paths.NormData+'Y_Xmax_train.pt')

X_val    = torch.load(Paths.NormData+'X_val.pt')
Xmax_val = torch.load(Paths.NormData+'Y_Xmax_val.pt')



# Begin Setup

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = SelectModel().to(device)

# Optimiser
LR = 0.001
optimiser = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=7, factor=0.1)

# Variables
BatchSize = int(X_train.shape[0]/512)
epochs    = 20

# Setup dataset 

X_train_tensor =  X_train.to(device)
Xmax_train_tensor = Xmax_train.to(device)

if True: # Reduce the Data Size for debug
    X_train_tensor =  X_train_tensor[:100000]
    Xmax_train_tensor = Xmax_train_tensor[:100000]


X_val_tensor =  X_val.to(device)
X_max_val_tensor = Xmax_val.to(device)

# Setup Dataloader
train_dataset = (X_train_tensor,Xmax_train_tensor)
val_dataset   = (X_val_tensor  ,X_max_val_tensor)

train_dataset = MyDataset(X_train_tensor,Xmax_train_tensor)
val_dataset   = MyDataset(X_val_tensor  ,X_max_val_tensor)

dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)




for epoch in range(epochs):
    
    model.train()
    epoch_T_loss = 0
    batchN = 0                               # Current batch number
    batchT = len(dataloader_train)           # Total N batches
    for batchX, batchXmax in dataloader_train:
        print(f'\rCurrent batch : {batchN} out of {batchT}',end='')
        batchN += 1

        optimiser.zero_grad()
        predictions = model(batchX)
        
        T_loss = Loss_function(predictions,batchXmax)
        T_loss.backward()

        optimiser.step()

        epoch_T_loss += T_loss.item()

    epoch_T_loss /= len(dataloader_train)
    model.T_Loss_history.append(epoch_T_loss)

    # Validation 

    T_Loss_val = validate(model, dataloader_val, Loss_function)
    model.T_Loss_history_val.append(T_Loss_val)

    
    scheduler.step(T_Loss_val)

    print()
    print('Epoch: ',(' '+str(epoch))[-3:],'/','{:<4}'.format(str(epochs)),'Loss: ',str(epoch_T_loss)[:9])
    print('                  ',                                                    str(T_Loss_val  )[:9])
    
    # break
    

torch.save(model,Paths.models_path+model.Name+'.pt')


