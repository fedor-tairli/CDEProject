##############################################################
#                 Here we train the models_path                   #
#            Using the models_path define in Model.py             #
##############################################################

# Importing the libraries
import os 
os.system('clear')
import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys
import time

import pprint 
pp = pprint.PrettyPrinter().pprint


# from matplotlib import pyplot as plt

# Increase the threshold 
# Not sure if this actually works but whatever.
torch.set_printoptions(threshold=10000)

if os.path.exists('out.txt'):
    os.remove('out.txt')



def print_memory_usage():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f'GPU memory usage: allocated {allocated / 1024**2:.2f}MB, cached {cached / 1024**2:.2f}MB')



def prints(model):
    print('################################################################################')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Name : ',name)
            print('Shape: ',param.data.shape)
            print('Mean : ',param.data.mean().item())
            print('STD  : ',param.data.std().item())
            print()
    print('################################################################################')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #         pp(param.grad.data)
    print('################################################################################')

# call this function at various points in your code
# print_memory_usage()



# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.models_path)
# import model

from Model_X_1 import Model_X_1 as SelectModel
from Model_X_1 import Loss as Loss_function
from Model_X_1 import MyDataset
from Model_X_1 import validate
from Model_X_1 import Training_Track

# Importing the dataset


D_main_train = torch.load(Paths.NormData+'D_main_train.pt')
D_aux_train  = torch.load(Paths.NormData+'D_aux_train.pt')
Xmax_train   = torch.load(Paths.NormData+'Xmax_train.pt')

D_main_val = torch.load(Paths.NormData+'D_main_val.pt')
D_aux_val  = torch.load(Paths.NormData+'D_aux_val.pt')
Xmax_val   = torch.load(Paths.NormData+'Xmax_val.pt')

# ### Introducing random normal distribution instead

# Xmax_train = torch.randn_like(Xmax_train)
# Xmax_val   = torch.randn_like(Xmax_val)

# Begin Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'Using device: {device}')
model = SelectModel().to(device)
del SelectModel






if True: # Reduce the Data Size for debug
    if D_main_train.is_sparse:
        D_main_train = D_main_train.index_select(0,torch.arange(100000))
        D_aux_train  = D_aux_train[:100000]
        Xmax_train = Xmax_train[:100000]
    else:
        D_main_train = D_main_train[:100000]
        D_aux_train  = D_aux_train[:100000]
        Xmax_train = Xmax_train[:100000]

# Drop the data where zero
goodindex = torch.where(Xmax_train!=0)[0]

print(f'Dropped {len(Xmax_train)-len(goodindex)} events where Xmax = 0')

D_main_train = D_main_train.index_select(0,goodindex)
D_aux_train  = D_aux_train.index_select(0,goodindex)
Xmax_train   = Xmax_train.index_select(0,goodindex)


# Optimiser
LR = 0.001 
optimiser = optim.Adam(model.parameters(), lr=LR) ### Weight_decay is Regularisation L2


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, factor=0.1,min_lr=1e-6,verbose=True)

# Variables
# print(X_train.shape)
# BatchSize = int(Core_train.shape[0]/512)
BatchSize = 30
epochs    = 5
epoch_done = 0

# Datasets
data_train        = (D_main_train,D_aux_train)
data_val          = (D_main_val,D_aux_val)
targets_train     = Xmax_train
targets_val       = Xmax_val

train_dataset = MyDataset(data_train,targets_train)
val_dataset   = MyDataset(data_val  ,targets_val)


dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)

print('Training model: ',model.Name)


# for gradient accumulation
accumulation_steps = 100  # probably best to make sure that the entire length of the dataset is divisible by this number

if False: # Load model if you want to continue training more epochs
    model = torch.load(Paths.models_path+model.Name+'.pt')
    print('Continue Training')
    print('Model Loaded')
    epoch_done = len(model.X_Loss_history)
    print('Epochs done: ',epoch_done)
else:
    print(f'Training from scratch of {epochs} epochs')
    


epoch_finish = epoch_done + epochs
for epoch in range(epoch_done,epoch_finish):
    start_time = time.time()
    
    model.train()
    epoch_X_loss = 0

    batchN = 0                               # Current batch number
    batchT = len(dataloader_train)           # Total N batches
    
    for batchD_main,batchD_aux,batchXmax in dataloader_train:

        
        # Bring Batches to Device
        batchD_main = batchD_main.to(device)
        batchD_aux  = batchD_aux.to(device)
        batchXmax   = batchXmax.to(device)
        

        batchN += 1
        print(f'\rCurrent batch : {batchN} out of {batchT}',end='')
        with open('out.txt','a') as f:
            f.write(f'Current batch : {batchN} out of {batchT}\n')


        if batchN % accumulation_steps == 1:
            optimiser.zero_grad()
        
        predictions = model(batchD_main,batchD_aux)
        
        X_Loss = Loss_function(predictions,batchXmax)
        

        (X_Loss/accumulation_steps).backward()

        if batchN % accumulation_steps == 0:
            # prints(model) # some extra prints
            optimiser.step()
        
        epoch_X_loss += X_Loss.item()
        torch.cuda.empty_cache()
        # if batchN == 10:
        #     break
    print()

    epoch_X_loss /= len(dataloader_train)

    model.X_Loss_history.append(epoch_X_loss)

    # Validation 

    X_Loss_val = validate(model,dataloader_val,Loss_function)
    model.X_Loss_history_val.append(X_Loss_val)

    scheduler.step(X_Loss_val)
    track     = str( 'Epoch: '+(' '+str(epoch+1))[-3:]+'/'+'{:<4}'.format(str(epochs))+'Loss: '+str(epoch_X_loss)[:9])
    val_track = str(                                                                          str(X_Loss_val  )[:9]).rjust(len(track))
    # track     = str( 'Epoch: '+(' '+str(epoch+1))[-3:]+'/'+'{:<4}'.format(str(epochs))+'Loss: '+str(epoch_T_loss)[:9]+' E: '+str(epoch_E_loss)[:9]+' C: '+str(epoch_C_loss)[:9]+' A: '+str(epoch_A_loss)[:9]+' X: '+str(epoch_X_loss)[:9])
    # val_track = str(                                                                          str(T_Loss_val  )[:9]+' E: '+str(E_Loss_val  )[:9]+' C: '+str(C_Loss_val  )[:9]+' A: '+str(A_Loss_val  )[:9]+' X: '+str(X_Loss_val  )[:9]).rjust(len(track))
    end_time = time.time()

    print()
    print(track)
    print(val_track)
    Training_Track(model,dataloader_val)
    print('Time: ',end_time-start_time)
    print()
    print()
    # print('Including L1 Norm: ',l1_norm.item())
    # break
    

torch.save(model,Paths.models_path+model.Name+'.pt')
print('Training Finished')

