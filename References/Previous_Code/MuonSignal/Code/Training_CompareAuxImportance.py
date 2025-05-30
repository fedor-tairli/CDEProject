##############################################################
#                 Here we train the models_path              #
#            Using the models_path define in Model.py        #
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
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# from matplotlib import pyplot as plt

# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.models_path)

# import model

from Model_3_0 import Loss as Loss_function
from Model_3_0 import MyDataset
from Model_3_0 import validate
from TrainingModule import Train , Tracker

from Model_3_0 import Model_3_Baseline
from Model_3_0 import Model_3_BaselineAll
from Model_3_0 import Model_3_logEOnly
from Model_3_0 import Model_3_XmaxOnly 
from Model_3_0 import Model_3_CosZenOnly
from Model_3_0 import Model_3_CoreDistOnly
from Model_3_0 import Model_3_Renormalisation
from Model_3_0 import Model_3_SquaredLoss

def UnnormaliseY(Y):
    # Unnorm -> out = np.log10(out+1)/np.log10(101)
    Y = Y*np.log10(101)
    Y = 10**Y
    Y = Y-1
    return Y


Stop_Early_Debug         = False # If you want to stop training early, only to check computation
Use_Sample_Set           = False # If you want to use a sample set of the data
Set_Custom_Seed          = False # If you want to set a custom seed

print('Flags:')
if Stop_Early_Debug:
    print('      Training will stop early at 1 epoch  - 1* accumulation_steps at reduced batch size')
    print('      Batch size will also be reduced by a factor of 10')
if Use_Sample_Set:
    print('      Using a test set for speed training')

if Set_Custom_Seed:
    seed = 1234
    print('      Setting Custom Seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Reading the dataset
if not Use_Sample_Set:

    print('Importing Datasets - Training (Full)')
    Main  = torch.load(Paths.NormData+'Main_train.pt')
    Aux   = torch.load(Paths.NormData+'Aux_train.pt')
    Truth = torch.load(Paths.NormData+'Truth_train.pt')

    print('                   - Validation (Full)')
    Main_val  = torch.load(Paths.NormData+'Main_val.pt')
    Aux_val   = torch.load(Paths.NormData+'Aux_val.pt')
    Truth_val = torch.load(Paths.NormData+'Truth_val.pt')

if Use_Sample_Set: # Use the test set for speed training
    print('Importing Datasets - Training (Sample)')
    Main  = torch.load(Paths.NormData+'Main_test.pt')
    Aux   = torch.load(Paths.NormData+'Aux_test.pt')
    Truth = torch.load(Paths.NormData+'Truth_test.pt')

    print('                   - Validation (Sample)')
    Main_val  = torch.load(Paths.NormData+'Main_test.pt')
    Aux_val   = torch.load(Paths.NormData+'Aux_test.pt')
    Truth_val = torch.load(Paths.NormData+'Truth_test.pt')

# Adjust the Shapes
Aux = Aux.unsqueeze(2)
Aux_val = Aux_val.unsqueeze(2)

Truth = Truth.unsqueeze(2)
Truth_val = Truth_val.unsqueeze(2)

# Datasets
Main = Main.transpose(1,2)          # Channels come Last in LSTM
Main_val = Main_val.transpose(1,2)

# Learning the Fraction of the Signal rather than actual Signal

Truth = UnnormaliseY(Truth).sum(dim=1)/UnnormaliseY(Main).mean(dim=2).sum(dim=1).unsqueeze(1)
Truth_val = UnnormaliseY(Truth_val).sum(dim=1)/UnnormaliseY(Main_val).mean(dim=2).sum(dim=1).unsqueeze(1)

data_train        = (Main,Aux,Truth)
data_val          = (Main_val,Aux_val,Truth_val)

train_dataset = MyDataset(*data_train)
val_dataset   = MyDataset(*data_val)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_Baseline        = Model_3_Baseline().to(device)
# model_BaselineAll     = Model_3_BaselineAll().to(device)
# model_logEOnly        = Model_3_logEOnly().to(device)
# model_XmaxOnly        = Model_3_XmaxOnly().to(device)
# model_CosZenOnly      = Model_3_CosZenOnly().to(device)
# model_CoreDistOnly    = Model_3_CoreDistOnly().to(device)
# model_Renormalisation = Model_3_Renormalisation().to(device)
model_SquaredLoss     = Model_3_SquaredLoss().to(device)


# models = [model_Baseline,model_logEOnly,model_XmaxOnly,model_CosZenOnly,model_CoreDistOnly]
# models = [model_BaselineAll]
# models = [model_BaselineAll,model_Renormalisation]
models = [model_SquaredLoss]
for model in models:

    # Optimiser
    LR = 0.001 
    optimiser = optim.Adam(model.parameters(), lr=LR) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=4, factor=0.1,min_lr=1e-6,verbose=True)

    BatchSize = 1000
    if Stop_Early_Debug:
        BatchSize//=10
    epochs     = 100
    if Stop_Early_Debug: epochs = 1
    epoch_done = 0
    accumulation_steps = 10
    batchBreak = 1e99
    if Stop_Early_Debug: batchBreak = accumulation_steps 

    print('Training model: ',model.Name)
    print('Accumulation Steps: ',accumulation_steps)
    print('Batch Size: ',BatchSize)

    dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
    dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)



    model,tracker = Train(model,dataloader_train,dataloader_val,optimiser,scheduler,Loss_function,validate,\
                        Tracker,Epochs=epochs,accum_steps=accumulation_steps,device = device,batchBreak = batchBreak)


    print(f'Training Finished, sasving {model.Name}')


    torch.save(model,Paths.models_path+model.Name+'.pt')
    torch.save(tracker,Paths.models_path+model.Name+'_Tracker.pt')
