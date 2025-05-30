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


# from matplotlib import pyplot as plt


Stop_Early_Debug         = False # If you want to stop training early, only to check computation
Use_Sample_Set           = True # If you want to use a sample set of the data
Set_Custom_Seed          = True # If you want to set a custom seed
Convert_to_Dense         = True # If you want to convert the sparse data to dense data

print('Flags:')
if Stop_Early_Debug:
    print('      Training will stop early at 1 epoch  - 1* accumulation_steps at reduced batch size')
    print('      Batch size will also be reduced by a factor of 10')
if Use_Sample_Set:
    print('      Using a test set for speed training')
if Convert_to_Dense:
    print('      Converting sparse data to dense data')

# Import paths
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.models_path)

if Set_Custom_Seed:
    seed = 1234
    print('      Setting Custom Seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Reading the dataset
if not Use_Sample_Set:

    print('Importing Datasets - Training (Full)')
    D_Main  = torch.load(Paths.NormData+'D_main_train.pt')
    if Convert_to_Dense:
        D_Main = D_Main.to_dense()
    D_Aux   = torch.load(Paths.NormData+'D_aux_train.pt')
    logE    = torch.load(Paths.NormData+'logE_train.pt')
    Core    = torch.load(Paths.NormData+'Core_train.pt')
    Axis    = torch.load(Paths.NormData+'Axis_train.pt')
    Xmax    = torch.load(Paths.NormData+'Xmax_train.pt')

    print('                   - Validation (Full)')
    D_Main_val  = torch.load(Paths.NormData+'D_main_val.pt')
    if Convert_to_Dense:
        D_Main_val = D_Main_val.to_dense()
    D_Aux_val   = torch.load(Paths.NormData+'D_aux_val.pt')
    logE_val    = torch.load(Paths.NormData+'logE_val.pt')
    Core_val    = torch.load(Paths.NormData+'Core_val.pt')
    Axis_val    = torch.load(Paths.NormData+'Axis_val.pt')
    Xmax_val    = torch.load(Paths.NormData+'Xmax_val.pt')

if Use_Sample_Set: # Use the test set for speed training
    print('Importing Datasets - Training (Sample)')
    D_Main  = torch.load(Paths.NormData+'D_main_test.pt')
    if Convert_to_Dense:
        D_Main = D_Main.to_dense()
    D_Aux   = torch.load(Paths.NormData+'D_aux_test.pt')
    logE    = torch.load(Paths.NormData+'logE_test.pt')
    Core    = torch.load(Paths.NormData+'Core_test.pt')
    Axis    = torch.load(Paths.NormData+'Axis_test.pt')
    Xmax    = torch.load(Paths.NormData+'Xmax_test.pt')

    print('                   - Validation (Sample)')
    D_Main_val  = torch.load(Paths.NormData+'D_main_val.pt')
    if Convert_to_Dense:
        D_Main_val = D_Main_val.to_dense()
    D_Aux_val   = torch.load(Paths.NormData+'D_aux_val.pt')
    logE_val    = torch.load(Paths.NormData+'logE_val.pt')
    Core_val    = torch.load(Paths.NormData+'Core_val.pt')
    Axis_val    = torch.load(Paths.NormData+'Axis_val.pt')
    Xmax_val    = torch.load(Paths.NormData+'Xmax_val.pt')
    


# import model
from Model_5_0 import Model_5_0 as SelectModel
from Model_5_0 import Loss as Loss_function
from Model_5_0 import MyDataset
from Model_5_0 import validate
from TrainingModule import Train , Tracker

# Setup 
if True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = SelectModel().to(device)
    del SelectModel
    
    # Optimiser
    LR = 0.001 
    optimiser = optim.Adam(model.parameters(), lr=LR) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, factor=0.1,min_lr=1e-6,verbose=True)

    BatchSize = 50
    if Stop_Early_Debug:
        BatchSize//=10
    epochs     = 10
    if Stop_Early_Debug: epochs = 1
    epoch_done = 0
    accumulation_steps = 10
    batchBreak = 1e99
    if Stop_Early_Debug: batchBreak = accumulation_steps 

    print('Training model: ',model.Name)
    print('Accumulation Steps: ',accumulation_steps)
    print('Batch Size: ',BatchSize)

    if Stop_Early_Debug:
        print(f'Training from scratch for Debug')
        epochs = 1
        accumulation_steps = 1
        batchBreak = 10


# Qucik Augmentation
# Change D_main Order to fit the input order

D_Main = D_Main.permute(0,3,4,1,2)
D_Main_val = D_Main_val.permute(0,3,4,1,2)



data_train        = (D_Main,D_Aux)
targets_train     = (logE,Core,Axis,Xmax)

data_val          = (D_Main_val,D_Aux_val)
targets_val       = (logE_val,Core_val,Axis_val,Xmax_val)

train_dataset = MyDataset(data_train,targets_train)
val_dataset   = MyDataset(data_val  ,targets_val)

dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)



model,tracker = Train(model,dataloader_train,dataloader_val,optimiser,scheduler,Loss_function,validate,\
                      Tracker,Epochs=epochs,accum_steps=accumulation_steps,device = device,batchBreak = batchBreak)


print('Training Finished')


torch.save(model,Paths.models_path+model.Name+'.pt')
torch.save(tracker,Paths.models_path+model.Name+'_Tracker.pt')
