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
Use_Sample_Set           = False # If you want to use a sample set of the data
Set_Custom_Seed          = True # If you want to set a custom seed

print('Flags:')
if Stop_Early_Debug:
    print('      Training will stop early at 1 epoch  - 1* accumulation_steps at reduced batch size')
    print('      Batch size will also be reduced by a factor of 10')
if Use_Sample_Set:
    print('      Using a test set for speed training')

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






# import model
from Model_4_0 import Model_4_Features3 as SelectModel
from Model_4_0 import Loss as Loss_function
from Model_4_0 import MyDataset
from Model_4_0 import validate
from TrainingModule_Features import Train , Tracker

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

    if Stop_Early_Debug:
        print(f'Training from scratch for Debug')
        epochs = 1
        accumulation_steps = 1
        batchBreak = 10

# Datasets
Main = Main.transpose(1,2)          # Channels come Last in LSTM
Main_val = Main_val.transpose(1,2)

# Learning the Fraction of the Signal rather than actual Signal

Truth = model.UnnormaliseY(Truth).sum(dim=1)/model.UnnormaliseY(Main).mean(dim=2).sum(dim=1).unsqueeze(1)
Truth_val = model.UnnormaliseY(Truth_val).sum(dim=1)/model.UnnormaliseY(Main_val).mean(dim=2).sum(dim=1).unsqueeze(1)

data_train        = (Main,Aux,Truth)
data_val          = (Main_val,Aux_val,Truth_val)

train_dataset = MyDataset(*data_train)
val_dataset   = MyDataset(*data_val)

dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)



model,tracker = Train(model,dataloader_train,dataloader_val,optimiser,scheduler,Loss_function,validate,\
                      Tracker,Epochs=epochs,accum_steps=accumulation_steps,device = device,batchBreak = batchBreak)


print('Training Finished')


torch.save(model,Paths.models_path+model.Name+'.pt')
torch.save(tracker,Paths.models_path+model.Name+'_Tracker.pt')
