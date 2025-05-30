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
import ManageData as MD

# from matplotlib import pyplot as plt



# Flags
Stop_Early_Debug         = False # If you want to stop training early, only to check computation
Use_Sample_Set           = False # If you want to use a sample set of the data
Set_Custom_Seed          = False # If you want to set a custom seed
Clean_Infs               = True # If you want to clean infs from the data
useDtype                 = torch.float32 # If you want to use a different dtype



print('Flags:')
if Stop_Early_Debug:
    print('      Training will stop early at 1 epoch  - 1* accumulation_steps at reduced batch size')
    print('      Batch size will also be reduced by a factor of 10')
if Use_Sample_Set:
    print('      Using a test set for speed training')
if useDtype != torch.float32:
    print(f'      Using dtype {useDtype}')
    torch.set_default_dtype(useDtype)

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
    Main_train        = torch.load(Paths.NormData+'Main_train.pt')
    PixDur_train      = torch.load(Paths.NormData+'PixDur_train.pt')
    GenGeometry_train = torch.load(Paths.NormData+'GenGeometry_train.pt')
    Meta_train        = torch.load(Paths.NormData+'Meta_train.pt')

    print('                   - Validation (Full)')
    Main_val          = torch.load(Paths.NormData+'Main_val.pt')
    PixDur_val        = torch.load(Paths.NormData+'PixDur_val.pt')
    GenGeometry_val   = torch.load(Paths.NormData+'GenGeometry_val.pt')
    Meta_val          = torch.load(Paths.NormData+'Meta_val.pt')

if Use_Sample_Set: # Use the test set for speed training
    print('Importing Datasets - Training (Sample)')
    Main_train        = torch.load(Paths.NormData+'Main_test.pt')
    PixDur_train      = torch.load(Paths.NormData+'PixDur_test.pt')
    GenGeometry_train = torch.load(Paths.NormData+'GenGeometry_test.pt')
    Meta_train        = torch.load(Paths.NormData+'Meta_test.pt')

    print('                   - Validation (Sample)')
    Main_val          = torch.load(Paths.NormData+'Main_val.pt')
    PixDur_val        = torch.load(Paths.NormData+'PixDur_val.pt')
    GenGeometry_val   = torch.load(Paths.NormData+'GenGeometry_val.pt')
    Meta_val          = torch.load(Paths.NormData+'Meta_val.pt')

if useDtype != Main_train.dtype:
    print('Changing dtype')
    Main_train = Main_train.to(useDtype)
    Main_val   = Main_val.to(useDtype)
    PixDur_train = PixDur_train.to(useDtype)
    PixDur_val   = PixDur_val.to(useDtype)
    GenGeometry_train = GenGeometry_train.to(useDtype)
    GenGeometry_val   = GenGeometry_val.to(useDtype)

if Clean_Infs:
    print('Cleaning Infs')
    infs_mask = torch.isinf(Main_train)
    print('Found {} Infs in Main_train'.format(infs_mask.sum()))
    Main_train[infs_mask] = 0.0
    infs_mask = torch.isinf(Main_val)
    print('Found {} Infs in Main_val'.format(infs_mask.sum()))
    Main_val[infs_mask] = 0.0


# import model
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_Time_Charge
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_Time_Only
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_Charge_Only
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_ActivePixel
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_PulseDuration
from Model_ImportanceAnalysis import Model_ImportanceAnalysis_Combined


from Model_ImportanceAnalysis import Loss as Loss_function
from Model_ImportanceAnalysis import MyDataset
from Model_ImportanceAnalysis import validate
from TrainingModule import Train , Tracker

List_of_Models = [Model_ImportanceAnalysis_Time_Charge,Model_ImportanceAnalysis_Time_Only,Model_ImportanceAnalysis_Charge_Only,Model_ImportanceAnalysis_ActivePixel]

With_durations = True
if With_durations:
    List_of_Models = [Model_ImportanceAnalysis_Combined,Model_ImportanceAnalysis_PulseDuration]


# Datasets
# Prepare the data for Net

Main_train[:,0,:,:] = MD.PixTime_to_net(Main_train[:,0,:,:])
Main_train[:,1,:,:] = MD.PixSig_to_net(Main_train[:,1,:,:])
Main_val[:,0,:,:]   = MD.PixTime_to_net(Main_val[:,0,:,:])
Main_val[:,1,:,:]   = MD.PixSig_to_net(Main_val[:,1,:,:])
PixDur_train        = MD.PixDur_to_net(PixDur_train)
PixDur_val          = MD.PixDur_to_net(PixDur_val)
MirrorIds_train = Meta_train[:,9]
MirrorIds_val  = Meta_val[:,9]

Truth_train = GenGeometry_train[:,[3,4]]
Truth_val   = GenGeometry_val[:,[3,4]]

# Shift Phi to Mirror Frame though
Truth_train[:,0] = MD.Phi_to_mirror(Truth_train[:,0],MirrorIds_train)
Truth_val[:,0]   = MD.Phi_to_mirror(Truth_val[:,0],MirrorIds_val)

Truth_train[:,0] = MD.Phi_to_net(Truth_train[:,0])
Truth_val[:,0]   = MD.Phi_to_net(Truth_val[:,0])

Truth_train[:,1] = MD.Theta_to_net(Truth_train[:,1])
Truth_val[:,1]   = MD.Theta_to_net(Truth_val[:,1])

# For sanity, This shoudlnt do anything really
Theta_mask = torch.isfinite(Truth_train[:,1])
Phi_mask   = torch.isfinite(Truth_train[:,0])
Comb_mask = Theta_mask&Phi_mask
Main_train = Main_train[Comb_mask]
Truth_train = Truth_train[Comb_mask]

Theta_mask = torch.isfinite(Truth_val[:,1])
Phi_mask   = torch.isfinite(Truth_val[:,0])
Comb_mask = Theta_mask&Phi_mask
Main_val = Main_val[Comb_mask]
Truth_val = Truth_val[Comb_mask]

if With_durations:
    Main_train = torch.cat((Main_train[:,[0,1],:,:],PixDur_train),dim=1)
    Main_val   = torch.cat((Main_val[:,[0,1],:,:],PixDur_val),dim=1)
    print('PixDur Shape : ',PixDur_train.shape)
    print('Main Shape   : ',Main_train.shape)

del GenGeometry_train, GenGeometry_val, Theta_mask, Phi_mask, Comb_mask, MirrorIds_train, MirrorIds_val


# Datasets
train_dataset = MyDataset(Main_train,Truth_train)
val_dataset   = MyDataset(Main_val,Truth_val)

BatchSize = 32

dataloader_train    = data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
dataloader_val      = data.DataLoader(val_dataset,batch_size=BatchSize,shuffle=True)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

for SelectModel in List_of_Models:

    model = SelectModel(N_filters = 32,NDenseNodes = 8,Dtype=useDtype).to(device)

    # Optimiser
    LR = 0.0005 
    epochs     = 100
    optimiser = optim.Adam(model.parameters(), lr=LR) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = 0.95, last_epoch=-1, verbose=True)


    accumulation_steps = 1
    batchBreak = 1e99

    print('Training model: ',model.Name)
    print('Accumulation Steps: ',accumulation_steps)
    print('Batch Size: ',BatchSize)

    model,tracker = Train(model,dataloader_train,dataloader_val,optimiser,scheduler,Loss_function,validate,\
                        Tracker,Epochs=epochs,accum_steps=accumulation_steps,device = device,batchBreak = batchBreak,\
                        normStateIn='Net',normStateOut='Net',plotHistograms = False)


    print('Training Finished')


    torch.save(model,Paths.models_path+model.Name+'.pt')
    torch.save(tracker,Paths.models_path+model.Name+'_Tracker.pt')
    del model, tracker, optimiser, scheduler

