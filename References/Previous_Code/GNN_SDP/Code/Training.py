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
from Dataset import ProcessingDatasetContainer, GraphDatasetContainer
# from matplotlib import pyplot as plt



# Flags
Set_Custom_Seed    = True
Use_Test_Set       = False
Dataset_RandomIter = True

if Set_Custom_Seed:
    seed = 1234
    print('      Setting Custom Seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Reading the dataset
    
Path_To_Data = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Data/RawData'
GraphDataset = GraphDatasetContainer(ExpectedSize=0)

if Use_Test_Set: GraphDataset.Load(Path_To_Data,'Test')
else: 
    if os.path.exists(Path_To_Data+'ALL'):
        GraphDataset.Load(Path_To_Data,'ALL')
    elif os.path.exists(Path_To_Data+'RunALL'):
        GraphDataset.Load(Path_To_Data,'RunALL')
    else:
        GraphDataset.Load(Path_To_Data,'Run010')

def GetTruths(GraphDataset):
    ALLSDPPhi = GraphDataset._otherData[:,18]
    ALLSDPTheta = GraphDataset._otherData[:,19]
    
    Truths = torch.cat((torch.cos(ALLSDPPhi).unsqueeze(1),torch.sin(ALLSDPPhi).unsqueeze(1),torch.cos(ALLSDPTheta).unsqueeze(1),torch.sin(ALLSDPTheta).unsqueeze(1)),dim=1)
    return Truths

def GetFeatures(GraphDataset):
    # Get Phi,Theta,Charge
    Features = GraphDataset._pixelData[:,[9,10,4]]
    # Normalise Features
    Features[:,0] = Features[:,0]/180
    Features[:,1] = Features[:,1]/90
    
    Features[:,2] = torch.log10(Features[:,2]+1)/7.25
    return Features

Dataset = GraphDataset.GetProcessingDataset(GetTruths,GetFeatures,True,True)
Dataset.AssignIndices()
Dataset.RandomIter = Dataset_RandomIter
# import model
ModelPath = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Models/'
sys.path.append(ModelPath)
from Model_1_0 import Model_1_2 as SelectModel
from Model_1_0 import Loss as Loss_function
from Model_1_0 import validate
from TrainingModule import Train , Tracker

# Setup 
if True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = SelectModel(NDenseNodes=128,GCNNodes=64).to(device) # Use the Default Settings
    
    
    # Optimiser
    LR = 0.005 
    epochs     = 30


    optimiser = optim.Adam(model.parameters(), lr=LR) 
    
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=10, factor=0.1,min_lr=1e-6,verbose=True)

    # gamma = 0.95
    gamma = 0.001**(1/epochs)
    print(f'Gamma in LR Reduction: {gamma}')
    scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=True)

    BatchSize = 32
    epoch_done = 0
    accumulation_steps = 1 
    batchBreak = 1e99
    

    print('Training model: ',model.Name)
    print('Accumulation Steps: ',accumulation_steps)
    print('Batch Size: ',BatchSize)

plotSavePath = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Results/TrainingPlots/'
model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,\
                      Tracker,Epochs=epochs,BatchSize=BatchSize,Accumulation_steps=accumulation_steps,device = device,batchBreak = batchBreak,\
                      normStateIn='Net',normStateOut='Net',plotOnEpochCompletionPath=plotSavePath)


print('Training Finished')
SavePath = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Models/'

torch.save(model,SavePath+model.Name+'.pt')
torch.save(tracker,SavePath+model.Name+'_Tracker.pt')
