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
from Dataset import ProcessingDatasetContainer, GraphDatasetContainer
# from matplotlib import pyplot as plt



def GetTruths(GraphDataset):
    print(f'Truth Values are : [Chi0,Rp]')
    ALL_Chi0   = GraphDataset.GetValues('GenChi0')
    ALL_Rp     = GraphDataset.GetValues('GenRp')

    Mean_Rp     = 12800
    STD_Rp      = 5800

    Norm_Chi0   = torch.cos(ALL_Chi0)
    Norm_Rp     = (ALL_Rp-Mean_Rp)/STD_Rp

    return torch.stack([Norm_Chi0,Norm_Rp],dim=1)

def GetRecTruths(GraphDataset):
    ALL_Chi0   = GraphDataset.GetValues('RecChi0')
    ALL_Rp     = GraphDataset.GetValues('RecRp')

    Mean_Rp     = 12800
    STD_Rp      = 5800

    Norm_Chi0   = torch.cos(ALL_Chi0)
    Norm_Rp     = (ALL_Rp-Mean_Rp)/STD_Rp

    return torch.stack([Norm_Chi0,Norm_Rp],dim=1)

def GetFeatures(GraphDataset):
    # Get Chi,Charge,Time,Duration
    print(f'Feature Values are : [Chi,Charge,Centroid,Duration]')
    Chi      = GraphDataset._pixelData[:,5]
    Charge   = GraphDataset._pixelData[:,4] 
    Centroid = GraphDataset._pixelData[:,7]
    Duration = GraphDataset._pixelData[:,8] - GraphDataset._pixelData[:,6]
    TelID    = GraphDataset._pixelData[:,1]
    EyeID    = GraphDataset._pixelData[:,2]

    # Normalise the values
    Chi = Chi
    Charge = torch.log10(Charge+1)/5
    Centroid = Centroid/1000
    Mask = (TelID == 7) | (TelID == 8) | (TelID == 9) | (EyeID == 5)
    Duration[Mask] *= 0.5
    Duration = Duration/100
    
    return torch.stack([Chi,Charge,Centroid,Duration],dim=1)

def GetStationFeatures(GraphDataset):
    print(f'Station Feature Values are : [Chi,Charge,Centroid,Duration]')
    # Chi_i for station doesnt exist in the dataset so pretend its zero casue thats basically close enough
    # StationDuration also doesnt exist in the dataset so i am pretending its 1
    StationCharge   = GraphDataset._otherData[:,0]
    StationTime     = GraphDataset._otherData[:,1]

    StationChi      = torch.zeros(StationCharge.shape)
    StationDuration = torch.ones(StationCharge.shape)

    # Need to normalise the values
    StationCharge = torch.log10(StationCharge+1)/4
    StationTime = StationTime/1000

    return torch.stack([StationChi,StationCharge,StationTime,StationDuration],dim=1)


if __name__ == '__main__':

    # Flags
    Set_Custom_Seed    = False
    Use_Test_Set       = False
    Use_All_Sets       = True
    Dataset_RandomIter = True

    if Set_Custom_Seed:
        seed = 1234
        print('      Setting Custom Seed')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # Reading the dataset
        
    Path_To_Data = '/remote/tychodata/ftairli/work/Projects/Geometry_Reconstruction/Data/RawData'
    GraphDataset = GraphDatasetContainer(ExpectedSize=0)

    if Use_Test_Set:
        GraphDataset.Load(Path_To_Data,'Test')
        print(f'Using Test Set with Nevents: {GraphDataset._Nevents}')
    elif not Use_All_Sets:
        GraphDataset.Load(Path_To_Data,'Run010')
        print(f'Using Run010 Set with Nevents: {GraphDataset._Nevents}')
    else: 
        GraphDataset.Load(Path_To_Data,'ALL')
        print(f'Using ALL Set with Nevents: {GraphDataset._Nevents}')






    Dataset = GraphDataset.GetProcessingDataset(GetTruths,GetFeatures,GetStationFeatures,True,True,True)
    Dataset.AssignIndices()
    Dataset.RandomIter = Dataset_RandomIter

    # import model
    ModelPath = '/remote/tychodata/ftairli/work/Projects/Geometry_Reconstruction/Models/'
    sys.path.append(ModelPath)
    from Model_2_0 import Model_2_2 as SelectModel
    from Model_2_0 import Loss as Loss_function
    from Model_2_0 import validate
    from TrainingModule import Train , Tracker

    # Setup 
    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
        print(f'Using device: {device}')
        # 128,128 is about as big as it can go
        model = SelectModel(NDenseNodes=64,GCNNodes=64,attention_heads = 1).to(device) # Use the Default Settings
        print()
        print('Model Description')
        print(model.Description)
        print()     
        # Optimiser
        LR         = 0.005  # Starting Learning Rate
        epochs     = 50     # Number of Epochs


        optimiser = optim.Adam(model.parameters(), lr=LR) 
        
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=10, factor=0.1,min_lr=1e-6,verbose=True)

        # gamma = 0.95
        gamma = 0.001**(1/epochs) # Should Reduce the learning rate by a factor of 0.001 over the course of the training
        print(f'Gamma in LR Reduction: {gamma}')
        scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=False)

        BatchSize = 32
        epoch_done = 0
        accumulation_steps = 1 
        batchBreak = 1e99
        

        print('Training model: ',model.Name)
        print('Accumulation Steps: ',accumulation_steps)
        print('Batch Size: ',BatchSize)

    plotSavePath = '/remote/tychodata/ftairli/work/Projects/Geometry_Reconstruction/Results/TrainingPlots/'
    if plotSavePath != None : print(f'Plot Save Path: {plotSavePath}')
    else: print('Plots will not be saved')
    model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,\
                        Tracker,Epochs=epochs,BatchSize=BatchSize,Accumulation_steps=accumulation_steps,device = device,batchBreak = batchBreak,\
                        normStateIn='Net',normStateOut='Net',plotOnEpochCompletionPath=plotSavePath)


    print('Training Finished')
    SavePath = '/remote/tychodata/ftairli/work/Projects/Geometry_Reconstruction/Models/'

    torch.save(model,SavePath+model.Name+'.pt')
    torch.save(tracker,SavePath+model.Name+'_Tracker.pt')
