# Importing the libraries
import os 
import sys
import time
import warnings
os.system('clear')
warnings.filterwarnings("ignore")


# Special guest pandas
# import numpy as np
import pandas as pd

import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


ModelPath = os.path.abspath('../Models') + '/'
sys.path.append(ModelPath)
# Dataset modules
from Dataset2 import DatasetContainer, ProcessingDatasetContainer
def Do_Nothing(Truth, *kwargs):
    '''
    A function that does nothing, used as a placeholder
    '''
    return Truth


def LoadProcessingDataset(Path_To_Data,DatasetName = None,RecalculateDataset = False):
    # Load a clean ProcessingDatasetContainer, then Load the data into it from CSV File
    if not RecalculateDataset and os.path.exists(DatasetName + '.pt'):
        
        ProcessingDataset = torch.load(DatasetName + '.pt',weights_only=False)
    else:

        ProcessingDataset = ProcessingDatasetContainer()
        # Read Shit from csv
        # print('Readig In csv')
        MetaData = []
        Traces   = []
        for DataChunk in pd.read_csv(Path_To_Data,delimiter=',',chunksize = 1000):
            MetaData.append(torch.tensor(DataChunk.iloc[:, 0:3 ].values, dtype=torch.int64 ))
            Traces  .append(torch.tensor(DataChunk.iloc[:, 3:-1].values, dtype=torch.float32))
        MetaData = torch.cat(MetaData,dim=0)
        Traces   = torch.cat(Traces  ,dim=0)

        # Need Traces to be of shape (N,1,1000) -> add extra dimension
        Traces = Traces.unsqueeze(1)
        print(f'Traces Computed with shape : {Traces.shape}')

        # precompute Loss Weights
        Weights = torch.ones_like(Traces)
        
        for i in range(len(Traces)):
            if i%100 == 0: print(f'Calculating weights, {i}/{len(Traces)}',end = '\r')
            # Apply the weight of the status of trace
            Status_weight = 1 if MetaData[i,0] == 4 else 0.05
            Weights[i,0,:]*= Status_weight
            # Apply the weight of the calculated trigger
            # MetaData[:,1] is the start of the trigger, MetaData[:,2] is the end of the trigger

            if MetaData[i,0] < 1:
                Weights[i,0,:] *= 0.05
            else:
                start = max(int(MetaData[i,1]) -5,0   )
                end   = min(int(MetaData[i,2]) +5,1000)
                Weights[i,0,:start] *= 0.05
                Weights[i,0,end:]   *= 0.05
            

        ProcessingDataset.GraphData = False
        ProcessingDataset._Main.append(Traces)
        ProcessingDataset._Truth = Traces
        ProcessingDataset._Rec   = Traces
        ProcessingDataset._EventIds = torch.zeros(len(Traces))
        ProcessingDataset._Aux      = Weights
        ProcessingDataset._MetaData = MetaData

        ProcessingDataset.Name = DatasetName
        ProcessingDataset.Unnormalise_Truth = Do_Nothing
        ProcessingDataset.Truth_Keys = ('TraceLikeness',)
        ProcessingDataset.Truth_Units = ('',)
        ProcessingDataset.Aux_Keys = ('Weights',)
        ProcessingDataset.Aux_Units   = ('',)



        ProcessingDataset.Save('.',Name = DatasetName)
    return ProcessingDataset



if __name__ == '__main__':

    # Flags
    Set_Custom_Seed      = False
    Use_Test_Set         = False
    Use_All_Sets         = True
    Dataset_RandomIter   = True
    RecalculateDataset   = True
    NeedTraces           = True
    LoadModel            = False
    DoNotTrain           = False
    DatasetName          = 'TracesDataset' #No / or .pt JUST NAME, eg GraphStructure  Use None to save as default


    if DoNotTrain: assert RecalculateDataset, 'Recalculate Dataset must be True if DoNotTrain is True'

    if Set_Custom_Seed:
        seed = 1234
        print('      Setting Custom Seed')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save Paths
    SavePath     = os.path.abspath('../Models') + '/'
    plotSavePath = None
    LogPath      = os.path.abspath('.') + '/'
    # Check that all the paths exist
    assert os.path.exists(SavePath)     , f'SavePath {SavePath} does not exist'
    if plotSavePath is not None: assert os.path.exists(plotSavePath) , f'plotSavePath {plotSavePath} does not exist'
    assert os.path.exists(LogPath)      , f'LogPath {LogPath} does not exist'


    if plotSavePath != None:  # Purge the directory
        os.system(f'rm -r {plotSavePath}')
        os.system(f'mkdir {plotSavePath}')

    # Reading the dataset    
    Path_To_Data      = os.path.abspath('../Data') + '/EPOSTLHC_R_180_200_allmass_HybridSd_CORSIKA78010_FLUKA_Runall_traces.csv'
    
    if DoNotTrain: print('No Training will be done, Just Reading the Dataset')
    Dataset = LoadProcessingDataset(Path_To_Data,RecalculateDataset = RecalculateDataset,DatasetName = DatasetName)
    Dataset.AssignIndices()
    Dataset.RandomIter = Dataset_RandomIter

    
    if not DoNotTrain:
        # import model
        from TrainingModule import Train , Tracker
        from TraceTriggerModel import Loss as Loss_function
        from TraceTriggerModel import validate, metric
        from TraceTriggerModel import TraceTriggerModel

        
        Models = [
            TraceTriggerModel,
            # Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta,
            # Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi,            
        ]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
        print(f'Using device: {device}')

        # Model Parameters 
        Model_Parameters = {
            'in_main_channels': (1,),
            'in_node_channels': 5   ,
            'in_edge_channels': 2   ,
            'in_aux_channels' : 0   ,
            'N_kernels'       : 16  ,
            'N_heads'         : 16  ,
            'N_dense_nodes'   : 64  ,
            'N_LSTM_nodes'    : 64  ,
            'N_LSTM_layers'   : 5   ,
            'kernel_size'     : 10  ,
            'conv2d_init_type': 'normal',
            'model_Dropout'   : 0.2 ,
        }
        
        Training_Parameters = {
            'LR': 0.0001,
            'epochs': 30,
            'BatchSize': 64,
            'accumulation_steps': 1,
            'epoch_done': 0,
            'batchBreak': 1e99,
            'ValLossIncreasePatience': 5,
            'Optimiser': 'Adam'
        }
        for Model in Models:
            model = Model(**Model_Parameters).to(device)
            if LoadModel and os.path.exists(ModelPath+model.Name+'.pt'):
                model = torch.load(ModelPath+model.Name+'.pt')
                tracker = torch.load(ModelPath+model.Name+'_Tracker.pt')
                print(f'Loaded Model: {model.Name}')

            print('Training Model')
            print()
            print('Model Description')
            print(model.Description)
            print()

            if Training_Parameters['Optimiser'] == 'Adam': optimiser = optim.Adam(model.parameters(), lr=Training_Parameters['LR'])
            if Training_Parameters['Optimiser'] == 'SGD' : optimizer = optim.SGD (model.parameters(), lr=Training_Parameters['LR'], momentum=0.9)
            gamma = 0.001**(1/30) if Training_Parameters['epochs']>30 else 0.001**(1/Training_Parameters['epochs']) # Reduce the LR by factor of 1000 over 30 epochs or less
            print(f'Gamma in LR Reduction: {gamma}')
            scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1)


            print('Training model: '     ,model.Name)
            print('Accumulation Steps: ' ,Training_Parameters['accumulation_steps'])
            Dataset.BatchSize =           Training_Parameters['BatchSize']
            print('Batch Size: '         ,Dataset.BatchSize)

            if plotSavePath != None : print(f'Plot Save Path: {plotSavePath}')
            else: print('Plots will not be saved')
            if LogPath != None: print(f'Log Path: {LogPath}')
            else: print('Logs will not be saved')

            model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,metric ,Tracker,device = device,\
                                plotOnEpochCompletionPath=plotSavePath,Training_Parameters=Training_Parameters,Model_Parameters=Model_Parameters,LogPath=LogPath)
            
            torch.save(model  ,SavePath+model.Name+'.pt')
            torch.save(tracker,SavePath+model.Name+'_Tracker.pt')
    
