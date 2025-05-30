# Importing the libraries
import os 
import sys
import time
import warnings
os.system('clear')
warnings.filterwarnings("ignore")


import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

hostname = os.uname()
if 'tycho'not in hostname:
    print('Setting up paths for remote')
    sys.path.append('/remote/tychodata/ftairli/work/Projects/Common/')

# Dataset modules
from Dataset2 import DatasetContainer, ProcessingDatasetContainer
from DataGenFunctions import Pass_Main,Pass_Aux,Pass_Truth,Pass_Rec,Pass_Graph

def LoadProcessingDataset(Path_To_Data,RunNames,RecalculateDataset = False):
    '''Loads the dataset from the path and returns a ProcessingDatasetContainer'''
    # Check if path to data endswith '/'
    if Path_To_Data.endswith('/'):Path_To_Data = Path_To_Data[:-1]
    
    
    if (not RecalculateDataset) and (os.path.exists(Path_To_Data+'/CurrentProcessingDataset.pt')):
        print('Loading Dataset')
        Dataset = ProcessingDatasetContainer(Path_To_Data)
    else:
        RecalculateDataset = True

    if RecalculateDataset:
        print('Recalculating Dataset')
        GlobalDataset = DatasetContainer()
        GlobalDataset.Load(Path_To_Data+'/RawData',RunNames)
        Dataset = ProcessingDatasetContainer()
        Dataset.set_Name(GlobalDataset.Name)

        # Pass the data to the ProcessingDataset

        Pass_Main (GlobalDataset,Dataset)
        Pass_Aux  (GlobalDataset,Dataset)
        Pass_Truth(GlobalDataset,Dataset)
        Pass_Rec  (GlobalDataset,Dataset)
        Pass_Graph(GlobalDataset,Dataset)

        # Save the dataset
        Dataset.Save(Path_To_Data)
    return Dataset



if __name__ == '__main__':

    # Flags
    Set_Custom_Seed      = False
    Use_Test_Set         = False
    Use_All_Sets         = False
    Dataset_RandomIter   = True
    RecalculateDataset   = False
    LoadModel            = False
    
    if Set_Custom_Seed:
        seed = 1234
        print('      Setting Custom Seed')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save Paths
    SavePath     = '/remote/tychodata/ftairli/work/Projects/ProfileReconstruction/Models/'
    plotSavePath = '/remote/tychodata/ftairli/work/Projects/ProfileReconstruction/Results/TrainingPlots/'
    if plotSavePath != None:  # Purge the directory
        os.system(f'rm -r {plotSavePath}')
        os.system(f'mkdir {plotSavePath}')

    # Reading the dataset    
    Path_To_Data = '/remote/tychodata/ftairli/work/Projects/ProfileReconstruction/Data/'
    
    if Use_Test_Set:
        RunNames = 'Test'
    else:
        if Use_All_Sets:
            RunNames = ['Run010','Run030','Run080','Run090']
        else:
            RunNames = 'Run010'

    Dataset = LoadProcessingDataset(Path_To_Data,RunNames,RecalculateDataset = RecalculateDataset)
    Dataset.AssignIndices()
    Dataset.RandomIter = Dataset_RandomIter

    # import model
    ModelPath = '/remote/tychodata/ftairli/work/Projects/ProfileReconstruction/Models/'
    sys.path.append(ModelPath)
    from TrainingModule import Train , Tracker
    from Model_Geom_Conv import Loss as Loss_function
    from Model_Geom_Conv import validate, metric
    from Model_Geom_Conv import Model_Geom_Conv_0 as Model


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'Using device: {device}')

    # Model Parameters 
    Model_Parameters = {
        'in_main_channels': (3,),
        'in_aux_channels': 1,
        'N_kernels': 32,
        'N_dense_nodes': 128,
        'conv2d_init_type': 'normal',
        'model_Dropout': 0
    }
    
    Training_Parameters = {
        'LR': 0.001,
        'epochs': 50,
        'BatchSize': 32,
        'accumulation_steps': 1,
        'epoch_done': 0,
        'batchBreak': 1e99,
        'ValLossIncreasePatience': 15,
        'Optimiser': 'Adam'
    }
    
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
    gamma = 0.001**(1/30) if Training_Parameters['epochs']>30 else 0.001**(1/Training_Parameters['epochs']) # Reduce the LR by factor of 1000 over 30 epochs or less
    print(f'Gamma in LR Reduction: {gamma}')
    scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=False)


    print('Training model: '     ,model.Name)
    print('Accumulation Steps: ' ,Training_Parameters['accumulation_steps'])
    Dataset.BatchSize =           Training_Parameters['BatchSize']
    print('Batch Size: '         ,Dataset.BatchSize)

    if plotSavePath != None : print(f'Plot Save Path: {plotSavePath}')
    else: print('Plots will not be saved')

    model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,metric ,Tracker,device = device,\
                          plotOnEpochCompletionPath=plotSavePath,Training_Parameters=Training_Parameters,Model_Parameters=Model_Parameters)
    
    torch.save(model  ,SavePath+model.Name+'.pt')
    torch.save(tracker,SavePath+model.Name+'_Tracker.pt')


