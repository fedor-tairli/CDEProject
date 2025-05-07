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
if 'tycho' in hostname:
    # Common folder is already in the path
    pass
elif 'tedtop' in hostname:
    print('Setting up paths for tedtop')
    sys.path.append('/home/fedor-tairli/work/CDEs/Dataset/')
else: 
    sys.path.append('/remote/tychodata/ftairli/work/Projects/Common/')

ModelPath = os.path.abspath('../Models')
sys.path.append(ModelPath)
# Dataset modules
from Dataset2 import DatasetContainer, ProcessingDatasetContainer
from DataGenFunctions import Pass_Main,Pass_Aux,Pass_Truth,Pass_Rec,Pass_Graph,Pass_MetaData,Clean_Data

def LoadProcessingDataset(Path_To_Data,Path_To_Proc_Data,RunNames,RecalculateDataset = False,NeedTraces = False,OptionalName = None):
    if OptionalName is None: OptionalName = 'CurrentProcessingDataset'
    '''Loads the dataset from the path and returns a ProcessingDatasetContainer'''
    # Check if path to data endswith '/'
    if Path_To_Data     .endswith('/'):Path_To_Data      = Path_To_Data[:-1]
    if Path_To_Proc_Data.endswith('/'):Path_To_Proc_Data = Path_To_Proc_Data[:-1]
    
    
    if (not RecalculateDataset) and (os.path.exists(Path_To_Proc_Data+f'/{OptionalName}.pt')):
        print(f'Loading Dataset {OptionalName}')
        Dataset = torch.load(Path_To_Proc_Data+f'/{OptionalName}.pt')
        # Dataset.Truth_Keys = ('x','y','z','SDPPhi','CEDist')
        # Dataset.Truth_Units =('','','','rad','m')
        # torch.save(Dataset,Path_To_Proc_Data+f'/{OptionalName}.pt')
        # print('Dataset Loaded, and saved with adjusted Truth_Keys and Truth_Units')
    else:
        RecalculateDataset = True

    if RecalculateDataset:
        print('Recalculating Dataset')
        GlobalDataset = DatasetContainer()
        GlobalDataset.Load(Path_To_Data,RunNames,LoadTraces=NeedTraces)
        Dataset = ProcessingDatasetContainer()
        Dataset.set_Name(GlobalDataset.Name)

        # Pass the data to the ProcessingDataset

        Pass_Main (GlobalDataset,Dataset)
        print()
        Pass_Aux  (GlobalDataset,Dataset)
        print()
        Pass_Truth(GlobalDataset,Dataset)
        print()
        Pass_Rec  (GlobalDataset,Dataset)
        print()
        Pass_Graph(GlobalDataset,Dataset)
        print()
        Pass_MetaData(GlobalDataset,Dataset)
        print()

        # Save the dataset
        Dataset.Save(Path_To_Proc_Data,Name = OptionalName)
        print(f'Dataset used graphs = {Dataset.GraphData}')
    Clean_Data(Dataset)
    return Dataset

TestingThings = False
if __name__ == '__main__' and TestingThings:
    # Reading the dataset    
    Path_To_Data      = os.path.abspath('../../Data/Processed/')
    Path_To_Proc_Data = os.path.abspath('../Data/')
    RunNames = ['CDEsDataset']
    RecalculateDataset = True
    NeedTraces = True
    DatasetName = 'SDP_Dataset'
    Dataset = LoadProcessingDataset(Path_To_Data,Path_To_Proc_Data,RunNames,RecalculateDataset = RecalculateDataset,NeedTraces = NeedTraces,OptionalName = DatasetName)

    for data in Dataset._Main:
        # replace the nan values with 0
        data[torch.isnan(data)] = 0
    
    Dataset.Save(Path_To_Proc_Data,Name = DatasetName)

if __name__ == '__main__' and not TestingThings:

    # Flags
    Set_Custom_Seed      = False
    Use_Test_Set         = False
    Use_All_Sets         = True
    Dataset_RandomIter   = True
    RecalculateDataset   = False
    NeedTraces           = True
    LoadModel            = False
    DoNotTrain           = False
    DatasetName          = 'SDP_Dataset' #No / or .pt JUST NAME, eg GraphStructure  Use None to save as default


    if DoNotTrain: assert RecalculateDataset, 'Recalculate Dataset must be True if DoNotTrain is True'

    if Set_Custom_Seed:
        seed = 1234
        print('      Setting Custom Seed')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save Paths
    SavePath     = os.path.abspath('../Models/')
    plotSavePath = os.path.abspath('../Results/TrainingPlots/')
    LogPath      = os.path.abspath('../../TrainingLogs/')
    # Check that all the paths exist
    assert os.path.exists(SavePath)     , f'SavePath {SavePath} does not exist'
    assert os.path.exists(plotSavePath) , f'plotSavePath {plotSavePath} does not exist'
    assert os.path.exists(LogPath)      , f'LogPath {LogPath} does not exist'


    if plotSavePath != None:  # Purge the directory
        os.system(f'rm -r {plotSavePath}')
        os.system(f'mkdir {plotSavePath}')

    # Reading the dataset    
    Path_To_Data      = os.path.abspath('../../Data/Processed/')
    Path_To_Proc_Data = os.path.abspath('../Data/')
    
    RunNames = ['CDEsDataset']

    if DoNotTrain: print('No Training will be done, Just Reading the Dataset')
    Dataset = LoadProcessingDataset(Path_To_Data,Path_To_Proc_Data,RunNames,RecalculateDataset = RecalculateDataset,NeedTraces = NeedTraces,OptionalName = DatasetName)
    Dataset.AssignIndices()
    Dataset.RandomIter = Dataset_RandomIter

    
    if not DoNotTrain:
        # import model
        from TrainingModule import Train , Tracker
        from Model_SDP import Loss as Loss_function
        from Model_SDP import validate, metric
        from Model_SDP import Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta , Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi

        # Models = [Model_XmaxE_Conv_3d_Distances_JustXmax, Model_XmaxE_Conv_3d_Distances_JustLogE] # Here for copilot to see
        Models = [Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta, Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
        print(f'Using device: {device}')

        # Model Parameters 
        Model_Parameters = {
            'in_main_channels': (2,),
            'in_node_channels': 5   ,
            'in_edge_channels': 2   ,
            'in_aux_channels' : 2   ,
            'N_kernels'       : 16  ,
            'N_heads'         : 16  ,
            'N_dense_nodes'   : 128 ,
            'N_LSTM_nodes'    : 64  ,
            'N_LSTM_layers'   : 5   ,
            'kernel_size'     : 10  ,
            'conv2d_init_type': 'normal',
            'model_Dropout'   : 0.2 ,
        }
        
        Training_Parameters = {
            'LR': 0.00001,
            'epochs': 25,
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
            scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=False)


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
    
