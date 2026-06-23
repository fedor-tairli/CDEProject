# Importing the libraries
import os 
import sys
import time
import warnings
os.system('clear')
warnings.filterwarnings("ignore")
import argparse

import torch 
import torch.optim as optim
import torch.utils.data as data
torch.set_printoptions(threshold=512, linewidth=512)

hostname = os.uname()
if 'tycho' in hostname:
    pass
elif 'tedtop' in hostname:
    print('Setting up paths for tedtop')
    sys.path.append('/home/fedor-tairli/work/CDEs/Dataset/')
elif 'ycho' in hostname: 
    sys.path.append('/remote/tychodata/ftairli/work/Projects/Common/')
else:
    sys.path.append('/cr/work/tairli/CDEs/Dataset/')

# -------------------------------------------------------------------------

ModelPath = os.path.abspath('../Models') + '/'
sys.path.append(ModelPath)

from Dataset2 import DatasetContainer, ProcessingDatasetContainer
from DataGenFunctions import Pass_Main,Pass_Aux,Pass_Truth,Pass_Rec,Pass_Graph,Pass_MetaData,Clean_Data

def LoadProcessingDataset(Path_To_Data, Path_To_Proc_Data, RunNames, RecalculateDataset=False, NeedTraces=False, OptionalName=None):
    if OptionalName is None: OptionalName = 'CurrentProcessingDataset'
    if Path_To_Data     .endswith('/'): Path_To_Data      = Path_To_Data[:-1]
    if Path_To_Proc_Data.endswith('/'): Path_To_Proc_Data = Path_To_Proc_Data[:-1]

    if (not RecalculateDataset) and (os.path.exists(Path_To_Proc_Data+f'/{OptionalName}.pt')):
        print(f'Loading Dataset {OptionalName}')
        Dataset = torch.load(Path_To_Proc_Data+f'/{OptionalName}.pt', weights_only=False)
    else:
        RecalculateDataset = True

    if RecalculateDataset:
        print('Recalculating Dataset')
        GlobalDataset = DatasetContainer()
        GlobalDataset.Load(Path_To_Data, RunNames, LoadTraces=NeedTraces)
        Dataset = ProcessingDatasetContainer()
        Dataset.set_Name(GlobalDataset.Name)

        Pass_Main (GlobalDataset, Dataset); print()
        Pass_Aux  (GlobalDataset, Dataset); print()
        Pass_Truth(GlobalDataset, Dataset); print()
        Pass_Rec  (GlobalDataset, Dataset); print()
        Pass_Graph(GlobalDataset, Dataset); print()
        Pass_MetaData(GlobalDataset, Dataset); print()

        Dataset.Save(Path_To_Proc_Data, Name=OptionalName)
        print(f'Dataset used graphs = {Dataset.GraphData}')
    Clean_Data(Dataset)
    return Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--bottlenecksize', type=int, default=None, help='Size of the bottleneck layer')
parser.add_argument('--selectnetwork',  type=int, default=None, help='Select Network to train')
args = parser.parse_args()
BottleNeckSize = args.bottlenecksize
SelectNetwork  = args.selectnetwork

# -------------------------------------------------------------------------
# Flags
Set_Custom_Seed      = False
Use_Test_Set         = False
Use_All_Sets         = True
Dataset_RandomIter   = True
RecalculateDataset   = False
NeedTraces           = True
LoadModel            = False
DoNotTrain           = False
# DatasetName          = 'XmaxEnergy_Conv3d_Dataset_SpoofedEnergy'
DatasetName          = 'XmaxEnergy_Conv3d_Dataset'

Debug_Mode           = False

if Debug_Mode: 
    Use_All_Sets = False
    Use_Test_Set = True

if DoNotTrain: assert RecalculateDataset, 'Recalculate Dataset must be True if DoNotTrain is True'

if Set_Custom_Seed:
    seed = 1234
    print('      Setting Custom Seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Save Paths
SavePath     = os.path.abspath('../Models/') + '/'
plotSavePath = None
LogPath      = os.path.abspath('../../TrainingLogs/') + '/'

assert os.path.exists(SavePath), f'SavePath {SavePath} does not exist'
assert os.path.exists(LogPath) , f'LogPath {LogPath} does not exist'

if plotSavePath != None:
    assert os.path.exists(plotSavePath), f'plotSavePath {plotSavePath} does not exist'
    os.system(f'rm -r {plotSavePath}')
    os.system(f'mkdir {plotSavePath}')

# Dataset paths
Path_To_Data      = os.path.abspath('../../Data/Processed/') + '/'
Path_To_Proc_Data = os.path.abspath('../Data') + '/'

if Use_Test_Set:
    RunNames = 'CDEsDataset'
    if "test" not in DatasetName.lower(): DatasetName += '_Test'
else:
    if Use_All_Sets:
        RunNames = ['CDEsDataset']
    else:
        RunNames = ['CDEsDataset']

if SelectNetwork == -1:
    DoNotTrain = True

if DoNotTrain: print('No Training will be done, Just Reading the Dataset')
Dataset = LoadProcessingDataset(Path_To_Data, Path_To_Proc_Data, RunNames, RecalculateDataset=RecalculateDataset, NeedTraces=NeedTraces, OptionalName=DatasetName)
Dataset.AssignIndices()
Dataset.RandomIter = Dataset_RandomIter

# -------------------------------------------------------------------------
# Manual Mask Manipulation. # This is not the best idea, but rough testing to see if it works at all

# Low_AngVel_Cut_Mask       = Dataset._Good_Event_Mask
# Dataset._Good_Event_Mask  = torch.ones_like(Dataset._Good_Event_Mask)

# -------------------------------------------------------------------------

if not DoNotTrain:
    import importlib
    import TrainingModule2 as TrainingModule
    importlib.reload(TrainingModule)
    Train   = TrainingModule.Train
    Tracker = TrainingModule.Tracker

    import Model_XmaxEnergy
    importlib.reload(Model_XmaxEnergy)

    Loss_function = Model_XmaxEnergy.Loss_class
    validate      = Model_XmaxEnergy.Validate_class
    metric        = Model_XmaxEnergy.Metric_class

    # Model_XmaxEnergy_Conv3d = Model_XmaxEnergy.Model_XmaxEnergy_Conv3d
    # Model_XmaxEnergy_Conv3d_withRejection = Model_XmaxEnergy.Model_XmaxEnergy_Conv3d_withRejection
    # Model_XmaxEnergy_Conv3d_withRejection_ForSpoofedDataset = Model_XmaxEnergy.Model_XmaxEnergy_Conv3d_withRejection_ForSpoofedDataset
    Model_XmaxEnergy_Conv3d_fromRejection = Model_XmaxEnergy.Model_XmaxEnergy_Conv3d_fromRejection

    Models = [
        # Model_XmaxEnergy_Conv3d,
        # Model_XmaxEnergy_Conv3d_withRejection,
        # Model_XmaxEnergy_Conv3d_withRejection_ForSpoofedDataset,
        Model_XmaxEnergy_Conv3d_fromRejection,
    ]

    if SelectNetwork is not None:
        assert SelectNetwork < len(Models), f'SelectNetwork {SelectNetwork} is out of range, max is {len(Models)-1}'
        Models = [Models[SelectNetwork]]
        print(f'Selected Model: {Models[0].Name}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    Model_Parameters = {
        'in_main_channels'        : (1,),
        'N_kernels'               : 24,
        'N_dense_nodes'           : 128,

        'OutWeights'              : torch.tensor([0.0, 1.0]),
        'Gate_Threshold'          : 0.75 ,

        'Train_Type'              : 'Default',
        'PredStyle'               : 'Default',

        'pixel_embedding_size'    : 32,
        'latent_space_size'       : 16 if BottleNeckSize is None else BottleNeckSize,
        'N_heads'                 : 4,
        'max_latent_iterations'   : 5,
        'latent_space_activation' : torch.nn.GELU,
        'in_node_channels'        : 5,
        'in_aux_channels'         : 0,
        'in_edge_channels'        : 2,
        'N_LSTM_nodes'            : 5,
        'N_LSTM_layers'           : 3,
        'kernel_size'             : 10,
        'conv2d_init_type'        : 'normal',
        'model_Dropout'           : 0.2,
        'Debug_Mode'              : Debug_Mode,
    }

    Training_Parameters = {
        'LR'                      : 0.0001,
        'epochs'                  : 150,
        'BatchSize'               : 32,
        'accumulation_steps'      : 1,
        'epoch_done'              : 0,
        'batchBreak'              : 1e99,
        'ValLossIncreasePatience' : 5,
        'Optimiser'               : 'AdamW',
        'Debug_Mode'              : Debug_Mode,

        'T_G_Loss_ratio'          : torch.tensor(1.0),
        'Rec_Loss_Weight'         : torch.tensor(1.0),

        'Train_Type'              : Model_Parameters['Train_Type'],
        'OutWeights'              : Model_Parameters['OutWeights'],
        'PredStyle'               : Model_Parameters['PredStyle'],
        'Gate_Threshold'          : Model_Parameters['Gate_Threshold'],



    }

    if not Set_Custom_Seed:
        seed = int(time.time())
        print(f'      Setting Random Seed to {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    for Model in Models:
        model = Model(**Model_Parameters).to(device)
        if LoadModel:
            if type(LoadModel) == bool and os.path.exists(ModelPath+model.Name+'.pt'):
                model   = torch.load(ModelPath+model.Name+'.pt')
                tracker = torch.load(ModelPath+model.Name+'_Tracker.pt')
                print(f'Loaded Model: {model.Name}')
            elif type(LoadModel) == str and os.path.exists(LoadModel):
                if LoadModel.endswith('Tracker.pt'):
                    donation_model_tracker = torch.load(LoadModel)
                    Model_state = donation_model_tracker.ModelStates[-1]
                    Model_Parameters['RegressionBlockWeighs'] = Model_state['RegressionBlockWeighs']
                    model = Model(**Model_Parameters).to(device)
                else:
                    donation_model = torch.load(LoadModel)
                    Model_Parameters['RegressionBlockWeighs'] = donation_model.state_dict()
                    model = Model(**Model_Parameters).to(device)
            else:
                print(f'Could not find model at {ModelPath+model.Name+".pt"}, training from scratch')

        print('Training Model')
        print()
        print('Model Description')
        print(model.Description)
        print()

        optimiser = optim.Adam(model.parameters(), lr=Training_Parameters['LR'])
        if Training_Parameters['Optimiser'] == 'SGD': optimizer = optim.SGD(model.parameters(), lr=Training_Parameters['LR'], momentum=0.9)

        gamma = 0.001**(1/30) if Training_Parameters['epochs'] > 30 else 0.001**(1/Training_Parameters['epochs'])
        print(f'Gamma in LR Reduction: {gamma}')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma, last_epoch=-1)

        print('Training model: '     , model.Name)
        print('Accumulation Steps: ' , Training_Parameters['accumulation_steps'])
        Dataset.BatchSize =            Training_Parameters['BatchSize']
        print('Batch Size: '         , Dataset.BatchSize)

        if plotSavePath != None: print(f'Plot Save Path: {plotSavePath}')
        else: print('Plots will not be saved')
        if (LogPath != None) and not Debug_Mode: print(f'Log Path: {LogPath}')
        else: print('Logs will not be saved')

        if Debug_Mode:
            print(f'\n\n Begining training in Debug Mode \n\n')

        model, tracker = Train(model, Dataset, optimiser, scheduler, Loss_function, validate, metric, Tracker, device=device,
                               plotOnEpochCompletionPath=plotSavePath, Training_Parameters=Training_Parameters, Model_Parameters=Model_Parameters, LogPath=LogPath)

        if not Debug_Mode:
            torch.save(model  , SavePath+model.Name+'.pt')
            torch.save(tracker, SavePath+model.Name+'_Tracker.pt')

        # break # This is for training only one model, removed for multiple trainings, (no real effective change)