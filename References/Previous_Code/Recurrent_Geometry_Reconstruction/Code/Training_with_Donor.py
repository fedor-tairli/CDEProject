# Importing the libraries
import os 
os.system('clear')
import torch 
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

hostname = os.uname()
if 'tycho'not in hostname:
    print('Setting up paths for remote')
    sys.path.append('/remote/tychodata/ftairli/work/Projects/Common/')

import time
import warnings
warnings.filterwarnings("ignore")
from Dataset import LSTMProcessingDatasetContainer, DatasetContainer
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

def GetTraces(dataset):
    assert dataset.HasTraces, 'Dataset doesnt have traces'
    shape = [len(dataset),1000,3]
    Traces = torch.zeros(shape) - 1
    
    
    for ev in range(len(dataset)):
        EvPixStart  = dataset._EventPixelPosition[ev,0]
        EvPixStop   = dataset._EventPixelPosition[ev,1]
        
        PulseStarts = dataset._pixelData[EvPixStart:EvPixStop,6]
        PulseStops  = dataset._pixelData[EvPixStart:EvPixStop,8]
        TimeOffsets = dataset._pixelData[EvPixStart:EvPixStop,11]
        Chis        = dataset._pixelData[EvPixStart:EvPixStop,5]
        TelIds      = dataset._pixelData[EvPixStart:EvPixStop,1]
        EyeIds      = dataset._pixelData[EvPixStart:EvPixStop,2]
        Status      = dataset._pixelData[EvPixStart:EvPixStop,3]
        EventTraces = dataset._pixelTraces[EvPixStart:EvPixStop]

        # Cut on Status 
        PulseStarts = PulseStarts[Status>2]
        PulseStops  = PulseStops[Status>2]
        TimeOffsets = TimeOffsets[Status>2]
        Chis        = Chis[Status>2]
        TelIds      = TelIds[Status>2]
        EyeIds      = EyeIds[Status>2]
        EventTraces = EventTraces[Status>2]
        Status      = Status[Status>2]

        # HeCo Events
        PulseStarts[TelIds>6] *= 0.5
        PulseStops[TelIds>6]  *= 0.5
        # HEAT Only Events
        PulseStarts[EyeIds == 5] *= 0.5
        PulseStops[EyeIds == 5]  *= 0.5
        PulseStarts = PulseStarts.to(torch.int32)
        PulseStops  = PulseStops.to(torch.int32)
        TimeOffsets = TimeOffsets.to(torch.int32)

        CombTraceStart = (PulseStarts+TimeOffsets).min().item()
        CombTraceStop  = (PulseStops+TimeOffsets).max().item()
        CombTrace      = torch.zeros(CombTraceStop-CombTraceStart)
        TraceChi       = torch.zeros(CombTraceStop-CombTraceStart)
        TraceChiW      = torch.zeros(CombTraceStop-CombTraceStart) # Will be weighed equally by the pixel count in the CombPulse
        for i in range(len(PulseStarts)):
            CombPulseStart = PulseStarts[i] + TimeOffsets[i] - CombTraceStart
            CombPulseStop  = PulseStops[i]  + TimeOffsets[i] - CombTraceStart
            CombTrace[CombPulseStart:CombPulseStop] += EventTraces[i][PulseStarts[i]:PulseStops[i]]
            TraceChi [CombPulseStart:CombPulseStop] += Chis[i] #*torch.clip(EventTraces[i][PulseStarts[i]:PulseStops[i]],0)
            TraceChiW[CombPulseStart:CombPulseStop] += 1       #torch.clip(EventTraces[i][PulseStarts[i]:PulseStops[i]],0)
            
        TraceChi[TraceChiW>0] /= TraceChiW[TraceChiW>0]
        
        # Normalise_Trace 
        CombTrace[CombTrace<=0] = -torch.log10(-CombTrace[CombTrace<=0] + 1)
        CombTrace[CombTrace>0]  = torch.log10(CombTrace[CombTrace>0] + 1)
        
        # Add to Traces 
        Traces[ev,-(CombTraceStop-CombTraceStart):,0] = CombTrace[-1000:]
        Traces[ev,-(CombTraceStop-CombTraceStart):,1] = TraceChi[-1000:]

        StationTime    = dataset.GetValues('StationTime')[ev]
        LastValue      = StationTime - CombTraceStop
        Traces[ev,:,2] = torch.arange(1000).flip(0) + LastValue
        Traces[ev,Traces[ev,:,0]==-1,2] = -1 
    return Traces


def GetAuxData(dataset):
    Station_Chi  = dataset.GetValues('StationChi')
    Station_Time = dataset.GetValues('StationTime')
    Station_Dist = dataset.GetValues('StationDistance')
    
    AllPixel_Chi     = dataset._pixelData[:,5]
    AllPixel_Centroid = dataset._pixelData[:,7]
    AllPixel_Status   = dataset._pixelData[:,3]
    AllPixel_Pos      = dataset._EventPixelPosition

    LastPixel_Chi     = torch.zeros(len(Station_Chi))
    LastPixel_Centroid= torch.zeros(len(Station_Chi))
    for i,iPix in enumerate(AllPixel_Pos[:,1]):
        iPix -= 1
        while AllPixel_Status[iPix] <= 2:
            iPix -= 1
        LastPixel_Chi[i]=AllPixel_Chi[iPix]
        LastPixel_Centroid[i]=AllPixel_Centroid[iPix]

    LastPixel_Station_ChiDiff = LastPixel_Chi - Station_Chi
    LastPixel_Station_TimeDiff= Station_Time - LastPixel_Centroid


    # Normalise the data
    # Station_Chi = Station_Chi
    Station_Time /= 1000
    Station_Dist /= 40000
    LastPixel_Station_ChiDiff_Mean = 0.16
    LastPixel_Station_ChiDiff_STD  = 0.15
    LastPixel_Station_ChiDiff = (LastPixel_Station_ChiDiff - LastPixel_Station_ChiDiff_Mean)/LastPixel_Station_ChiDiff_STD
    LastPixel_Station_TimeDiff = torch.log10(torch.clip(LastPixel_Station_TimeDiff,0)+1)

    # return torch.stack([Station_Chi,Station_Time,Station_Dist,LastPixel_Station_ChiDiff,LastPixel_Station_TimeDiff],dim=1)
    # Dont use the Station_Chi and Station_Time

    ALL_Chi0   = dataset.GetValues('RecChi0')
    ALL_Rp     = dataset.GetValues('RecRp')

    Mean_Rp     = 12800
    STD_Rp      = 5800

    Norm_Chi0   = torch.cos(ALL_Chi0)
    Norm_Rp     = (ALL_Rp-Mean_Rp)/STD_Rp

    return torch.stack([Station_Dist,LastPixel_Station_ChiDiff,LastPixel_Station_TimeDiff,Norm_Chi0,Norm_Rp],dim=1)


def LoadProcessingDataset(Path_To_Data,RunName,RecalculateDataset = False):
    '''Loads the dataset from the path and returns a ProcessingDatasetContainer'''
    # Check if path to data endswith '/'
    if Path_To_Data[-1] != '/':Path_To_Data += '/'
    # Check if Datasets exits, if not create them (files are : RunName_TraceInputs.pt,RunName_Truths.pt,RunName_AuxInputs.pt)


    if RecalculateDataset:
        if RunName == 'ALL':
            raise 'Multiple Run recalculation not supported yet'
        else:
            print(f'Calculating Dataset for {RunName}')
            Dataset = DatasetContainer(1)
            Dataset.Load(Path_To_Data+'RawData/',RunName)
            Dataset.LoadTraces(Path_To_Data+'RawData/',RunName)
            ProcessingDataset = Dataset.GetLSTMProcessingDataset(GetTraces,GetAuxData,GetTruths)
            ProcessingDataset.AssignIndices()
            ProcessingDataset.Save(Path_To_Data+'NormData/',RunName)
    else:
            print(f'Loading Dataset for {RunName}')
            ProcessingDataset = LSTMProcessingDatasetContainer()
            ProcessingDataset.Load(Path_To_Data+'NormData/',RunName)
        
    return ProcessingDataset      



if __name__ == '__main__':

    # Flags
    Set_Custom_Seed      = False
    Use_Test_Set         = False # Doesnt work with new validation. Need to fix
    Use_All_Sets         = True
    Dataset_RandomIter   = True
    RecalculateDataset   = False
    Train_for_difference = False

    if Set_Custom_Seed:
        seed = 1234
        print('      Setting Custom Seed')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Save Paths
    SavePath     = '/remote/tychodata/ftairli/work/Projects/Recurrent_Geometry_Reconstruction/Models/'
    plotSavePath = '/remote/tychodata/ftairli/work/Projects/Recurrent_Geometry_Reconstruction/Results/TrainingPlots/'
    if plotSavePath != None:  # Purge the directory
        os.system(f'rm -r {plotSavePath}')
        os.system(f'mkdir {plotSavePath}')

    ReTrainDonor       = False
    ReTrainRecipient   = True



    # Reading the dataset
        
    Path_To_Data = '/remote/tychodata/ftairli/work/Projects/Recurrent_Geometry_Reconstruction/Data/'
    
    if Use_Test_Set:
        RunNames = 'Test'
    else:
        if Use_All_Sets:
            RunNames = 'ALL'
        else:
            RunNames = 'Run010'

    Dataset = LoadProcessingDataset(Path_To_Data,RunNames,RecalculateDataset = RecalculateDataset)
    Dataset.AssignIndices()
    Dataset.RandomIter = Dataset_RandomIter

    # import model
    ModelPath = '/remote/tychodata/ftairli/work/Projects/Recurrent_Geometry_Reconstruction/Models/'
    sys.path.append(ModelPath)
    from TrainingModule import Train , Tracker
    from Model_3_0 import LossMSE as Loss_function
    from Model_3_0 import validate
    from Model_3_0 import Model_3_0_Donor     as DonorModel
    from Model_3_0 import Model_3_0_Recipient as RecipientModel


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'Using device: {device}')


    # Common Model Parameters
    in_channels = 3
    in_AuxData  = 3
    n_LSTM_nodes  = 32
    LSTM_layers = 3

    # DnorModel Parameters
    Donor_n_dense_nodes = 64
    Donor_dropout_prob  = 0.3

    # Recipient Model Parameters
    Recipient_n_dense_nodes = 64
    Recipient_dropout_prob  = 0.1

    Dataset.AuxInputs = Dataset.AuxInputs[:,:in_AuxData] # In Case there is more AuxData than needed
    if Train_for_difference: Dataset.Truths    = Dataset.Truths - Dataset.AuxInputs[:,-2:] # Training for the difference to Rec
    
    # Donor Model
    if ReTrainDonor:
        model = DonorModel(in_channels = in_channels,in_AuxData = in_AuxData,n_dense_nodes = Donor_n_dense_nodes,n_LSTM_nodes = n_LSTM_nodes,LSTM_layers = LSTM_layers,dropout_prob=Donor_dropout_prob,Dtype = torch.float32).to(device)
        print('Training Donor Model')
        print()
        print('Model Description')
        print(model.Description)
        print()
        LR = 1e-3
        epochs = 20 # May Need to increase/decrease, cause dont want to train the donor model too much

        optimiser = optim.Adam(model.parameters(), lr=LR)
        gamma = 0.001**(1/epochs) # Should Reduce the learning rate by a factor of 0.001 over the course of the training
        print(f'Gamma in LR Reduction: {gamma}')
        scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=False)

        BatchSize = 32
        accumulation_steps = 1
        epoch_done = 0
        batchBreak = 1e99

        print('Training model: ',model.Name)
        print('Accumulation Steps: ',accumulation_steps)
        print('Batch Size: ',BatchSize)

        if plotSavePath != None : print(f'Plot Save Path: {plotSavePath}')
        else: print('Plots will not be saved')

        model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,\
                        Tracker,Epochs=epochs,BatchSize=BatchSize,Accumulation_steps=accumulation_steps,device = device,batchBreak = batchBreak,\
                        normStateIn='Net',normStateOut='Net',plotOnEpochCompletionPath=plotSavePath)

        torch.save(model,SavePath+model.Name+'.pt')
        torch.save(tracker,SavePath+model.Name+'_Tracker.pt')
    
    if ReTrainRecipient:
        # Load Donor Model
        if os.path.exists(SavePath+DonorModel.Name+'.pt'):
            DonorModel = torch.load(SavePath+DonorModel.Name+'.pt')
            model      = RecipientModel(in_channels = in_channels,in_AuxData = in_AuxData,n_dense_nodes = Recipient_n_dense_nodes,n_LSTM_nodes = n_LSTM_nodes,LSTM_layers = LSTM_layers,dropout_prob=Recipient_dropout_prob,Dtype = torch.float32).to(device)
            model.TransplantWeights(DonorModel)
        else:
            print('No Donor Model Found')
            model      = RecipientModel(in_channels = in_channels,in_AuxData = in_AuxData,n_dense_nodes = Recipient_n_dense_nodes,n_LSTM_nodes = n_LSTM_nodes,LSTM_layers = LSTM_layers,dropout_prob=Recipient_dropout_prob,Dtype = torch.float32).to(device)
        
        print('Training Recipient Model')
        print()
        print('Model Description')
        print(model.Description)
        print()
        LR = 1e-3
        epochs = 90 # Expect Val Loss Exit way before that. 

        optimiser = optim.Adam(model.parameters(), lr=LR)
        gamma = 0.001**(1/30) if epochs>30 else 0.001**(1/epochs) # Reduce the LR by factor of 1000 over 30 epochs or less
        print(f'Gamma in LR Reduction: {gamma}')
        scheduler = torch.optim.lr_scheduler.ExponentialLR    (optimiser, gamma = gamma, last_epoch=-1, verbose=False)

        BatchSize = 32
        accumulation_steps = 1
        epoch_done = 0
        batchBreak = 1e99

        print('Training model: ',model.Name)
        print('Accumulation Steps: ',accumulation_steps)
        print('Batch Size: ',BatchSize)

        if plotSavePath != None : print(f'Plot Save Path: {plotSavePath}')
        else: print('Plots will not be saved')

        model,tracker = Train(model,Dataset,optimiser,scheduler,Loss_function,validate,\
                        Tracker,Epochs=epochs,BatchSize=BatchSize,Accumulation_steps=accumulation_steps,device = device,batchBreak = batchBreak,\
                        normStateIn='Net',normStateOut='Net',plotOnEpochCompletionPath=plotSavePath)
        
        torch.save(model,SavePath+model.Name+'.pt')
        torch.save(tracker,SavePath+model.Name+'_Tracker.pt')


if ReTrainDonor or ReTrainRecipient : print('Training Finished')
elif RecalculateDataset: print('Dataset Recalculated, No Training Done')
else: print('Nothing was done, What are you silly?')

