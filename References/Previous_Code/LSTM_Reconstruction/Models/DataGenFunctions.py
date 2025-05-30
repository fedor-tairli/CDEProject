import torch
# This file is basically a config file 
# Select the functions that will be used in the data generation process
# For the Processing Dataset


# PlaceHolders for functions
def Pass_Main_DoNothing(Dataset,ProcessingDataset):
    pass

def Pass_Aux_DoNothing(Dataset,ProcessingDataset):
    if ProcessingDataset._Aux is None: ProcessingDataset._Aux = torch.zeros(len(Dataset))

def Pass_Truth_DoNothing(Dataset,ProcessingDataset):
    pass

def Pass_Rec_DoNothing(Dataset,ProcessingDataset):
    pass

def Pass_Graph_DoNothing(Dataset,ProcessingDataset):
    pass

def Pass_MetaData_DoNothing(Dataset,ProcessingDataset):
    pass

def Clean_Data_DoNothing(Dataset):
    pass




import Axis_LSTM_DataGen
# import Axis_Conv_DataGen




# Data Generation Functions
Pass_Main     = Axis_LSTM_DataGen.LSTM_Axis

Pass_Aux      = Pass_Aux_DoNothing

Pass_Truth    = Pass_Truth_DoNothing

Pass_Rec      = Pass_Rec_DoNothing

Pass_Graph    = Pass_Graph_DoNothing

Pass_MetaData = Pass_MetaData_DoNothing


# Extra Cleaning, probably not needed
Clean_Data = Clean_Data_DoNothing