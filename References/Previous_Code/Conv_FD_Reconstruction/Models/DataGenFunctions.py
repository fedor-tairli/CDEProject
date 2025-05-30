import torch
# This file is basically a config file 
# Select the functions that will be used in the data generation process
# For the Processing Dataset


# PlaceHolders for functions
def Pass_Main_Example(Dataset,ProcessingDataset):
    pass

def Pass_Aux_Example(Dataset,ProcessingDataset):
    if ProcessingDataset._Aux is None: ProcessingDataset._Aux = torch.zeros(len(Dataset))

def Pass_Truth_Example(Dataset,ProcessingDataset):
    pass

def Pass_Rec_Example(Dataset,ProcessingDataset):
    pass

def Pass_Graph_Example(Dataset,ProcessingDataset):
    pass

def Pass_MetaData_Example(Dataset,ProcessingDataset):
    pass

def Clean_Data_Example(Dataset):
    pass




import SDP_Conv_DataGen
import Axis_Conv_DataGen




# Data Generation Functions
Pass_Main     = Axis_Conv_DataGen.Graph_Conv3d_Traces

Pass_Aux      = Axis_Conv_DataGen.Aux_Station

Pass_Truth    = Axis_Conv_DataGen.Truth_Axis

Pass_Rec      = Pass_Rec_Example

Pass_Graph    = Pass_Graph_Example

Pass_MetaData = Pass_MetaData_Example


# Extra Cleaning, probably not needed
Clean_Data = Clean_Data_Example