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

# Import the functions from the other files, this is just the selection of the used functions
import Geometry_DataGen

# Data Generation Functions
Pass_Main     = Pass_Main_DoNothing

Pass_Aux      = Geometry_DataGen.Aux_Descriptors

Pass_Truth    = Geometry_DataGen.Geometry_InCameraPlane_Axis

Pass_Rec      = Pass_Rec_DoNothing

Pass_Graph    = Geometry_DataGen.Standard_Graph_Conv3d_Traces

Pass_MetaData = Pass_MetaData_DoNothing

# Extra Cleaning, probably not needed
Clean_Data = Clean_Data_DoNothing