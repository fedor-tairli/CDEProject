
# This file is basically a config file 
# Select the functions that will be used in the data generation process
# For the Processing Dataset


# PlaceHolders for functions
def Pass_Main_Example(Dataset,ProcessingDataset):
    pass

def Pass_Aux_Example(Dataset,ProcessingDataset):
    pass

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




import SDP_Graph_DataGen
import Geom_Graph_DataGen
import Axis_Graph_DataGen




# Data Generation Functions
Pass_Main     = Axis_Graph_DataGen.Graph_Edges_wStation_wAngVelAsNodeAndEdge_Axis

Pass_Aux      = Pass_Aux_Example

Pass_Truth    = Pass_Truth_Example

Pass_Rec      = Pass_Rec_Example

Pass_Graph    = Pass_Graph_Example

Pass_MetaData = Pass_MetaData_Example


# Extra Cleaning, probably not needed
Clean_Data = Clean_Data_Example