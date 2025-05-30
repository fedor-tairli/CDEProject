##########################################
#               Take Data                #
#         Clean The Data of NaNs         #
#         Save A Smaller Tensor          #
##########################################





import torch
import paths
import os
import numpy as np
os.system('clear')
# from matplotlib import pyplot as plt

# Prepare the paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# print(dir(Paths))
Paths.NormData = Paths.data_path + 'NormData/'

# Paths.PrintVariables(values=True)

if not os.path.exists(Paths.NormData):
    os.makedirs(Paths.NormData)


def ProcessData(DataSection):

    print('Processing '+DataSection+' Data')
    # Load Data
    print('Loading Data')
    EventMyId = torch.load(Paths.NormData + 'EventMyId_'+DataSection+'.pt')
    Main      = torch.load(Paths.NormData + 'Main_'+DataSection+'.pt')
    Aux       = torch.load(Paths.NormData + 'Aux_'+DataSection+'.pt')
    Truth     = torch.load(Paths.NormData + 'Truth_'+DataSection+'.pt')

    Original_Size = len(EventMyId)


# Nans
    # Find out where NaNs are
    print('Finding NaNs')
    Main_nans_mask   = torch.isnan(torch.sum(Main  , dim=tuple(range(1,Main.dim()))))
    Aux_nans_mask    = torch.isnan(torch.sum(Aux   , dim=tuple(range(1,Aux.dim()))))
    Truth_nans_mask  = torch.isnan(torch.sum(Truth , dim=tuple(range(1,Truth.dim()))))

    
    # Slap the masks together
    Total_nans_mask  = Main_nans_mask | Aux_nans_mask | Truth_nans_mask
    print('Cutting NaNs')
    # Remove the NaNs
    EventMyId = EventMyId[~Total_nans_mask]
    Main      = Main[~Total_nans_mask]
    Aux       = Aux[~Total_nans_mask]
    Truth     = Truth[~Total_nans_mask]

    
# Signal too high
    Main_signal_mask = np.any(Main.cpu().numpy() > 1.1, axis=(1, 2))
    Truth_signal_mask = np.any(Truth.cpu().numpy() > 1.1, axis=(1))
    Total_signal_mask = Main_signal_mask | Truth_signal_mask
    print('Cutting Saturated Signal Events')
    EventMyId = EventMyId[~Total_signal_mask]
    Main      = Main[~Total_signal_mask]
    Aux       = Aux[~Total_signal_mask]
    Truth     = Truth[~Total_signal_mask]

# Signal too low
    Main_signal_low_mask  = (Main.sum(dim=(1,2)).cpu().numpy() < 0.001)
    Truth_signal_low_mask = (Truth.sum(dim=(1)).cpu().numpy() < 0.001)
    Total_signal_low_mask = Main_signal_low_mask | Truth_signal_low_mask
    print('Cutting Low Signal Events')
    EventMyId = EventMyId[~Total_signal_low_mask]
    Main      = Main[~Total_signal_low_mask]
    Aux       = Aux[~Total_signal_low_mask]
    Truth     = Truth[~Total_signal_low_mask]


    # Save Full Samples
    print('Saving Full Samples')
    
    torch.save(EventMyId, Paths.NormData + 'EventMyId_'+DataSection+'.pt')
    torch.save(Main, Paths.NormData + 'Main_'+DataSection+'.pt')
    torch.save(Aux, Paths.NormData + 'Aux_'+DataSection+'.pt')
    torch.save(Truth, Paths.NormData + 'Truth_'+DataSection+'.pt')


    Final_Size = len(EventMyId)

    


    print('Done with '+DataSection+' data') 
    print('    Original Size: ',Original_Size)
    print('    Final Size:    ',Final_Size)




for DataSection in ['train','val','test']:
    ProcessData(DataSection)