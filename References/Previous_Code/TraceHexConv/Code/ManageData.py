##########################################
#               Take Data                #
#         Clean The Data of NaNs         #
#         Save A Smaller Tensor          #
##########################################





import torch
import paths
import os
os.system('clear')
# from matplotlib import pyplot as plt

# Prepare the paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# print(dir(Paths))
Paths.NormData = Paths.data_path + 'NormData/'

# Paths.PrintVariables(values=True)

if not os.path.exists(Paths.NormData):
    os.makedirs(Paths.NormData)




def ConvToSparse(Dense):
    indices = torch.nonzero(Dense,as_tuple=False).t().contiguous()
    values = Dense[indices[0],indices[1],indices[2],indices[3],indices[4]]
    return torch.sparse_coo_tensor(indices, values, Dense.size())



def ProcessData(type):

    print('Processing '+type+' Data')
    # Load Data
    print('Loading Data')
    D_main = torch.load(Paths.NormData + 'D_main_'+type+'.pt')
    D_main = D_main.to_dense()
    D_aux  = torch.load(Paths.NormData + 'D_aux_'+type+'.pt')
    logE   = torch.load(Paths.NormData + 'logE_'+type+'.pt')
    Core   = torch.load(Paths.NormData + 'Core_'+type+'.pt')
    Axis   = torch.load(Paths.NormData + 'Axis_'+type+'.pt')
    Xmax   = torch.load(Paths.NormData + 'Xmax_'+type+'.pt')
    EventMyId = torch.load(Paths.NormData + 'EventMyId_'+type+'.pt')

    # Find out where NaNs are
    print('Finding NaNs')
    D_main_nans_mask = torch.isnan(torch.sum(D_main, dim=tuple(range(1,D_main.dim()))))
    D_aux_nans_mask  = torch.isnan(torch.sum(D_aux , dim=tuple(range(1,D_aux.dim()))))
    logE_nans_mask   = torch.isnan(torch.sum(logE  , dim=tuple(range(1,logE.dim()))))
    Core_nans_mask   = torch.isnan(torch.sum(Core  , dim=tuple(range(1,Core.dim()))))
    Axis_nans_mask   = torch.isnan(torch.sum(Axis  , dim=tuple(range(1,Axis.dim()))))
    Xmax_nans_mask   = torch.isnan(torch.sum(Xmax  , dim=tuple(range(1,Xmax.dim()))))
    
    
    # Slap the masks together
    Total_nans_mask  = D_main_nans_mask | D_aux_nans_mask | logE_nans_mask | Core_nans_mask | Axis_nans_mask | Xmax_nans_mask
    print('Cutting NaNs')
    # Remove the NaNs
    D_main = D_main[~Total_nans_mask]
    D_aux  = D_aux[~Total_nans_mask]
    logE   = logE[~Total_nans_mask]
    Core   = Core[~Total_nans_mask]
    Axis   = Axis[~Total_nans_mask]
    Xmax   = Xmax[~Total_nans_mask]
    EventMyId = EventMyId[~Total_nans_mask]
    print('Making Samples')
    # Save Smaller Samples
    Original_Size = D_main.size(0)
    New_Size = int(Original_Size/10)

    indices = torch.randperm(Original_Size)[:New_Size]

    Samp_D_main = D_main[indices]
    Samp_D_aux  = D_aux[indices]
    Samp_logE   = logE[indices]
    Samp_Core   = Core[indices]
    Samp_Axis   = Axis[indices]
    Samp_Xmax   = Xmax[indices]
    Samp_EventMyId = EventMyId[indices]


    # Save Samples
    print('Saving Samples')
    Samp_D_main = ConvToSparse(Samp_D_main)

    torch.save(Samp_D_main, Paths.NormData + 'D_main_'+type+'_Samp.pt')
    torch.save(Samp_D_aux, Paths.NormData + 'D_aux_'+type+'_Samp.pt')
    torch.save(Samp_logE, Paths.NormData + 'logE_'+type+'_Samp.pt')
    torch.save(Samp_Core, Paths.NormData + 'Core_'+type+'_Samp.pt')
    torch.save(Samp_Axis, Paths.NormData + 'Axis_'+type+'_Samp.pt')
    torch.save(Samp_Xmax, Paths.NormData + 'Xmax_'+type+'_Samp.pt')
    torch.save(Samp_EventMyId, Paths.NormData + 'EventMyId_'+type+'_Samp.pt')

    # Save Full Samples
    print('Saving Full Samples')
    D_main = ConvToSparse(D_main)

    torch.save(D_main, Paths.NormData + 'D_main_'+type+'.pt')
    torch.save(D_aux, Paths.NormData + 'D_aux_'+type+'.pt')
    torch.save(logE, Paths.NormData + 'logE_'+type+'.pt')
    torch.save(Core, Paths.NormData + 'Core_'+type+'.pt')
    torch.save(Axis, Paths.NormData + 'Axis_'+type+'.pt')
    torch.save(Xmax, Paths.NormData + 'Xmax_'+type+'.pt')
    torch.save(EventMyId, Paths.NormData + 'EventMyId_'+type+'.pt')

    print('Done with '+type+' data') 



for type in ['train','val','test']:
    ProcessData(type)