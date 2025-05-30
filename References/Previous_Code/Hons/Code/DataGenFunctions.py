###############################################################
#                                                             #
#           A set of functions for generating data            #
#        Need to take Dataset and ProcessingDataset           #
#          Return Nothing, Pass Data between 2 sets           #
#                   Necessary functions are :                 #
#                      - Pass_Main                            #
#                      - Pass_Aux                             #
#                      - Pass_Truth                           #
#                      - Pass_Rec                             #
#                      - Pass_Graph                           #
#                                                             # 
###############################################################

import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly

# Auxilary functions
def IndexToXY(indices,return_tensor=False):
    indices -=1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()




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

# Once Defined, the functions are to be unchanged


# Chunk for the standard CNN model
def Main_Conv2d_Grid_ChargeSigOnly(Dataset,ProcessingDataset):
    raise 'THis is bugged'
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,22,20) where C is the number of channels
    Selects only the last telescope signal
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),1,20,22))

    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        
        TelIDs        = Event.get_pixel_values('TelID')
        PulseCentroid = Event.get_pixel_values('PulseCentroid')
        Status        = Event.get_pixel_values('Status')

        # Take a shortcut here and instead of looking for actually last telescope, will just take the telescope of the last pixel with status 4
        # find index of the last pixel with status 4
        SelectedTelescopeID = TelIDs[torch.argmax(PulseCentroid[Status == 4])]
        
        PixelIDs      = Event.get_pixel_values('PixelID')[(TelIDs == SelectedTelescopeID)*(Status>0)]
        Charge        = Event.get_pixel_values('Charge')[(TelIDs == SelectedTelescopeID)*(Status>0)]
        # Normalise Charge
        Charge = torch.log10(Charge+1)/3.75
        Xs,Ys         = IndexToXY(PixelIDs)
        Main[i,0,Xs,Ys] = Charge
    print()
    # Pass the data to the ProcessingDataset
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Aux_No_Aux(Dataset,ProcessingDataset):
    ''' Function Defines no auxilary data to be passed.
    More Precisely, as Aux is required, this function will return zerose tensor
    '''

    ProcessingDataset._Aux = torch.zeros((len(Dataset),1))

def Unnormalise_Truth(Truth): # Just for Truth Func below
        Gen_SDPTheta = torch.acos(Truth[:,0])
        Gen_SDPPhi   = torch.asin(Truth[:,1])
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)
    
def Truth_Just_SDP_single(Dataset,ProcessingDataset):
    '''Gets just the SDP values,
    Does not use atan2 as unnormalisation (<- what single means)
    '''
    IDsList = ()
    Offsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    
    SelectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        # Check if only HEAT (Check if any EyeID is not equal to 5)
        if torch.sum(EyeIDs != 5) == 0:
            continue

        Mask = (TelIDs != 7)*(TelIDs != 8)*(TelIDs != 9)
        if torch.sum(Mask) == 0: # Should be impossible cause of the above but who knows?
            continue
        SelectedTelescopeID   = TelIDs[Mask].int().bincount().argmax()
        SelectedTelescopes[i] = SelectedTelescopeID
        Gen_SDPPhi  [i]       = Event.get_value('Gen_SDPPhi')
        Gen_SDPTheta[i]       = Event.get_value('Gen_SDPTheta')
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
    print()

    # Normalise Theta
    Gen_SDPTheta = torch.cos(Gen_SDPTheta)
    Rec_SDPTheta = torch.cos(Rec_SDPTheta)

    # Adjust Phi to be centred around mirror
    Gen_SDPPhi   = Gen_SDPPhi+2*torch.pi*(Gen_SDPPhi<0)
    Gen_SDPPhi   -= torch.pi
    Rec_SDPPhi   = Rec_SDPPhi+2*torch.pi*(Rec_SDPPhi<0)
    Rec_SDPPhi   -= torch.pi
    for i in range(1,7): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
    # Normalise Phi
    Gen_SDPPhi   = torch.sin(Gen_SDPPhi)
    Rec_SDPPhi   = torch.sin(Rec_SDPPhi)


    ProcessingDataset._Truth = torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
    ProcessingDataset.Truth_Keys = ('SDPTheta','SDPPhi')
    ProcessingDataset.Truth_Units =('rad','rad')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Unnormalise_Geom(Truth): # Just for Truth Func below
    Truth[:,0] = torch.acos(Truth[:,0])
    Truth[:,1] *=30000
    Truth[:,2] *=22000
    return Truth

def Truth_Just_Geometry(Dataset,ProcessingDataset):
    '''Gets the values for Chi0,Rp,T0'''

    IDsList = ()
    Gen_Chi0 = torch.zeros(len(Dataset))
    Gen_Rp   = torch.zeros(len(Dataset))
    Gen_T0   = torch.zeros(len(Dataset))

    Rec_Chi0 = torch.zeros(len(Dataset))
    Rec_Rp   = torch.zeros(len(Dataset))
    Rec_T0   = torch.zeros(len(Dataset))

    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        if i%100 == 0:print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        Gen_Chi0[i] = Event.get_value('Gen_Chi0').item()
        Gen_Rp  [i] = Event.get_value('Gen_Rp').item()
        Gen_T0  [i] = Event.get_value('Gen_T0').item()
        
        Rec_Chi0[i] = Event.get_value('Rec_Chi0').item()
        Rec_Rp  [i] = Event.get_value('Rec_Rp').item()
        Rec_T0  [i] = Event.get_value('Rec_T0').item()

    # Normalisation
    Gen_Chi0 = torch.cos(Gen_Chi0)
    Rec_Chi0 = torch.cos(Rec_Chi0)

    Gen_Rp  /= 30000
    Rec_Rp  /= 30000

    Gen_T0  /=22000
    Rec_T0  /=22000

    ProcessingDataset._Truth = torch.stack((Gen_Chi0,Gen_Rp,Gen_T0),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_Chi0,Rec_Rp,Rec_T0),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_Geom
    ProcessingDataset.Truth_Keys = ('Chi0','Rp','T0')
    ProcessingDataset.Truth_Units =('rad','m','100 ns')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Rec_Just_SDP_single(Dataset,ProcessingDataset):
    pass
    # Handled in the Truth Function for Faster Processing

def Rec_Just_Geometry(Dataset,ProcessingDataset):
    pass
    # Handled in the Truth Function for Faster Proc

def Graph_No_Graph(Dataset,ProcessingDataset):
    ''' Function Defines no graph data to be passed.
    Graphs are not required for the dataset to work, so no graph data
    '''
    ProcessingDataset.GraphData = False
    

def Main_Conv2d_Grid_Charge_and_Time(Dataset,ProcessingDataset):
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22) where C is the number of channels
    Selects only the hottest telescope
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),2,20,22))

    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        PulseCentroid = Event.get_pixel_values('PulseCentroid')
        
        # Check if only HEAT (Check if any EyeID is not equal to 5)
        if torch.sum(EyeIDs != 5) == 0:
            continue
        
        Mask = (TelIDs != 7)*(TelIDs != 8)*(TelIDs != 9)
        if torch.sum(Mask) == 0: # Should be impossible cause of the above but who knows?
            continue
        SelectedTelescopeID = TelIDs[Mask].int().bincount().argmax()
        
        PixelIDs      = Event.get_pixel_values('PixelID')[(TelIDs == SelectedTelescopeID)]
        Charge        = Event.get_pixel_values('Charge')[(TelIDs == SelectedTelescopeID)]
        Time          = PulseCentroid[(TelIDs == SelectedTelescopeID)]
        # Normalise
        Charge = torch.log10(Charge+1)/3.75
        Time   = (Time-375)/150
        Xs,Ys         = IndexToXY(PixelIDs)
        Main[i,0,Xs,Ys] = Charge
        Main[i,1,Xs,Ys] = Time

    print()
    # Pass the data to the ProcessingDataset
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

        
def Main_Conv2d_Grid_StationTimeDiff(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22)
    Selects only the hottest telescope
    Gives Charge on pixels and The Centroid - Station
    '''
    Main1 = torch.zeros((len(Dataset),3,20,22)) # Abs  Times

    IDsList = ()
    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        TelIDs = Event.get_pixel_values('TelID')
        EyeIDs = Event.get_pixel_values('EyeID')
        # Status = Event.get_pixel_values('Status')
        # Check if only HEAT (Check if any EyeID is not equal to 5)
        if torch.sum(EyeIDs != 5) == 0:
            continue
        Mask = (TelIDs != 7)*(TelIDs != 8)*(TelIDs != 9)
        if torch.sum(Mask) == 0: # Should be impossible cause of the above but who knows?
            continue
        SelectedTelescopeID = TelIDs[Mask].int().bincount().argmax()

        Mask = (TelIDs == SelectedTelescopeID)

        PixelIDs      = Event.get_pixel_values('PixelID')      [Mask]
        Charge        = Event.get_pixel_values('Charge')       [Mask]
        PulseCentroid = Event.get_pixel_values('PulseCentroid')[Mask]
        Chi_i         = Event.get_pixel_values('Chi_i')        [Mask]
        StationTime   = Event.get_value('Station_Time').item()
        # Normalise
        Xs,Ys  = IndexToXY(PixelIDs)
        Main1[i,0,Xs,Ys] = Charge
        Main1[i,1,Xs,Ys] = PulseCentroid - StationTime
        Main1[i,2,Xs,Ys] = Chi_i

    # Normalise Main1
    Main1[:,0,:,:] = torch.log10(Main1[:,0,:,:]+1)/3.75
    Main1[:,1,:,:] = Main1[:,1,:,:]/1000
    print()
    ProcessingDataset._Main.append(Main1)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Main_Conv3d_Traces(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel traces'''
    Main1 = torch.zeros((len(Dataset),100,20,22))
    Main2 = torch.zeros(len(Dataset),20,22)

    IDsList = ()
    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        EyeIDs = Event.get_pixel_values('EyeID')
        TelIDs = Event.get_pixel_values('TelID')
        PixIDs = Event.get_pixel_values('PixelID')
        Pstart = Event.get_pixel_values('PulseStart')


# Functions to be used during training are selected here
Pass_Main  = Main_Conv2d_Grid_StationTimeDiff
Pass_Aux   = Aux_No_Aux
Pass_Truth = Truth_Just_Geometry
Pass_Rec   = Rec_Just_Geometry
Pass_Graph = Graph_No_Graph
