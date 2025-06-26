import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly


# Once Defined, the functions are to be unchanged
########################################################################################################################

# Auxilary functions
def IndexToXY(indices,return_tensor=False):
    indices -=1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()





def Main_Conv2d_Grid_Charge_and_Time(Dataset,ProcessingDataset):
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22) where C is the number of channels
    Selects only the hottest telescope
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),2,20,22))
    if ProcessingDataset is None:
        Main = torch.zeros(10000,2,20,22)
    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        if i == 10000 and ProcessingDataset is None: break
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # There only should exist only one telescope. So no need to check for telescope selection
        PixelIDs      = Event.get_pixel_values('PixelID')
        TelIDs        = Event.get_pixel_values('TelID')
        Charge        = Event.get_pixel_values('Charge')
        PulseCentroid = Event.get_pixel_values('PulseCentroid')
        PulseCentroid_ZeroMask = PulseCentroid != 0
        # Normalise
        Charge = torch.log10(torch.clamp_min(Charge,0)+1)/3.75
        Time   = PulseCentroid[PulseCentroid_ZeroMask]
        Time = Time - torch.min(Time)
        Time = Time / 40

        Xs,Ys         = IndexToXY(PixelIDs-(TelIDs-1)*440+1,return_tensor=True)
        Main[i,0,Xs,Ys] = Charge
        Xs = Xs[PulseCentroid_ZeroMask]
        Ys = Ys[PulseCentroid_ZeroMask]
        Main[i,1,Xs,Ys] = Time

    # Pass the data to the ProcessingDataset
    if ProcessingDataset is None:
        return Main
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'



def Unnormalise_SDP(Truth): 
    # Just for Truth Func below
        Gen_SDPTheta = torch.acos(Truth[:,0])
        Gen_SDPPhi   = torch.asin(Truth[:,1])
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)



def Truth_Just_SDP_single(Dataset,ProcessingDataset):
    '''Gets just the SDP values,
    Does not use atan2 as unnormalisation (<- what single means)
    '''
    IDsList = ()
    Offsets = {1:44.45/180*torch.pi,2:89.87/180*torch.pi,3:132.83/180*torch.pi}#,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    
    SelectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        SelectedTelescopeID   = TelIDs.int().bincount().argmax()
        
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
    for i in range(1,4): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
    # Normalise Phi
    Gen_SDPPhi   = torch.sin(Gen_SDPPhi)
    Rec_SDPPhi   = torch.sin(Rec_SDPPhi)

    if ProcessingDataset is None:
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1), torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)
    ProcessingDataset._Truth = torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
    ProcessingDataset.Truth_Keys = ('SDPTheta','SDPPhi')
    ProcessingDataset.Truth_Units =('rad','rad')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Truth_Just_SDP_single_InvertedBehind(Dataset,ProcessingDataset):
    '''Gets just the SDP values,
    Does not use atan2 as unnormalisation (<- what single means)
    Inverting the Normal Vectors for events where shower lands behind camera
    '''
    IDsList = ()
    Offsets = {1:44.45/180*torch.pi,2:89.87/180*torch.pi,3:132.83/180*torch.pi}#,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    Offsets = {1:(43.92 -89.34)/180*torch.pi, 2:0, 3:(132.30 -89.34)/180*torch.pi}

    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    
    SelectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        SelectedTelescopeID   = TelIDs.int().bincount().argmax()
        
        SelectedTelescopes[i] = SelectedTelescopeID
        Gen_SDPPhi  [i]       = Event.get_value('Gen_SDPPhi')
        Gen_SDPTheta[i]       = Event.get_value('Gen_SDPTheta')
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
    print()


    # Adjust Phi to be centred around mirror
    for i in range(1,4): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]

    # Do the inversion of the normal vectors for events where shower lands behind camera
    IsBehind = (Gen_SDPPhi <torch.pi/2) & (Gen_SDPPhi >torch.pi/2)
    Gen_SDPPhi[IsBehind] += torch.pi
    Gen_SDPPhi[Gen_SDPPhi >  torch.pi] -= 2*torch.pi
    Gen_SDPPhi[Gen_SDPPhi < -torch.pi] += 2*torch.pi
    Rec_SDPPhi[IsBehind] += torch.pi
    Rec_SDPPhi[Rec_SDPPhi >  torch.pi] -= 2*torch.pi
    Rec_SDPPhi[Rec_SDPPhi < -torch.pi] += 2*torch.pi

    # Normalise Theta
    Gen_SDPTheta[IsBehind] = torch.pi - Gen_SDPTheta[IsBehind]
    Rec_SDPTheta[IsBehind] = torch.pi - Rec_SDPTheta[IsBehind] # No need to place bounds, it never happens

    Gen_SDPPhi   = Gen_SDPPhi+2*torch.pi*(Gen_SDPPhi<0)
    Gen_SDPPhi   -= torch.pi
    Rec_SDPPhi   = Rec_SDPPhi+2*torch.pi*(Rec_SDPPhi<0)
    Rec_SDPPhi   -= torch.pi

    # Normalise Theta
    Gen_SDPTheta = torch.cos(Gen_SDPTheta)
    Rec_SDPTheta = torch.cos(Rec_SDPTheta)
    # Normalise Phi
    Gen_SDPPhi   = torch.sin(Gen_SDPPhi)
    Rec_SDPPhi   = torch.sin(Rec_SDPPhi)

    if ProcessingDataset is None:
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1), torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)
    ProcessingDataset._Truth = torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
    ProcessingDataset.Truth_Keys = ('SDPTheta','SDPPhi')
    ProcessingDataset.Truth_Units =('rad','rad')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'




def Aux_Descriptors(Dataset, ProcessingDataset):
    ''' Will just provide some event descriptors for dependence inspection
    Values are : 'Event_Class', 'Primary', 'Gen_LogE', 'Gen_CosZenith', 'Gen_Xmax','Gen_Chi0', 'Gen_Rp'
    '''

    IDsList = ()
    Event_Class   = torch.zeros(len(Dataset),1)
    Primary       = torch.zeros(len(Dataset),1)
    Gen_LogE      = torch.zeros(len(Dataset),1)
    Gen_CosZenith = torch.zeros(len(Dataset),1)
    Gen_Xmax      = torch.zeros(len(Dataset),1)
    Gen_Chi0      = torch.zeros(len(Dataset),1)
    Gen_Rp        = torch.zeros(len(Dataset),1)

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Aux {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values
        Event_Class[i]   = Event.get_value('Event_Class')
        Primary[i]       = Event.get_value('Primary')
        Gen_LogE[i]      = Event.get_value('Gen_LogE')
        Gen_CosZenith[i] = Event.get_value('Gen_CosZenith')
        Gen_Xmax[i]      = Event.get_value('Gen_Xmax')
        Gen_Chi0[i]      = Event.get_value('Gen_Chi0')
        Gen_Rp[i]        = Event.get_value('Gen_Rp')

    
    if ProcessingDataset is None:
        return torch.stack(Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp)
    else:
        ProcessingDataset._Aux = torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp),dim=1)
        ProcessingDataset.Aux_Keys = ('Event_Class','Primary','LogE','CosZenith','Xmax','Chi0','Rp')
        ProcessingDataset.Aux_Units = ('','','','','g/cm^2','rad','m')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'




def Standard_Graph_Conv3d_Traces(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel traces'''
    IDsList = ()
    Graph = [] # Will have Event in dim1, [Trace,X,Y,PulseStart] in dim2, values of dim2 in dim3
    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        Traces = Event.get_trace_values()
        Pstart = Event.get_pixel_values('PulseStart')
        # Pstop  = Event.get_pixel_values('PulseStop')

        EyeID = Event.get_pixel_values('EyeID'  )
        TelID = Event.get_pixel_values('TelID'  )
        PixID = Event.get_pixel_values('PixelID')
        PixStatus = Event.get_pixel_values('Status')

        Xs,Ys = IndexToXY(PixID-(TelID-1)*440+1,return_tensor=True)

        # Traces normalised Here
        Traces = torch.log1p((Traces).clip(min=0))
        # Pstart normalised Here
        if len(Pstart)>0 : Pstart = Pstart - torch.min(Pstart)
        # Append to the GraphData
        Graph.append([Traces,Xs,Ys,Pstart])
        
        
    
    if ProcessingDataset is None:
        return Graph
    ProcessingDataset._Graph = Graph
    ProcessingDataset.GraphData = True
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'    




def Main_Conv2d_Grid_Charge_Status4Only(Dataset,ProcessingDataset):
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22) where C is the number of channels
    Selects only the hottest telescope
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),1,20,22))
    if ProcessingDataset is None:
        Main = torch.zeros(10000,1,20,22)
    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        if i == 10000 and ProcessingDataset is None: break
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # There only should exist only one telescope. So no need to check for telescope selection
        PixelIDs      = Event.get_pixel_values('PixelID')
        TelIDs        = Event.get_pixel_values('TelID')
        Charge        = Event.get_pixel_values('Charge')
        Status        = Event.get_pixel_values('Status')
        
        PixelIDs = PixelIDs[Status == 4]
        TelIDs   = TelIDs  [Status == 4]
        Charge   = Charge  [Status == 4]


        Charge = torch.log10(torch.clamp_min(Charge,0)+1)/3.75
        
        Xs,Ys         = IndexToXY(PixelIDs-(TelIDs-1)*440+1,return_tensor=True)
        Main[i,0,Xs,Ys] = Charge

    # Pass the data to the ProcessingDataset
    if ProcessingDataset is None:
        return Main
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'



def Truth_Just_SDP_single_DropPhiBehind(Dataset,ProcessingDataset):
    '''Gets just the SDP values,
    Does not use atan2 as unnormalisation (<- what single means)
    Drops the events where shower lands behind camera
    '''
    IDsList = ()
    Offsets = {1:44.45/180*torch.pi,2:89.87/180*torch.pi,3:132.83/180*torch.pi}#,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    
    SelectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        SelectedTelescopeID   = TelIDs.int().bincount().argmax()
        
        SelectedTelescopes[i] = SelectedTelescopeID
        Gen_SDPPhi  [i]       = Event.get_value('Gen_SDPPhi')
        Gen_SDPTheta[i]       = Event.get_value('Gen_SDPTheta')
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
    print()

    IDsList = torch.tensor(IDsList)
    Bad_Phi_Mask = (Gen_SDPPhi <= torch.pi/2) & (Gen_SDPPhi >= -torch.pi/2) # These are behind
    Bad_Phi_Mask = ~Bad_Phi_Mask # Invert the mask to keep only the events where shower lands in front of the camera
    Gen_SDPPhi   = Gen_SDPPhi[Bad_Phi_Mask]
    Gen_SDPTheta = Gen_SDPTheta[Bad_Phi_Mask]
    Rec_SDPPhi   = Rec_SDPPhi[Bad_Phi_Mask]
    Rec_SDPTheta = Rec_SDPTheta[Bad_Phi_Mask]
    SelectedTelescopes = SelectedTelescopes[Bad_Phi_Mask]
    IDsList = IDsList[Bad_Phi_Mask]

    # Move the IDsList to a tuple
    IDsList = tuple(IDsList.tolist())


    # Normalise Theta
    Gen_SDPTheta = torch.cos(Gen_SDPTheta)
    Rec_SDPTheta = torch.cos(Rec_SDPTheta)

    # Adjust Phi to be centred around mirror
    Gen_SDPPhi   = Gen_SDPPhi+2*torch.pi*(Gen_SDPPhi<0)
    Gen_SDPPhi   -= torch.pi
    Rec_SDPPhi   = Rec_SDPPhi+2*torch.pi*(Rec_SDPPhi<0)
    Rec_SDPPhi   -= torch.pi
    for i in range(1,4): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
    # Normalise Phi
    Gen_SDPPhi   = torch.sin(Gen_SDPPhi)
    Rec_SDPPhi   = torch.sin(Rec_SDPPhi)

    if ProcessingDataset is None:
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1), torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)
    ProcessingDataset._Truth = torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
    ProcessingDataset.Truth_Keys = ('SDPTheta','SDPPhi')
    ProcessingDataset.Truth_Units =('rad','rad')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

    
def Main_Conv2d_Grid_Charge_and_Time_DropPhiBehind(Dataset,ProcessingDataset):
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22) where C is the number of channels
    Selects only the hottest telescope
    Also checks what the Phi is and drops the events where shower lands behind camera
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),2,20,22))
    Gen_Phis = torch.zeros((len(Dataset)))
    if ProcessingDataset is None:
        Main = torch.zeros(10000,2,20,22)
        Gen_Phis = torch.zeros(10000,1)
    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        if i == 10000 and ProcessingDataset is None: break
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # There only should exist only one telescope. So no need to check for telescope selection
        PixelIDs      = Event.get_pixel_values('PixelID')
        TelIDs        = Event.get_pixel_values('TelID')
        Charge        = Event.get_pixel_values('Charge')
        PulseCentroid = Event.get_pixel_values('PulseCentroid')
        PulseCentroid_ZeroMask = PulseCentroid != 0
        # Normalise
        Charge = torch.log10(torch.clamp_min(Charge,0)+1)/3.75
        Time   = PulseCentroid[PulseCentroid_ZeroMask]
        Time = Time - torch.min(Time)
        Time = Time / 40

        Xs,Ys         = IndexToXY(PixelIDs-(TelIDs-1)*440+1,return_tensor=True)
        Main[i,0,Xs,Ys] = Charge
        Xs = Xs[PulseCentroid_ZeroMask]
        Ys = Ys[PulseCentroid_ZeroMask]
        Main[i,1,Xs,Ys] = Time
        Gen_Phis[i] = Event.get_value('Gen_SDPPhi')

    Bad_Phi_Mask = (Gen_Phis <= torch.pi/2) & (Gen_Phis >= -torch.pi/2) # These are behind
    Bad_Phi_Mask = ~Bad_Phi_Mask # Invert the mask to keep only the events where shower lands in front of the camera
    Main = Main[Bad_Phi_Mask]
    IDsList = torch.tensor(IDsList)[Bad_Phi_Mask]   
    IDsList = tuple(IDsList.tolist()) # Move the IDsList to a tuple

    # Make the Cut on t
    # Pass the data to the ProcessingDataset
    if ProcessingDataset is None:
        return Main
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Aux_Descriptors_DropPhiBehind(Dataset, ProcessingDataset):
    ''' Will just provide some event descriptors for dependence inspection
    Values are : 'Event_Class', 'Primary', 'Gen_LogE', 'Gen_CosZenith', 'Gen_Xmax','Gen_Chi0', 'Gen_Rp'
    '''

    IDsList = ()
    Event_Class   = torch.zeros(len(Dataset),1)
    Primary       = torch.zeros(len(Dataset),1)
    Gen_LogE      = torch.zeros(len(Dataset),1)
    Gen_CosZenith = torch.zeros(len(Dataset),1)
    Gen_Xmax      = torch.zeros(len(Dataset),1)
    Gen_Chi0      = torch.zeros(len(Dataset),1)
    Gen_Rp        = torch.zeros(len(Dataset),1)
    Gen_Phis      = torch.zeros(len(Dataset))

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Aux {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values
        Event_Class[i]   = Event.get_value('Event_Class')
        Primary[i]       = Event.get_value('Primary')
        Gen_LogE[i]      = Event.get_value('Gen_LogE')
        Gen_CosZenith[i] = Event.get_value('Gen_CosZenith')
        Gen_Xmax[i]      = Event.get_value('Gen_Xmax')
        Gen_Chi0[i]      = Event.get_value('Gen_Chi0')
        Gen_Rp[i]        = Event.get_value('Gen_Rp')
        Gen_Phis[i]      = Event.get_value('Gen_SDPPhi')

    Bad_Phi_Mask = (Gen_Phis <= torch.pi/2) & (Gen_Phis >= -torch.pi/2) # These are behind
    Bad_Phi_Mask = ~Bad_Phi_Mask # Invert the mask to
    # keep only the events where shower lands in front of the camera
    Event_Class   = Event_Class[Bad_Phi_Mask]
    Primary       = Primary[Bad_Phi_Mask]
    Gen_LogE      = Gen_LogE[Bad_Phi_Mask]
    Gen_CosZenith = Gen_CosZenith[Bad_Phi_Mask]
    Gen_Xmax      = Gen_Xmax[Bad_Phi_Mask]
    Gen_Chi0      = Gen_Chi0[Bad_Phi_Mask]
    Gen_Rp        = Gen_Rp[Bad_Phi_Mask]
    IDsList = torch.tensor(IDsList)[Bad_Phi_Mask]
    IDsList = tuple(IDsList.tolist()) # Move the IDsList to a tuple

    
    if ProcessingDataset is None:
        return torch.stack(Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp)
    else:
        ProcessingDataset._Aux = torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp),dim=1)
        ProcessingDataset.Aux_Keys = ('Event_Class','Primary','LogE','CosZenith','Xmax','Chi0','Rp')
        ProcessingDataset.Aux_Units = ('','','','','g/cm^2','rad','m')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'





## Some functions for the Truth Normalisation Below
# Some function needed for the next part
def spherical_to_cartesian(theta, phi):
    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=0)

def cartesian_to_spherical(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)  # polar angle
    phi = torch.atan2(y, x)    # azimuthal angle
    return theta, phi


def Unnormalise_3vector(Truth):
    return Truth   # Not Sure how to approach this, going to do this in post


def Truth_SDP_via3vector(Dataset,ProcessingDataset):
    ''' The truth is normalsed via 3-vector'''

    IDsList = ()
    Offsets = {1:44.45/180*torch.pi,2:89.87/180*torch.pi,3:132.83/180*torch.pi}#,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    
    SelectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        SelectedTelescopeID   = TelIDs.int().bincount().argmax()
        
        SelectedTelescopes[i] = SelectedTelescopeID
        Gen_SDPPhi  [i]       = Event.get_value('Gen_SDPPhi')
        Gen_SDPTheta[i]       = Event.get_value('Gen_SDPTheta')
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
    print()

    
    for i in range(1,4): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]

    # Compute the Cartesian Variant
    Gen_X,Gen_Y,Gen_Z = spherical_to_cartesian(Gen_SDPTheta,Gen_SDPPhi)
    Rec_X,Rec_Y,Rec_Z = spherical_to_cartesian(Rec_SDPTheta,Rec_SDPPhi)



    if ProcessingDataset is None:
        return torch.stack((Gen_X,Gen_Y,Gen_Z),dim=1), torch.stack((Rec_X,Rec_Y,Rec_Z),dim=1)
    ProcessingDataset._Truth = torch.stack((Gen_X,Gen_Y,Gen_Z),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_X,Rec_Y,Rec_Z),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
    ProcessingDataset.Truth_Keys = ('X','Y','Z')
    ProcessingDataset.Truth_Units =('','','')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'



