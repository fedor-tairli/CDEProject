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

def Unnormalise_Geom_and_SDP(Truth): # Just for Truth Func below
    Truth[:,0] = torch.acos(Truth[:,0])
    Truth[:,1] *=30000
    Truth[:,2] *=22000
    Truth[:,3] = torch.acos(Truth[:,3])
    Truth[:,4] = torch.asin(Truth[:,4])
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


def Truth_Geometry_and_SDP(Dataset,ProcessingDataset):
    '''Gets the values for Chi0,Rp,T0'''
    Offsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    IDsList = ()
    Gen_Chi0 = torch.zeros(len(Dataset))
    Gen_Rp   = torch.zeros(len(Dataset))
    Gen_T0   = torch.zeros(len(Dataset))
    Gen_Tht  = torch.zeros(len(Dataset))
    Gen_Phi  = torch.zeros(len(Dataset))

    Rec_Chi0 = torch.zeros(len(Dataset))
    Rec_Rp   = torch.zeros(len(Dataset))
    Rec_T0   = torch.zeros(len(Dataset))
    Rec_Tht  = torch.zeros(len(Dataset))
    Rec_Phi  = torch.zeros(len(Dataset))

    selectedTelescopes = torch.zeros(len(Dataset))
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        if i%100 == 0:print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        # Geometry
        Gen_Chi0[i] = Event.get_value('Gen_Chi0').item()
        Gen_Rp  [i] = Event.get_value('Gen_Rp').item()
        Gen_T0  [i] = Event.get_value('Gen_T0').item()
        
        Rec_Chi0[i] = Event.get_value('Rec_Chi0').item()
        Rec_Rp  [i] = Event.get_value('Rec_Rp').item()
        Rec_T0  [i] = Event.get_value('Rec_T0').item()

        # SDP
        TelIDs        = Event.get_pixel_values('TelID')
        EyeIDs        = Event.get_pixel_values('EyeID')
        
        # Check if only HEAT (Check if any EyeID is not equal to 5)
        if torch.sum(EyeIDs != 5) == 0:
            continue
        Mask = (TelIDs != 7)*(TelIDs != 8)*(TelIDs != 9)
        if torch.sum(Mask) == 0: # Should be impossible cause of the above but who knows?
            continue

        SelectedTelescopeID = TelIDs[Mask].int().bincount().argmax()
        selectedTelescopes[i] = SelectedTelescopeID
        Gen_Tht[i] = Event.get_value('Gen_SDPTheta')
        Gen_Phi[i] = Event.get_value('Gen_SDPPhi')
        Rec_Tht[i] = Event.get_value('Rec_SDPTheta')
        Rec_Phi[i] = Event.get_value('Rec_SDPPhi')



    # Normalisation
    Gen_Chi0 = torch.cos(Gen_Chi0)
    Rec_Chi0 = torch.cos(Rec_Chi0)

    Gen_Rp  /= 30000
    Rec_Rp  /= 30000

    Gen_T0  /=22000
    Rec_T0  /=22000

    Gen_Tht = torch.cos(Gen_Tht)
    Rec_Tht = torch.cos(Rec_Tht)

    Gen_Phi = Gen_Phi+2*torch.pi*(Gen_Phi<0)
    Gen_Phi -= torch.pi
    Rec_Phi = Rec_Phi+2*torch.pi*(Rec_Phi<0)
    Rec_Phi -= torch.pi
    for i in range(1,7): # Apply offsets
        Gen_Phi[SelectedTelescopeID == i] -= Offsets[i]
        Rec_Phi[SelectedTelescopeID == i] -= Offsets[i]
    Gen_Phi = torch.sin(Gen_Phi)
    Rec_Phi = torch.sin(Rec_Phi)

    ProcessingDataset._Truth = torch.stack((Gen_Chi0,Gen_Rp,Gen_T0,Gen_Tht,Gen_Phi),dim=1)
    ProcessingDataset._Rec   = torch.stack((Rec_Chi0,Rec_Rp,Rec_T0,Rec_Tht,Rec_Phi),dim=1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_Geom_and_SDP
    ProcessingDataset.Truth_Keys = ('Chi0','Rp','T0','SDPTheta','SDPPhi')
    ProcessingDataset.Truth_Units =('rad','m','100 ns','rad','rad')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Rec_Handled_by_Truth(Dataset,ProcessingDataset):
    print('Rec is handled by Truth')
    # Handled in the Truth Function for Faster Processing

def Rec_Just_Geometry(Dataset,ProcessingDataset):
    pass
    # Handled in the Truth Function for Faster Proc

def Graph_No_Graph(Dataset,ProcessingDataset):
    ''' Function Defines no graph data to be passed.
    Graphs are not required for the dataset to work, so no graph data
    '''
    ProcessingDataset.GraphData = False
    
def Main_No_Main(Dataset,ProcessingDataset):
    '''For when the Main is replaces by Graph'''
    print('Main is replaced by Graph')
    ProcessingDataset.GraphData = True
    pass


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


def Graph_Conv3d_Traces(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel traces'''
    IDsList = ()
    Graph = [] # Will have Event in dim1, [Trace,X,Y,PulseStart] in dim2, values of dim2 in dim3
    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        Traces = Event.get_trace_values()
        Pstart = Event.get_pixel_values('PulseStart')
        # Pstop  = Event.get_pixel_values('PulseStop')

        EyeID = Event.get_pixel_values('EyeID'  )
        TelID = Event.get_pixel_values('TelID'  )
        PixID = Event.get_pixel_values('PixelID')
        PixStatus = Event.get_pixel_values('Status')

        if torch.sum(EyeID != 5) == 0 or torch.sum(TelID<7) == 0:
            Mask = torch.zeros(len(EyeID)).bool()
        else:
            Mask =(EyeID != 5) * (TelID==torch.mode(TelID[TelID<7]).values.item()) * (PixStatus == 4)

        Traces = Traces[Mask]
        Pstart = Pstart[Mask]
        PixID  = PixID[Mask]

        Xs,Ys = IndexToXY(PixID,return_tensor=True)

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


def Main_LSTM_Trace(Dataset,ProcessingDataset):
    '''Will just provide full length trace for each event'''

    Main = torch.zeros(len(Dataset),1000,5) -1
    IDsList = ()
    EventsFailed = 0
    for i,Event in enumerate(Dataset):
        
        print(f'    Processing Main {i} / {len(Dataset)} | Bad Events : {EventsFailed}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        try:
            # Event Information

            Traces = Event.get_trace_values()
            Thetas = Event.get_pixel_values('Theta')
            Phis   = Event.get_pixel_values('Phi')
            PStart = Event.get_pixel_values('PulseStart')
            PStop  = Event.get_pixel_values('PulseStop')
            TimeOff= Event.get_pixel_values('TimeOffset')
            EyeID  = Event.get_pixel_values('EyeID')
            TelID  = Event.get_pixel_values('TelID')
            Status = Event.get_pixel_values('Status')

            Traces = Traces[Status>2]
            Thetas = Thetas[Status>2]
            Phis   = Phis[Status>2]
            PStart = PStart[Status>2]
            PStop  = PStop[Status>2]
            TimeOff= TimeOff[Status>2]
            EyeID  = EyeID[Status>2]
            TelID  = TelID[Status>2]
            Status = Status[Status>2]

            HEAT = ~((EyeID!=5)*(TelID<7))
            PStart[HEAT] *= 0.5
            PStop [HEAT] *= 0.5
            PStart = PStart.int()
            PStop  = PStop.int()
            TimeOff= TimeOff.int()

            # Make Trace

            CombTraceStart = (PStart+TimeOff).min().item()
            CombTraceStop  = (PStop +TimeOff).max().item()
            CombTrace      = torch.zeros(CombTraceStop-CombTraceStart)
            CombTheta      = torch.zeros(CombTraceStop-CombTraceStart)
            CombPhi        = torch.zeros(CombTraceStop-CombTraceStart)
            CombThetaDiff  = torch.zeros(CombTraceStop-CombTraceStart)
            CombPhiDiff    = torch.zeros(CombTraceStop-CombTraceStart)
            CombWeights    = torch.zeros(CombTraceStop-CombTraceStart)
            CombNpix       = torch.zeros(CombTraceStop-CombTraceStart)

            for iPix,IsHEAT in enumerate(HEAT):
                CombPulseStart = PStart[iPix]+TimeOff[iPix]-CombTraceStart
                CombPulseStop  = min([PStop [iPix]+TimeOff[iPix]-CombTraceStart,CombPulseStart+100])

                PulseLength = min([(PStop[iPix]-PStart[iPix]).item(),100])
                Trace = Traces[iPix][:PulseLength] if not IsHEAT else (Traces[iPix][::2]+Traces[iPix][1::2])[:PulseLength]
                
                CombTrace  [CombPulseStart:CombPulseStop]  = Trace
                CombTheta  [CombPulseStart:CombPulseStop] += Thetas[iPix] * torch.abs(Trace)
                CombPhi    [CombPulseStart:CombPulseStop] += Phis[iPix]   * torch.abs(Trace)
                CombWeights[CombPulseStart:CombPulseStop] += torch.abs(Trace)
                CombNpix   [CombPulseStart:CombPulseStop] += 1
            
            # Normalise
            CombTheta = CombTheta/CombWeights
            CombPhi   = CombPhi  /CombWeights

            CombTrace[CombTrace<=0] = - torch.log10(-CombTrace[CombTrace<=0]+1)
            CombTrace[CombTrace>0]  =   torch.log10( CombTrace[CombTrace>0 ]+1)

            # Replace NaNs in Theta and Phi with -1
            CombTheta[CombNpix==0] = -1
            CombPhi  [CombNpix==0] = -1

            # Calculate Diffs
            CombThetaDiff[1:] = CombTheta[1:] - CombTheta[:-1]
            CombPhiDiff  [1:] = CombPhi  [1:] - CombPhi  [:-1]
            Mask = (CombNpix[1:] * CombNpix[:-1]) == 0
            CombThetaDiff[:-1][Mask] = 0
            CombPhiDiff  [:-1][Mask] = 0
            CombThetaDiff[1:] [Mask] = 0
            CombPhiDiff  [1:] [Mask] = 0

            CombTheta = torch.cos(CombTheta*torch.pi/180)
            CombPhi   = torch.sin(CombPhi*torch.pi/180)
            
            # Add To Main
            Main[i,-len(CombTrace):,0] = CombTrace[-1000:]
            Main[i,-len(CombTrace):,1] = CombTheta[-1000:]
            Main[i,-len(CombTrace):,2] = CombPhi[-1000:]
            Main[i,-len(CombTrace):,3] = CombThetaDiff[-1000:]
            Main[i,-len(CombTrace):,4] = CombPhiDiff[-1000:]


        except:
            EventsFailed += 1
            continue    

    Main[torch.isnan(Main)] = -1
    if ProcessingDataset == None: return Main,IDsList # Testing
    else:
        ProcessingDataset._Main.append(Main)
        if ProcessingDataset._EventIds is None: ProcessingDataset._EventIds = IDsList
        else: assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Main_LSTM_Trace_NoDiffs(Dataset,ProcessingDataset):
    '''Will just provide full length trace for each event'''

    Main = torch.zeros(len(Dataset),1000,3) -1
    IDsList = ()
    EventsFailed = 0
    for i,Event in enumerate(Dataset):
        
        print(f'    Processing Main {i} / {len(Dataset)} | Bad Events : {EventsFailed}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        try:
            # Event Information

            Traces = Event.get_trace_values()
            Thetas = Event.get_pixel_values('Theta')
            Phis   = Event.get_pixel_values('Phi')
            PStart = Event.get_pixel_values('PulseStart')
            PStop  = Event.get_pixel_values('PulseStop')
            TimeOff= Event.get_pixel_values('TimeOffset')
            EyeID  = Event.get_pixel_values('EyeID')
            TelID  = Event.get_pixel_values('TelID')
            Status = Event.get_pixel_values('Status')

            Traces = Traces[Status>2]
            Thetas = Thetas[Status>2]
            Phis   = Phis[Status>2]
            PStart = PStart[Status>2]
            PStop  = PStop[Status>2]
            TimeOff= TimeOff[Status>2]
            EyeID  = EyeID[Status>2]
            TelID  = TelID[Status>2]
            Status = Status[Status>2]

            HEAT = ~((EyeID!=5)*(TelID<7))
            PStart[HEAT] *= 0.5
            PStop [HEAT] *= 0.5
            PStart = PStart.int()
            PStop  = PStop.int()
            TimeOff= TimeOff.int()

            # Make Trace

            CombTraceStart = (PStart+TimeOff).min().item()
            CombTraceStop  = (PStop +TimeOff).max().item()
            CombTrace      = torch.zeros(CombTraceStop-CombTraceStart)
            CombTheta      = torch.zeros(CombTraceStop-CombTraceStart)
            CombPhi        = torch.zeros(CombTraceStop-CombTraceStart)
            # CombThetaDiff  = torch.zeros(CombTraceStop-CombTraceStart)
            # CombPhiDiff    = torch.zeros(CombTraceStop-CombTraceStart)
            CombWeights    = torch.zeros(CombTraceStop-CombTraceStart)
            CombNpix       = torch.zeros(CombTraceStop-CombTraceStart)

            for iPix,IsHEAT in enumerate(HEAT):
                CombPulseStart = PStart[iPix]+TimeOff[iPix]-CombTraceStart
                CombPulseStop  = min([PStop [iPix]+TimeOff[iPix]-CombTraceStart,CombPulseStart+100])

                PulseLength = min([(PStop[iPix]-PStart[iPix]).item(),100])
                Trace = Traces[iPix][:PulseLength] if not IsHEAT else (Traces[iPix][::2]+Traces[iPix][1::2])[:PulseLength]
                
                CombTrace  [CombPulseStart:CombPulseStop]  = Trace
                CombTheta  [CombPulseStart:CombPulseStop] += Thetas[iPix] * torch.abs(Trace)
                CombPhi    [CombPulseStart:CombPulseStop] += Phis[iPix]   * torch.abs(Trace)
                CombWeights[CombPulseStart:CombPulseStop] += torch.abs(Trace)
                CombNpix   [CombPulseStart:CombPulseStop] += 1
            
            # Normalise
            CombTheta = CombTheta/CombWeights
            CombPhi   = CombPhi  /CombWeights
        
            # CombTrace[CombTrace<=0] = - torch.log10(-CombTrace[CombTrace<=0]+1)
            # CombTrace[CombTrace>0]  =   torch.log10( CombTrace[CombTrace>0 ]+1)

            CombTrace = torch.log10(torch.clip((CombTrace+1),min=1))

            # Replace NaNs in Theta and Phi with -1
            CombTheta[CombNpix==0] = -1
            CombPhi  [CombNpix==0] = -1

            # # Calculate Diffs
            # CombThetaDiff[1:] = CombTheta[1:] - CombTheta[:-1]
            # CombPhiDiff  [1:] = CombPhi  [1:] - CombPhi  [:-1]
            # Mask = (CombNpix[1:] * CombNpix[:-1]) == 0
            # CombThetaDiff[:-1][Mask] = 0
            # CombPhiDiff  [:-1][Mask] = 0
            # CombThetaDiff[1:] [Mask] = 0
            # CombPhiDiff  [1:] [Mask] = 0

            CombTheta = torch.cos(CombTheta*torch.pi/180)
            CombPhi   = torch.sin(CombPhi*torch.pi/180)
            
            # Add To Main
            Main[i,-len(CombTrace):,0] = CombTrace[-1000:]
            Main[i,-len(CombTrace):,1] = CombTheta[-1000:]
            Main[i,-len(CombTrace):,2] = CombPhi  [-1000:]
            # Main[i,:len(CombTrace),3] = CombThetaDiff[-1000:]
            # Main[i,:len(CombTrace),4] = CombPhiDiff[-1000:]


        except:
            EventsFailed += 1
            continue    

    Main[torch.isnan(Main)] = -1
    if ProcessingDataset == None: return Main,IDsList # Testing
    else:
        ProcessingDataset._Main.append(Main)
        if ProcessingDataset._EventIds is None: ProcessingDataset._EventIds = IDsList
        else: assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Main_ChiTraces(Dataset,ProcessingDataset = None):
    '''Make Traces as bins of Chi_i'''

    Trace       = torch.zeros(len(Dataset),100,1)
    NPixels     = torch.zeros(len(Dataset),100,1)
    TraceStop   = torch.zeros(len(Dataset),1)
    ChiBinEdges = torch.linspace(0,100,101)

    IDsList = ()
    EventsFailed = 0

    for i,Event in enumerate(Dataset):

        print(f'    Processing Main {i} / {len(Dataset)} | Bad Events : {EventsFailed}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        try:
            # Event Information

            PixelChi_i  = Event.get_pixel_values('Chi_i')*180/torch.pi
            PixelCharge = Event.get_pixel_values('Charge')
            PixelTraceIndex = PixelChi_i.int()
            # Make Trace
            Mask = (PixelTraceIndex>=0)*(PixelTraceIndex<100)
            PixelTraceIndex = PixelTraceIndex[Mask]
            PixelCharge     = PixelCharge[Mask]
            
            Trace [i, :] = torch.bincount(PixelTraceIndex, weights=PixelCharge, minlength=100).unsqueeze(1)+1 # +1 to avoid log(0)
            NPixels[i,:] = torch.bincount(PixelTraceIndex,                      minlength=100).unsqueeze(1)+1 # +1 to avoid   /0
            TraceStop[i] = PixelTraceIndex.max().item() if len(PixelTraceIndex)>0 else 0
        except:
            EventsFailed += 1
            continue
    
    # Normalise
    Trace = torch.log10(Trace/NPixels)

    # Append together traces and Npixels
    Main = torch.cat((Trace,NPixels),dim=2)



    if ProcessingDataset == None: return Main,IDsList # Testing
    else:
        ProcessingDataset._Main.append(Main)
        ProcessingDataset._Aux = TraceStop
        if           ProcessingDataset._EventIds is None   : ProcessingDataset._EventIds = IDsList
        else: assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Graph_Comb_All(Dataset,ProcessingDataset = None):
    # NeighboursLevel = 1
    # 1 =     connect all nodes in the same time slot
    # 2 = 1 + connect to  their own         next and previous time slot
    # 3 = 2 + connect to  adjacent pixel's  next and previous time slot

    
    # Setup some required things
    def Get_Ang_Diff(PhiA,ThetaA,PhiB,ThetaB):
        return torch.acos(torch.cos(ThetaA)*torch.cos(ThetaB) + torch.sin(ThetaA)*torch.sin(ThetaB)*torch.cos(PhiA-PhiB))

    PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}


    IDsList = ()
    Graph   = []
    TotalNValues = 0
    TotalSkippedEvents = 0
    for i,Event in enumerate(Dataset):
        # AverageEdgeSize = np.sum(list(map(lambda x: len(x[1]),Graph)))/len(Graph)
        # AverageEdgeSizeL3 = np.sum(list(map(lambda x: len(x[1][x[2][:,0]==3]),Graph)))/len(Graph)
        AverageValues = TotalNValues/(i+1)
        
        # print(f'Processing Graph {i+1}/{len(Dataset)}',end='\r')
        # print(f'Processing Graph {i+1}/{len(Dataset)} with average edge number {AverageEdgeSize}',end='\r')
        # print(f'Processing Graph {i+1}/{len(Dataset)} with average L3 edge number {AverageEdgeSizeL3}',end='\r')
        # print(f'Processing Graph {i+1}/{len(Dataset)} with average values {AverageValues}',end='\r')
        print(f'Processing Graph {i+1}/{len(Dataset)} with {round(100*TotalSkippedEvents/(i+1), 3)}% skipped events and {round(TotalNValues*1e-9,3)} G values', end='\r')
        # print(f'Processing Graph {i+1}/{len(Dataset)}')

        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        
        IDsList += (ID,)

        Phi        = Event.get_pixel_values('Phi'       )
        Theta      = Event.get_pixel_values('Theta'     )
        TOffset    = Event.get_pixel_values('TimeOffset')
        PulseStart = Event.get_pixel_values('PulseStart')
        PulseStop  = Event.get_pixel_values('PulseStop' )
        TelID      = Event.get_pixel_values('TelID'     )
        PixelID    = Event.get_pixel_values('PixelID'   )


        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        PulseStart[HEAT]= 0.5 * PulseStart[HEAT]
        PulseStop[HEAT] = 0.5 * PulseStop[HEAT]

        PulseDuration = PulseStop - PulseStart
        NNodes = PulseDuration.int().sum()//2
        # print(f'Number of nodes: {NNodes.item()}')
        if (not HEAT.all()) & (NNodes <900):
            HottestTel = TelID[~HEAT].mode().values.item()
            
            Signals = Event.get_trace_values()

            # Sum up adjacent time bins
            
            Signals = Signals[:,::2] + Signals[:,1::2]
            
            PulseDuration = PulseDuration/2
            indices = torch.arange(50).expand(Signals.shape[0], -1)
            Mask = (indices < PulseDuration.unsqueeze(1)).flatten()
            
            Signals = Signals.flatten()
            Times   = (torch.arange(50).repeat(len(Phi),1)*2+TOffset.unsqueeze(1)+PulseStart.unsqueeze(1)).flatten()//2 # *2 to keep bins in sync with TOffset and Start //2 to account for reduction of shape
            Xvals   = Phi    .repeat_interleave(50)/180*torch.pi - PhiOffsets[HottestTel] - torch.pi/2
            Yvals   = Theta  .repeat_interleave(50)/180*torch.pi
            PixelID = PixelID.repeat_interleave(50)

            # Normalisation
            Signals = torch.log10(torch.clamp(Signals+1,min=1))
            # Xvals   = Xvals # Consider already normalised
            Yvals   = torch.pi/2 - Yvals
            # Times   = Times # Consider already normalised

            NodeValues = torch.stack([Xvals,Yvals,Times,Signals,PixelID],dim=1)
            NodeValues = NodeValues[Mask]


            # Vectorised approach
            time_diffs = torch.abs(NodeValues[:,2].unsqueeze(1) - NodeValues[:,2].unsqueeze(0))
            pix_diffs  = torch.abs(NodeValues[:,4].unsqueeze(1) - NodeValues[:,4].unsqueeze(0))
            ang_diffs  = torch.abs(Get_Ang_Diff(NodeValues[:,0].unsqueeze(1),NodeValues[:,1].unsqueeze(1),NodeValues[:,0].unsqueeze(0),NodeValues[:,1].unsqueeze(0)))
            
            NB_L1 = (time_diffs == 0) & (pix_diffs > 0)
            NB_L2 = (time_diffs == 2) & (pix_diffs == 0) # 2 is now adjacent time bins cause of the rebinning
            NB_L3 = (time_diffs == 2) & (ang_diffs < 2/180*torch.pi) & (pix_diffs > 0)

            NB_L1_I = torch.nonzero(NB_L1)
            NB_L2_I = torch.nonzero(NB_L2)
            NB_L3_I = torch.nonzero(NB_L3)
            # print(NB_L1_I.shape)
            # print(NB_L2_I.shape)
            # print(NB_L3_I.shape)
            Edges     = torch.cat([NB_L1_I,NB_L2_I,NB_L3_I])
            NBH_level = torch.cat([torch.ones(len(NB_L1_I)),2*torch.ones(len(NB_L2_I)),3*torch.ones(len(NB_L3_I))])
            edge_values = torch.stack([NBH_level,ang_diffs[NB_L1|NB_L2|NB_L3]],dim=1)
            
            # replace any nans with -1
            NodeValues[torch.isnan(NodeValues)] = -1
            edge_values[torch.isnan(edge_values)] = -1
            

            EventGraphData = [NodeValues,Edges,edge_values]
            Graph.append(EventGraphData)
            
            # increase total number of values
            TotalNValues += len(NodeValues)*5 + len(Edges)*2 + len(edge_values)*2
            
        else:
            NodeValues = torch.tensor([[0,0,0,0,0]])  
            Edges = torch.tensor([[0,0]])
            edge_values = torch.tensor([[0,0]])

            EventGraphData = [NodeValues,Edges,edge_values]
            Graph.append(EventGraphData)
            TotalSkippedEvents += 1
    # Slap into Dataset
    if ProcessingDataset is None:
        return Graph
    else:
        ProcessingDataset._Graph = Graph
        ProcessingDataset.GraphData = True
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'

def Unnormalise_Axis_Xmax_Energy(Data):
    # Data[:,0] = Data[:,0]
    # Data[:,1] = Data[:,1]
    # Data[:,2] = Data[:,2]
    Data[:,3] = torch.asin(Data[:,3])
    Data[:,4] = Data[:,4]*30000
    Data[:,5] = Data[:,5]*70+750
    Data[:,6] = Data[:,6]+19
    return Data

def Truth_Axis_Xmax_Energy(Dataset,ProcessingDataset):
    '''Converts geometry to axis data:
    Gets Xmax, and Energy
    
    '''
    IDsList = ()
    Offsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_Chi0     = torch.zeros(len(Dataset))
    Gen_CEDist   = torch.zeros(len(Dataset))
    Gen_Xmax     = torch.zeros(len(Dataset))
    Gen_LogE     = torch.zeros(len(Dataset))

    Rec_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_Chi0     = torch.zeros(len(Dataset))
    Rec_CEDist   = torch.zeros(len(Dataset))
    Rec_Xmax     = torch.zeros(len(Dataset))
    Rec_LogE     = torch.zeros(len(Dataset))



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
        Gen_Chi0    [i]       = Event.get_value('Gen_Chi0')
        Gen_CEDist  [i]       = Event.get_value('Gen_CoreEyeDist')
        Gen_Xmax    [i]       = Event.get_value('Gen_Xmax')
        Gen_LogE    [i]       = Event.get_value('Gen_LogE')
        
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
        Rec_Chi0    [i]       = Event.get_value('Rec_Chi0')
        Rec_CEDist  [i]       = Event.get_value('Rec_CoreEyeDist')
        Rec_Xmax    [i]       = Event.get_value('Rec_Xmax')
        Rec_LogE    [i]       = Event.get_value('Rec_LogE')


    print()

    # Normalise Phi
    Gen_SDPPhi   = Gen_SDPPhi+2*torch.pi*(Gen_SDPPhi<0)
    Gen_SDPPhi   -= torch.pi
    Rec_SDPPhi   = Rec_SDPPhi+2*torch.pi*(Rec_SDPPhi<0)
    Rec_SDPPhi   -= torch.pi
    for i in range(1,7): # Apply offsets
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
    
    Gen_SDPPhi   = torch.sin(Gen_SDPPhi)
    Rec_SDPPhi   = torch.sin(Rec_SDPPhi)

    # Normalise x,y,z
    Gen_SDPTheta -= torch.pi/2
    Gen_Chi0     -= torch.pi/2
    Rec_SDPTheta -= torch.pi/2
    Rec_Chi0     -= torch.pi/2

    Gen_x = torch.sin(Gen_Chi0)*torch.cos(Gen_SDPTheta)
    Gen_y = torch.sin(Gen_SDPTheta)
    Gen_z = torch.cos(Gen_Chi0)*torch.cos(Gen_SDPTheta)

    Rec_x = torch.sin(Rec_Chi0)*torch.cos(Rec_SDPTheta)
    Rec_y = torch.sin(Rec_SDPTheta)
    Rec_z = torch.cos(Rec_Chi0)*torch.cos(Rec_SDPTheta)

    # Normalise CE Distance
    Gen_CEDist /= 30000
    Rec_CEDist /= 30000

    # Normalise Xmax
    Gen_Xmax -= 750
    Rec_Xmax -= 750

    Gen_Xmax /=70
    Rec_Xmax /=70

    # Normalise LogE

    Gen_LogE -= 19
    Rec_LogE -= 19

    

    if ProcessingDataset != None:

        ProcessingDataset._Truth = torch.stack((Gen_x,Gen_y,Gen_z,Gen_SDPPhi,Gen_CEDist,Gen_Xmax,Gen_LogE),dim=1)
        ProcessingDataset._Rec   = torch.stack((Rec_x,Rec_y,Rec_z,Rec_SDPPhi,Rec_CEDist,Rec_Xmax,Rec_LogE),dim=1)

        ProcessingDataset.Unnormalise_Truth = Unnormalise_Axis_Xmax_Energy
        ProcessingDataset.Truth_Keys = ('x','y','z','SDPPhi','CEDist','Xmax','LogE')
        ProcessingDataset.Truth_Units =('','','','rad','m','g/cm^2','')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'
    
    else:
        return torch.stack((Gen_x,Gen_y,Gen_z,Gen_SDPPhi,Gen_CEDist,Gen_Xmax,Gen_LogE),dim=1),torch.stack((Rec_x,Rec_y,Rec_z,Rec_SDPPhi,Rec_CEDist,Rec_Xmax,Rec_LogE),dim=1),IDsList

def Unnormalise_Xmax_Energy(Data):
    Data[:,0] = Data[:,0]*70+750
    Data[:,1] = Data[:,1]+19
    return Data

def Truth_Xmax_Energy(Dataset,ProcessingDataset):
    '''Gets Xmax, and Energy
    '''
    IDsList = ()
    # Offsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_Xmax     = torch.zeros(len(Dataset))
    Gen_LogE     = torch.zeros(len(Dataset))

    Rec_Xmax     = torch.zeros(len(Dataset))
    Rec_LogE     = torch.zeros(len(Dataset))



    
    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        
        Gen_Xmax    [i]       = Event.get_value('Gen_Xmax')
        Gen_LogE    [i]       = Event.get_value('Gen_LogE')
        
        Rec_Xmax    [i]       = Event.get_value('Rec_Xmax')
        Rec_LogE    [i]       = Event.get_value('Rec_LogE')


    print()

    # Normalise Xmax
    Gen_Xmax -= 750
    Rec_Xmax -= 750

    Gen_Xmax /=70
    Rec_Xmax /=70

    # Normalise LogE

    Gen_LogE -= 19
    Rec_LogE -= 19

    

    if ProcessingDataset != None:

        ProcessingDataset._Truth = torch.stack((Gen_Xmax,Gen_LogE),dim=1)
        ProcessingDataset._Rec   = torch.stack((Rec_Xmax,Rec_LogE),dim=1)

        ProcessingDataset.Unnormalise_Truth = Unnormalise_Xmax_Energy
        ProcessingDataset.Truth_Keys = ('Xmax','LogE')
        ProcessingDataset.Truth_Units =('g/cm^2','')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'
    
    else:
        return torch.stack((Gen_Xmax,Gen_LogE),dim=1),torch.stack((Rec_Xmax,Rec_LogE),dim=1),IDsList
 



def Aux_Station(Dataset,ProcessingDataset):
    '''Basically just all station information'''

    IDsList = ()
    StationData = torch.zeros(len(Dataset),3)

    for i,Event in enumerate(Dataset):
        print(f'    Processing Aux {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        StationData[i,0] = Event.get_value('Station_Time')
        StationData[i,1] = Event.get_value('Station_TotalSignal')
        StationData[i,2] = Event.get_value('Station_Distance')
    
    StationData[:,0] -= 300
    StationData[:,0] /= 700

    StationData[:,1] = torch.log10(StationData[:,1]+1)
    StationData[:,1] /= 4

    StationData[:,2] /= 30000


    if ProcessingDataset is None:
        return StationData
    ProcessingDataset._Aux = StationData
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'

def Meta_No_Meta(Dataset,ProcessingDataset):
    '''Just a dummy function to make sure the metadata is not added'''
    pass

def Meta_Primary_Eye_Tel(Dataset,ProcessingDataset):
    '''Just the primary eye and telescope'''
    IDsList = ()
    MetaData = torch.zeros(len(Dataset),3)

    for i,Event in enumerate(Dataset):
        print(f'    Processing Meta {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        EyeID = Event.get_pixel_values('EyeID')
        TelID = Event.get_pixel_values('TelID')
        if torch.sum((TelID<7)*(EyeID!=5)) ==0:
            SelectedTelID = torch.mode(TelID).values.item()
            SelectedEyeID = 5

        else:
            SelectedTelID = TelID[(TelID<7)*(EyeID!=5)].int().bincount().argmax()
            SelectedEyeID = EyeID[(TelID<7)*(EyeID!=5)].int().bincount().argmax()



        MetaData[i,0] = Event.get_value('Primary')
        MetaData[i,1] = SelectedEyeID
        MetaData[i,2] = SelectedTelID

    
    
    if ProcessingDataset is None:
        return MetaData
    ProcessingDataset._MetaData = MetaData
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'


def Graph_Conv3d_Traces_Distances(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel traces and also each pixel will have the distance to shower axis and height at shower axis'''
    EyeHeights = {1:1416,2:1416,3:1476,4:1712,5:1707,6:1710}
    
    IDsList = ()
    Graph = [] # Will have Event in dim1, [Traces,Distances,X,Y,PulseStart] in dim2, values of dim2 in dim3

    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)
        
        Traces = Event.get_trace_values()
        Pstart = Event.get_pixel_values('PulseStart')
        # Pstop  = Event.get_pixel_values('PulseStop')

        EyeID = Event.get_pixel_values('EyeID'  )
        TelID = Event.get_pixel_values('TelID'  )
        PixID = Event.get_pixel_values('PixelID')
        PixStatus = Event.get_pixel_values('Status')
        PixelChi_i = Event.get_pixel_values('Chi_i')

        Event_Chi0     = Event.get_value('Rec_Chi0')
        Event_RP       = Event.get_value('Rec_Rp')
        Event_SDPTheta = Event.get_value('Rec_SDPTheta')



        if torch.sum(EyeID != 5) == 0 or torch.sum(TelID<7) == 0:
            Mask = torch.zeros(len(EyeID)).bool()
            EventEye = 5
        else:
            EventEye = torch.mode(EyeID[EyeID != 5]).values.item()
            Mask     =(EyeID != 5) * (TelID==torch.mode(TelID[TelID<7]).values.item()) * (PixStatus == 4)

        Distances = Event_RP / torch.sin(Event_Chi0 - PixelChi_i)
        Heights   = - Distances * torch.sin(PixelChi_i) / torch.sin(PixelChi_i - Event_Chi0) * torch.sin(Event_SDPTheta) + EyeHeights[EventEye]

        Traces = Traces[Mask]
        Pstart = Pstart[Mask]
        PixID  = PixID[Mask]
        Distances = Distances[Mask]
        Heights = Heights[Mask]

        Xs,Ys = IndexToXY(PixID,return_tensor=True)

        # Traces normalised Here
        Traces = torch.log1p((Traces).clip(min=0))
        # Pstart normalised Here
        if len(Pstart)>0 : Pstart = Pstart - torch.min(Pstart)
        # Distances normalised Here
        Distances = Distances.clamp(min=0,max = 60000)
        Distances = torch.log10(Distances**2+1) -5.2393
        # Heights normalised Here
        Heights = Heights.clamp(min=1,max=1e5)
        Heights = torch.log10(Heights)/5

        # Append to the GraphData
        Graph.append([Traces,Distances,Heights,Xs,Ys,Pstart])
        
        
    
    if ProcessingDataset is None:
        return Graph
    ProcessingDataset._Graph = Graph
    ProcessingDataset.GraphData = True
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'    




def Clean_Axis(ProcDS):
    ''' Removes some events wherer the Rec was way off, meaning the event is probably screwed up in some way'''

    Truth = ProcDS._Truth
    # KeepMask = torch.ones(len(Truth)).bool()
    KeepMask = Truth[:,6] > -5

    print()
    print(f'Keeping {torch.sum(KeepMask)} / {len(Truth)} events')
    print()
    ProcDS._Aux      = ProcDS._Aux  [KeepMask]
    ProcDS._Truth    = ProcDS._Truth[KeepMask]
    ProcDS._Rec      = ProcDS._Rec  [KeepMask]
    ProcDS._EventIds = tuple([ProcDS._EventIds[i] for i in range(len(ProcDS._EventIds)) if KeepMask[i]])
    if ProcDS._MetaData !=None: ProcDS._MetaData = ProcDS._MetaData[KeepMask]
    if not ProcDS.GraphData:
        for i in range(len(ProcDS._Main)):
            ProcDS._Main[i] = ProcDS._Main[i][KeepMask]
    else:
        ProcDS._Graph = [ProcDS._Graph[i] for i in range(len(ProcDS._Graph)) if KeepMask[i]]
    

def Clean_Xmax_Energy(ProcDS):
    ''' Removes some events wherer the Rec was way off, meaning the event is probably screwed up in some way'''

    Truth = ProcDS._Truth
    # KeepMask = torch.ones(len(Truth)).bool()
    KeepMask = Truth[:,1] > -5

    print()
    print(f'Keeping {torch.sum(KeepMask)} / {len(Truth)} events')
    print()
    ProcDS._Aux      = ProcDS._Aux  [KeepMask]
    ProcDS._Truth    = ProcDS._Truth[KeepMask]
    ProcDS._Rec      = ProcDS._Rec  [KeepMask]
    ProcDS._EventIds = tuple([ProcDS._EventIds[i] for i in range(len(ProcDS._EventIds)) if KeepMask[i]])
    if ProcDS._MetaData !=None: ProcDS._MetaData = ProcDS._MetaData[KeepMask]
    if not ProcDS.GraphData:
        for i in range(len(ProcDS._Main)):
            ProcDS._Main[i] = ProcDS._Main[i][KeepMask]
    else:
        ProcDS._Graph = [ProcDS._Graph[i] for i in range(len(ProcDS._Graph)) if KeepMask[i]]



# Function Selection
Pass_Main     = Graph_Conv3d_Traces_Distances
Pass_Aux      = Aux_Station
Pass_Truth    = Truth_Xmax_Energy
Pass_Rec      = Rec_Handled_by_Truth
Pass_Graph    = Main_No_Main
Pass_MetaData = Meta_Primary_Eye_Tel


Clean_Data = Clean_Xmax_Energy