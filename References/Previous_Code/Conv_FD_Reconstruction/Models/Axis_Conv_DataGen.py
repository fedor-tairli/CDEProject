
import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly



# Auxilary functions
def IndexToXY(indices,return_tensor=False):
    indices -=1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()












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

def Unnormalise_Axis(Data):
    # Data[:,0] = Data[:,0]
    # Data[:,1] = Data[:,1]
    # Data[:,2] = Data[:,2]
    Data[:,3] = torch.asin(Data[:,3])
    Data[:,4] = Data[:,4]*30000
    return Data


def Truth_Axis(Dataset,ProcessingDataset):
    '''Converts geometry to axis data:
    '''
    IDsList = ()
    Offsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    Gen_SDPPhi   = torch.zeros(len(Dataset))
    Gen_SDPTheta = torch.zeros(len(Dataset))
    Gen_Chi0     = torch.zeros(len(Dataset))
    Gen_CEDist   = torch.zeros(len(Dataset))
    
    Rec_SDPPhi   = torch.zeros(len(Dataset))
    Rec_SDPTheta = torch.zeros(len(Dataset))
    Rec_Chi0     = torch.zeros(len(Dataset))
    Rec_CEDist   = torch.zeros(len(Dataset))
    

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
        
        Rec_SDPPhi  [i]       = Event.get_value('Rec_SDPPhi')
        Rec_SDPTheta[i]       = Event.get_value('Rec_SDPTheta')
        Rec_Chi0    [i]       = Event.get_value('Rec_Chi0')
        Rec_CEDist  [i]       = Event.get_value('Rec_CoreEyeDist')
        

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

    

    if ProcessingDataset != None:

        ProcessingDataset._Truth = torch.stack((Gen_x,Gen_y,Gen_z,Gen_SDPPhi,Gen_CEDist),dim=1)
        ProcessingDataset._Rec   = torch.stack((Rec_x,Rec_y,Rec_z,Rec_SDPPhi,Rec_CEDist),dim=1)

        ProcessingDataset.Unnormalise_Truth = Unnormalise_Axis
        ProcessingDataset.Truth_Keys = ('x','y','z','SDPPhi','CEDist')
        ProcessingDataset.Truth_Units =('','','','rad','m')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'
    
    else:
        return torch.stack((Gen_x,Gen_y,Gen_z,Gen_SDPPhi,Gen_CEDist),dim=1),torch.stack((Rec_x,Rec_y,Rec_z,Rec_SDPPhi,Rec_CEDist),dim=1),IDsList

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




