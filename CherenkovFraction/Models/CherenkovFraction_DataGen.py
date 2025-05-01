import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly



# Auxilary functions
def IndexToXY(indices,return_tensor=False):
    indices = indices -1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()



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


def Unnormalise_CherenkovFraction(Data):
    return Data*100


def Truth_CherenkovFraction(Dataset,ProcessingDataset):
    ''' Just Cherenkov Fraction for now, nothing more
    A bit Jenk, cause only 1 truth value
    Making sure the tensors for truth are Nx1 2D, not 1D'''

    IDsList = ()
    Gen_CherenkovFraction = torch.zeros(len(Dataset),1)
    Rec_CherenkovFraction = torch.zeros(len(Dataset),1)
    
    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the CherenkovFraction
        Gen_CherenkovFraction[i] = Event.get_value('Gen_CherenkovFraction')
        Rec_CherenkovFraction[i] = Event.get_value('Rec_CherenkovFraction')

    # Normalise
    Gen_CherenkovFraction = Gen_CherenkovFraction/100
    Rec_CherenkovFraction = Rec_CherenkovFraction/100

    # Slap Into Dataset
    if ProcessingDataset is None:
        return Gen_CherenkovFraction,Rec_CherenkovFraction
    else:
        ProcessingDataset._Truth = Gen_CherenkovFraction
        ProcessingDataset._Rec   = Rec_CherenkovFraction
        ProcessingDataset._Unnormalise_Truth = Unnormalise_CherenkovFraction
        ProcessingDataset.Truth_Keys = ('CherenkovFraction')
        ProcessingDataset.Truth_Units = ('')

        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'

   
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


