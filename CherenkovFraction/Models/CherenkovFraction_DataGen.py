import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly



# Auxilary functions
def IndexToXY(indices,return_tensor=False):
    indices -=1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()



def Standard_Graph_Conv3d_Traces(Dataset,ProcessingDataset):
    assert False, 'This function is not finished yet'
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

