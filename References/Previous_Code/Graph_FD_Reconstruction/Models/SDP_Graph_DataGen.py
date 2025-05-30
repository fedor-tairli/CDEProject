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


def cut_pixels(Pixel_Theta,Pixel_Phi,SDP_Theta,SDP_Phi,zeta=None,pixel_count=None): # Used in Graph Edges
    assert zeta is None, 'Zeta broken because the angle is supposed to be 90 and i havent implemented the thing to work with 90'
    if zeta is not None: assert pixel_count is None, 'must provide either zeta or pixel_count'
    if zeta is None and pixel_count is None: pixel_count = 40
    ''' returns the pixels cut by too far away from the SDP
    pretty much is a hack, cause i am not supposed to know the SDP at this point, but its whatever'''

    if zeta is None and len(Pixel_Theta)<pixel_count: return torch.ones(len(Pixel_Theta)).bool()
    
    SDP_Vector = torch.tensor([torch.sin(SDP_Theta)*torch.cos(SDP_Phi),
                                torch.sin(SDP_Theta)*torch.sin(SDP_Phi),
                                torch.cos(SDP_Theta)])
    Pixel_Vector = torch.stack([torch.sin(Pixel_Theta)*torch.cos(Pixel_Phi),
                                    torch.sin(Pixel_Theta)*torch.sin(Pixel_Phi),
                                    torch.cos(Pixel_Theta)],dim=1)
    Dot_product = torch.sum(SDP_Vector*Pixel_Vector,dim=1)
    Angle = torch.acos(Dot_product)
    if zeta is not None: return Angle<zeta
    else:
        # Find the indices of the 40 smallest angles
        _, indices = torch.topk(torch.abs(Dot_product), pixel_count, largest=False)
        mask = torch.zeros(len(Pixel_Theta), dtype=bool)
        mask[indices] = True
        
        return mask


########################################################################################################################





def Unnormalise_Graph_SDP(Truth,AuxData=None):
    Truth[:,0] = torch.acos(Truth[:,0])/torch.pi*180
    Truth[:,1] = (torch.asin(Truth[:,1])+torch.pi/2)/torch.pi*180
    Truth[:,1][Truth[:,1]>180] -= 360
    if AuxData is None:
        return Truth
    else:
        Truth[:,1] += AuxData
        return Truth

def Graph_Edges(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi, Centroid, Charge, PulseWidth
    EdgeValues are  : Angular Difference, Time Difference
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    # print(f'max ang diff = {max_ang_diff}')
    IDsList = ()
    Graph   = []
    AuxData = []

    Truth = []
    Rec   = []
    
    Meta  = []

    for i, Event in enumerate(Dataset):
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        Pix_Phi        = Event.get_pixel_values('Phi')
        Pix_Theta      = Event.get_pixel_values('Theta')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        # Meta Values 
        Primary  = Event.get_value('Primary')
        LogE     = Event.get_value('Gen_LogE')
        Xmax     = Event.get_value('Gen_Xmax')
        # Going to cheat here and throw away the noise pixels before i do the graph
        # cut = (Pixel_Status == 4)
        # Pix_Phi        = Pix_Phi       [cut]
        # Pix_Theta      = Pix_Theta     [cut]
        # Pix_Centroid   = Pix_Centroid  [cut]
        # Pix_PulseStart = Pix_PulseStart[cut]
        # Pix_PulseStop  = Pix_PulseStop [cut]
        # Pix_Charge     = Pix_Charge    [cut]
        # TelID          = TelID         [cut]
        # PixelIDs       = PixelIDs      [cut]
        # HEAT           = HEAT          [cut]

        SDP_Theta = Event.get_value('Gen_SDPTheta')
        SDP_Phi   = Event.get_value('Gen_SDPPhi')
        
        Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
        Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')

        
        # There is no HEAT in new Dataset, but keeping for backwards compatibility

        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,SDP_Theta,SDP_Phi,pixel_count=40)

        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        Pixel_Status   = Pixel_Status  [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            
            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Time_Gap  = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff  = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff) # 
            Edges = torch.stack(torch.where(Edges),dim=1)
            Edges = Edges[Edges[:,0]!=Edges[:,1]]
            # print(Edges.shape)
            
            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50

            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Centroid,Pix_Charge,Pix_PulseWidth],dim=1)

            # Edge_ang_div  = torch.sqrt((Pix_Phi[Edges[0,:]]-Pix_Phi[Edges[1,:]])**2 + (Pix_Theta[Edges[0,:]]-Pix_Theta[Edges[1,:]])**2)
            # Edge_time_div = torch.abs(Pix_Centroid[Edges[0]]-Pix_Centroid[Edges[1]])
            Edge_ang_div  = Pix_Ang_Diff[Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]

            # Normalise Edge Values
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div],dim=1)
            
            # Normalise SDP Values
            SDP_Theta     = torch.cos(SDP_Theta)
            Rec_SDP_Theta = torch.cos(Rec_SDP_Theta)

            # SDP Phi will be normalised later
            
            # Meta is not normalised
            
            TruthValues = torch.stack([SDP_Theta,SDP_Phi])
            RecValues   = torch.stack([Rec_SDP_Theta,Rec_SDP_Phi])
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges,Edge_values])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0]))
            Rec.append(torch.tensor([0,0]))
            Meta.append(torch.tensor([0,0,0]))

    


    # Slap into Dataset
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)
    
    # Normalise SDP Phi
    Truth[:,1][Truth[:,1]<0] += 2*torch.pi
    Rec  [:,1][Rec  [:,1]<0] += 2*torch.pi

    Truth[:,1] -= torch.pi/2
    Rec  [:,1] -= torch.pi/2
    
    Truth[:,1] -= AuxData.squeeze()/180*torch.pi
    Rec  [:,1] -= AuxData.squeeze()/180*torch.pi

    Truth[:,1]  = torch.sin(Truth[:,1])
    Rec  [:,1]  = torch.sin(Rec  [:,1])
    


    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_SDP
        ProcessingDataset.Truth_Keys        = ['SDPTheta','SDPPhi']
        ProcessingDataset.Truth_Units       = ['deg','deg']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'







def Graph_AllConnected(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi, Centroid, Charge, PulseWidth
    EdgeValues are  : Angular Difference, Time Difference
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph

    All Nodes of the graph will be connected to every node of the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    # print(f'max ang diff = {max_ang_diff}')
    IDsList = ()
    Graph   = []
    AuxData = []

    Truth = []
    Rec   = []
    
    Meta  = []

    for i, Event in enumerate(Dataset):
        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        Pix_Phi        = Event.get_pixel_values('Phi')
        Pix_Theta      = Event.get_pixel_values('Theta')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        # Meta Values 
        Primary  = Event.get_value('Primary')
        LogE     = Event.get_value('Gen_LogE')
        Xmax     = Event.get_value('Gen_Xmax')
        # Going to cheat here and throw away the noise pixels before i do the graph
        # cut = (Pixel_Status == 4)
        # Pix_Phi        = Pix_Phi       [cut]
        # Pix_Theta      = Pix_Theta     [cut]
        # Pix_Centroid   = Pix_Centroid  [cut]
        # Pix_PulseStart = Pix_PulseStart[cut]
        # Pix_PulseStop  = Pix_PulseStop [cut]
        # Pix_Charge     = Pix_Charge    [cut]
        # TelID          = TelID         [cut]
        # PixelIDs       = PixelIDs      [cut]
        # HEAT           = HEAT          [cut]

        SDP_Theta = Event.get_value('Gen_SDPTheta')
        SDP_Phi   = Event.get_value('Gen_SDPPhi')
        
        Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
        Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')

        
        # There is no HEAT in new Dataset, but keeping for backwards compatibility

        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,SDP_Theta,SDP_Phi,pixel_count=40)

        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        Pixel_Status   = Pixel_Status  [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            
            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Time_Gap  = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff  = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            # Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff) # 
            # Edges = torch.stack(torch.where(Edges),dim=1)
            # Edges = Edges[Edges[:,0]!=Edges[:,1]]
            # print(Edges.shape)
            Edges = torch.tensor([[i,j] for i in range(len(Pix_Theta)) for j in range(len(Pix_Theta))])
            
            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50

            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Centroid,Pix_Charge,Pix_PulseWidth],dim=1)

            # Edge_ang_div  = torch.sqrt((Pix_Phi[Edges[0,:]]-Pix_Phi[Edges[1,:]])**2 + (Pix_Theta[Edges[0,:]]-Pix_Theta[Edges[1,:]])**2)
            # Edge_time_div = torch.abs(Pix_Centroid[Edges[0]]-Pix_Centroid[Edges[1]])
            Edge_ang_div  = Pix_Ang_Diff[Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]

            # Normalise Edge Values
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div],dim=1)
            
            # Normalise SDP Values
            SDP_Theta     = torch.cos(SDP_Theta)
            Rec_SDP_Theta = torch.cos(Rec_SDP_Theta)

            # SDP Phi will be normalised later
            
            # Meta is not normalised
            
            TruthValues = torch.stack([SDP_Theta,SDP_Phi])
            RecValues   = torch.stack([Rec_SDP_Theta,Rec_SDP_Phi])
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges,Edge_values])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0]))
            Rec.append(torch.tensor([0,0]))
            Meta.append(torch.tensor([0,0,0]))

    


    # Slap into Dataset
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)
    
    # Normalise SDP Phi
    Truth[:,1][Truth[:,1]<0] += 2*torch.pi
    Rec  [:,1][Rec  [:,1]<0] += 2*torch.pi

    Truth[:,1] -= torch.pi/2
    Rec  [:,1] -= torch.pi/2
    
    Truth[:,1] -= AuxData.squeeze()/180*torch.pi
    Rec  [:,1] -= AuxData.squeeze()/180*torch.pi

    Truth[:,1]  = torch.sin(Truth[:,1])
    Rec  [:,1]  = torch.sin(Rec  [:,1])
    


    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_SDP
        ProcessingDataset.Truth_Keys        = ['SDPTheta','SDPPhi']
        ProcessingDataset.Truth_Units       = ['deg','deg']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'







