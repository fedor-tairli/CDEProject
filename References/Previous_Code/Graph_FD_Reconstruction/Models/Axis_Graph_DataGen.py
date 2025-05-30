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




def Unnormalise_Graph_Axis(Truth,AuxData=None):
    lightspeed = 299792458/1e7 # m/100ns
    # Truth[:,0] = Truth[:,0]
    # Truth[:,1] = Truth[:,1]
    # Truth[:,2] = Truth[:,2]
    Truth[:,4] = Truth[:,4]*lightspeed
    Truth[:,3] = (torch.asin(Truth[:,3])+torch.pi/2)/torch.pi*180
    Truth[:,3][Truth[:,3]>180] -= 360
    if AuxData is None:
        return Truth
    else:
        Truth[:,1] += AuxData
        return Truth


def Graph_Edges_Axis(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi,Chi_i, Centroid, Charge, PulseWidth
    EdgeValues are  : Not Given as this will be taken care by the graph network
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    lightspeed = 299792458/1e7 # m/100ns
    
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
        Pix_Chi_i      = Event.get_pixel_values('Chi_i')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
        Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            
        
        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,Gen_SDP_Theta,Gen_SDP_Phi,pixel_count=40)
        cut = cut & (Pixel_Status == 4) # For Testing Mostly
        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Chi_i      = Pix_Chi_i     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Time_Gap  = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff  = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff)
            Edges = torch.stack(torch.where(Edges),dim=1)
            Edges = Edges[Edges[:,0]!=Edges[:,1]]
            
            Edge_ang_div  = Pix_Ang_Diff[Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]

            # Normalise Edge Values
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div],dim=1)
            
            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50
            
            Pix_Chi_i    -= torch.min(Pix_Chi_i)
            Pix_Centroid -= torch.min(Pix_Centroid)

            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Chi_i,Pix_Centroid,Pix_Charge,Pix_PulseWidth],dim=1)



            # Meta Values 
            Primary  = Event.get_value('Primary')
            LogE     = Event.get_value('Gen_LogE')
            Xmax     = Event.get_value('Gen_Xmax')
            # Truth Values
            # Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
            # Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            Gen_Chi0      = Event.get_value('Gen_Chi0')
            Gen_CEDist    = Event.get_value('Gen_CoreEyeDist')
            # Rec Values
            Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
            Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')
            Rec_Chi0      = Event.get_value('Rec_Chi0')
            Rec_CEDist    = Event.get_value('Rec_CoreEyeDist')
            
            Gen_SDP_Theta -= torch.pi/2
            Gen_Chi0      -= torch.pi/2
            Rec_SDP_Theta -= torch.pi/2
            Rec_Chi0      -= torch.pi/2
            
            Gen_x = torch.sin(Gen_Chi0)*torch.cos(Gen_SDP_Theta)
            Gen_y = torch.sin(Gen_SDP_Theta)
            Gen_z = torch.cos(Gen_Chi0)*torch.cos(Gen_SDP_Theta)

            Rec_x = torch.sin(Rec_Chi0)*torch.cos(Rec_SDP_Theta)
            Rec_y = torch.sin(Rec_SDP_Theta)
            Rec_z = torch.cos(Rec_Chi0)*torch.cos(Rec_SDP_Theta)

            
            TruthValues = torch.stack([Gen_x,Gen_y,Gen_z,Gen_SDP_Phi,Gen_CEDist],dim=0)
            RecValues   = torch.stack([Rec_x,Rec_y,Rec_z,Rec_SDP_Phi,Rec_CEDist],dim=0)
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            # Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0,0,0,0]))
            Rec.append(torch.tensor([0,0,0,0,0]))
            Meta.append(torch.tensor([0,0,0]))


    # Stack Data
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)

    
    # Normalise Truth and Rec out of Loop
    Truth[:,3][Truth[:,3]<0] += 2*torch.pi
    Truth[:,3] -= torch.pi/2
    Truth[:,3] -= AuxData[:,0]/180*torch.pi
    Truth[:,3] = torch.sin(Truth[:,3])
    Truth[:,4] /= lightspeed*1000

    Rec[:,3][Rec[:,3]<0] += 2*torch.pi
    Rec[:,3] -= torch.pi/2
    Rec[:,3] -= AuxData[:,0]/180*torch.pi
    Rec[:,3] = torch.sin(Rec[:,3])
    Rec[:,4] /= lightspeed*1000





    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_Axis
        ProcessingDataset.Truth_Keys        = ['x','y','z','SDPPhi','CEDist']
        ProcessingDataset.Truth_Units       = ['','','','deg','km']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'







def Graph_Edges_wStation_Axis(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi,Chi_i, Centroid, Charge, PulseWidth
    EdgeValues are  : Not Given as this will be taken care by the graph network
    Unique Node is added to represent the Station, it is connected to all nodes already present
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    lightspeed = 299792458/1e7 # m/100ns
    
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
        Pix_Chi_i      = Event.get_pixel_values('Chi_i')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
        Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            
        
        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,Gen_SDP_Theta,Gen_SDP_Phi,pixel_count=40)
        cut = cut & (Pixel_Status == 4) # For Testing Mostly
        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Chi_i      = Pix_Chi_i     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            # Get Station Values
            Station_Theta      = Event.get_value('Station_Theta')*180/torch.pi
            Station_Phi        = Event.get_value('Station_Phi')*180/torch.pi
            Station_Chi_i      = Event.get_value('Station_Chi_i')
            Station_Centroid   = Event.get_value('Station_Time')
            Station_Charge     = -1
            Station_PulseWidth = -1
            
            
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Time_Gap  = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff  = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff)
            Edges = torch.stack(torch.where(Edges),dim=1)
            Edges = Edges[Edges[:,0]!=Edges[:,1]]
            # Station will be the last node
            # It will connect to all other nodes
            Station_Edges = torch.stack([torch.arange(len(Pix_Theta)),torch.ones(len(Pix_Theta))*len(Pix_Theta)],dim=1).to(torch.long)

            Edge_ang_div  = Pix_Ang_Diff[Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div],dim=1)

            Station_Edge_ang_div  = torch.sqrt((Station_Phi-Pix_Phi)**2 + (Station_Theta-Pix_Theta)**2) / 1.5
            Station_Edge_time_div = torch.abs(Station_Centroid-Pix_Centroid)                            / 100
            
            Station_Edge_values = torch.stack([Station_Edge_ang_div,Station_Edge_time_div],dim=1)
            
            Edges = torch.cat([Edges,Station_Edges],dim=0)
            Edge_values = torch.cat([Edge_values,Station_Edge_values],dim=0)

            # Append Station Node
            Pix_Theta    = torch.cat([Pix_Theta   ,torch.tensor([Station_Theta])])
            Pix_Phi      = torch.cat([Pix_Phi     ,torch.tensor([Station_Phi])])
            Pix_Chi_i    = torch.cat([Pix_Chi_i   ,torch.tensor([Station_Chi_i])])
            Pix_Centroid = torch.cat([Pix_Centroid,torch.tensor([Station_Centroid])])

            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50
            
            Pix_Chi_i    -= torch.min(Pix_Chi_i)
            Pix_Centroid -= torch.min(Pix_Centroid)

            # append the Non-Normalised Station Values
            Pix_Charge   = torch.cat([Pix_Charge,torch.tensor([Station_Charge])])
            Pix_PulseWidth = torch.cat([Pix_PulseWidth,torch.tensor([Station_PulseWidth])])

            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Chi_i,Pix_Centroid,Pix_Charge,Pix_PulseWidth],dim=1)



            # Meta Values 
            Primary  = Event.get_value('Primary')
            LogE     = Event.get_value('Gen_LogE')
            Xmax     = Event.get_value('Gen_Xmax')
            # Truth Values
            # Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
            # Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            Gen_Chi0      = Event.get_value('Gen_Chi0')
            Gen_CEDist    = Event.get_value('Gen_CoreEyeDist')
            # Rec Values
            Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
            Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')
            Rec_Chi0      = Event.get_value('Rec_Chi0')
            Rec_CEDist    = Event.get_value('Rec_CoreEyeDist')
            
            Gen_SDP_Theta -= torch.pi/2
            Gen_Chi0      -= torch.pi/2
            Rec_SDP_Theta -= torch.pi/2
            Rec_Chi0      -= torch.pi/2
            
            Gen_x = torch.sin(Gen_Chi0)*torch.cos(Gen_SDP_Theta)
            Gen_y = torch.sin(Gen_SDP_Theta)
            Gen_z = torch.cos(Gen_Chi0)*torch.cos(Gen_SDP_Theta)

            Rec_x = torch.sin(Rec_Chi0)*torch.cos(Rec_SDP_Theta)
            Rec_y = torch.sin(Rec_SDP_Theta)
            Rec_z = torch.cos(Rec_Chi0)*torch.cos(Rec_SDP_Theta)

            
            TruthValues = torch.stack([Gen_x,Gen_y,Gen_z,Gen_SDP_Phi,Gen_CEDist],dim=0)
            RecValues   = torch.stack([Rec_x,Rec_y,Rec_z,Rec_SDP_Phi,Rec_CEDist],dim=0)
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            # Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0,0,0,0]))
            Rec.append(torch.tensor([0,0,0,0,0]))
            Meta.append(torch.tensor([0,0,0]))


    # Stack Data
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)

    
    # Normalise Truth and Rec out of Loop
    Truth[:,3][Truth[:,3]<0] += 2*torch.pi
    Truth[:,3] -= torch.pi/2
    Truth[:,3] -= AuxData[:,0]/180*torch.pi
    Truth[:,3] = torch.sin(Truth[:,3])
    Truth[:,4] /= lightspeed*1000

    Rec[:,3][Rec[:,3]<0] += 2*torch.pi
    Rec[:,3] -= torch.pi/2
    Rec[:,3] -= AuxData[:,0]/180*torch.pi
    Rec[:,3] = torch.sin(Rec[:,3])
    Rec[:,4] /= lightspeed*1000





    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_Axis
        ProcessingDataset.Truth_Keys        = ['x','y','z','SDPPhi','CEDist']
        ProcessingDataset.Truth_Units       = ['','','','deg','km']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'



def Graph_Edges_wStation_wAngVel_Axis(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi,Chi_i, Centroid, Charge, PulseWidth
    EdgeValues are  : Ang_Diff, Time_Diff also Angular Velcity
    Unique Node is added to represent the Station, it is connected to all nodes already present
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    lightspeed = 299792458/1e7 # m/100ns
    
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
        Pix_Chi_i      = Event.get_pixel_values('Chi_i')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
        Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            
        
        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,Gen_SDP_Theta,Gen_SDP_Phi,pixel_count=40)
        cut = cut & (Pixel_Status == 4) # For Testing Mostly
        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Chi_i      = Pix_Chi_i     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            # Get Station Values
            Station_Theta      = Event.get_value('Station_Theta')*180/torch.pi
            Station_Phi        = Event.get_value('Station_Phi')*180/torch.pi
            Station_Chi_i      = Event.get_value('Station_Chi_i')
            Station_Centroid   = Event.get_value('Station_Time')
            Station_Charge     = -1
            Station_PulseWidth = -1
            
            
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff  = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Chi_i_Diff = torch.abs(Pix_Chi_i[None,:] - Pix_Chi_i[:,None])
            Pix_Ang_Vel    = Pix_Chi_i_Diff/Pix_Time_Diff
            Pix_Time_Gap   = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff   = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff)
            Edges = torch.stack(torch.where(Edges),dim=1)
            Edges = Edges[Edges[:,0]!=Edges[:,1]]
            # Station will be the last node
            # It will connect to all other nodes
            Station_Edges = torch.stack([torch.arange(len(Pix_Theta)),torch.ones(len(Pix_Theta))*len(Pix_Theta)],dim=1).to(torch.long)

            Edge_ang_div  = Pix_Ang_Diff [Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]
            Edge_ang_Vel  = Pix_Ang_Vel  [Edges[:,0],Edges[:,1]]
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_ang_Vel  = Edge_ang_Vel*100/1.5
            # Find any nans or infs in Angular Velocity and replace with 0
            Edge_ang_Vel[torch.isnan(Edge_ang_Vel)] = 0
            Edge_ang_Vel[torch.isinf(Edge_ang_Vel)] = 0

            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div,Edge_ang_Vel],dim=1)

            Station_Edge_ang_div  = torch.sqrt((Station_Phi-Pix_Phi)**2 + (Station_Theta-Pix_Theta)**2) / 1.5
            Station_Edge_time_div = torch.abs(Station_Centroid-Pix_Centroid)                            / 100
            Station_Edge_Ang_Vel  =-(Station_Chi_i-Pix_Chi_i) / (Station_Centroid-Pix_Centroid)
            Station_Edge_values = torch.stack([Station_Edge_ang_div,Station_Edge_time_div,Station_Edge_Ang_Vel],dim=1)
            
            Edges = torch.cat([Edges,Station_Edges],dim=0)
            Edge_values = torch.cat([Edge_values,Station_Edge_values],dim=0)

            # Append Station Node
            Pix_Theta    = torch.cat([Pix_Theta   ,torch.tensor([Station_Theta])])
            Pix_Phi      = torch.cat([Pix_Phi     ,torch.tensor([Station_Phi])])
            Pix_Chi_i    = torch.cat([Pix_Chi_i   ,torch.tensor([Station_Chi_i])])
            Pix_Centroid = torch.cat([Pix_Centroid,torch.tensor([Station_Centroid])])

            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50
            
            Pix_Chi_i    -= torch.min(Pix_Chi_i)
            Pix_Centroid -= torch.min(Pix_Centroid)

            # append the Non-Normalised Station Values
            Pix_Charge   = torch.cat([Pix_Charge,torch.tensor([Station_Charge])])
            Pix_PulseWidth = torch.cat([Pix_PulseWidth,torch.tensor([Station_PulseWidth])])

            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Chi_i,Pix_Centroid,Pix_Charge,Pix_PulseWidth],dim=1)



            # Meta Values 
            Primary  = Event.get_value('Primary')
            LogE     = Event.get_value('Gen_LogE')
            Xmax     = Event.get_value('Gen_Xmax')
            # Truth Values
            # Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
            # Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            Gen_Chi0      = Event.get_value('Gen_Chi0')
            Gen_CEDist    = Event.get_value('Gen_CoreEyeDist')
            # Rec Values
            Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
            Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')
            Rec_Chi0      = Event.get_value('Rec_Chi0')
            Rec_CEDist    = Event.get_value('Rec_CoreEyeDist')
            
            Gen_SDP_Theta -= torch.pi/2
            Gen_Chi0      -= torch.pi/2
            Rec_SDP_Theta -= torch.pi/2
            Rec_Chi0      -= torch.pi/2
            
            Gen_x = torch.sin(Gen_Chi0)*torch.cos(Gen_SDP_Theta)
            Gen_y = torch.sin(Gen_SDP_Theta)
            Gen_z = torch.cos(Gen_Chi0)*torch.cos(Gen_SDP_Theta)

            Rec_x = torch.sin(Rec_Chi0)*torch.cos(Rec_SDP_Theta)
            Rec_y = torch.sin(Rec_SDP_Theta)
            Rec_z = torch.cos(Rec_Chi0)*torch.cos(Rec_SDP_Theta)

            
            TruthValues = torch.stack([Gen_x,Gen_y,Gen_z,Gen_SDP_Phi,Gen_CEDist],dim=0)
            RecValues   = torch.stack([Rec_x,Rec_y,Rec_z,Rec_SDP_Phi,Rec_CEDist],dim=0)
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            # Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0,0,0,0]))
            Rec.append(torch.tensor([0,0,0,0,0]))
            Meta.append(torch.tensor([0,0,0]))


    # Stack Data
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)

    
    # Normalise Truth and Rec out of Loop
    Truth[:,3][Truth[:,3]<0] += 2*torch.pi
    Truth[:,3] -= torch.pi/2
    Truth[:,3] -= AuxData[:,0]/180*torch.pi
    Truth[:,3] = torch.sin(Truth[:,3])
    Truth[:,4] /= lightspeed*1000

    Rec[:,3][Rec[:,3]<0] += 2*torch.pi
    Rec[:,3] -= torch.pi/2
    Rec[:,3] -= AuxData[:,0]/180*torch.pi
    Rec[:,3] = torch.sin(Rec[:,3])
    Rec[:,4] /= lightspeed*1000





    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_Axis
        ProcessingDataset.Truth_Keys        = ['x','y','z','SDPPhi','CEDist']
        ProcessingDataset.Truth_Units       = ['','','','deg','km']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Graph_Edges_wStation_wAngVelAsNodeAndEdge_Axis(Dataset,ProcessingDataset = None):
    '''Produces graphs with Pixel and Edge Values
    PixelValues are : Theta, Phi,Chi_i, Centroid, Charge, PulseWidth, Angular Velcoity 
    EdgeValues are  : Ang_Diff, Time_Diff, Angular Velocity
    Unique Node is added to represent the Station, it is connected to all nodes already present
    Also Passes the Truth and Rec and Aux values as they need to be normalised with the graph
    '''
    # PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    leniency = 1.2
    max_ang_diff = 1.8*torch.sqrt(torch.tensor(3))
    lightspeed = 299792458/1e7 # m/100ns
    
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
        Pix_Chi_i      = Event.get_pixel_values('Chi_i')
        Pix_Centroid   = Event.get_pixel_values('PulseCentroid')
        Pix_PulseStart = Event.get_pixel_values('PulseStart')
        Pix_PulseStop  = Event.get_pixel_values('PulseStop')
        Pix_Charge     = Event.get_pixel_values('Charge')
        TelID          = Event.get_pixel_values('TelID')
        PixelIDs       = Event.get_pixel_values('PixelID')
        Pixel_Status   = Event.get_pixel_values('Status')
        HEAT = (TelID == 7)+(TelID == 8)+(TelID == 9)+(Event.get_pixel_values('EyeID') == 5)
        
        Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
        Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            
        
        cut = cut_pixels(Pix_Theta/180*torch.pi,Pix_Phi/180*torch.pi,Gen_SDP_Theta,Gen_SDP_Phi,pixel_count=40)
        cut = cut & (Pixel_Status == 4) # For Testing Mostly
        Pix_Phi        = Pix_Phi       [cut]
        Pix_Theta      = Pix_Theta     [cut]
        Pix_Chi_i      = Pix_Chi_i     [cut]
        Pix_Centroid   = Pix_Centroid  [cut]
        Pix_PulseStart = Pix_PulseStart[cut]
        Pix_PulseStop  = Pix_PulseStop [cut]
        Pix_Charge     = Pix_Charge    [cut]
        TelID          = TelID         [cut]
        PixelIDs       = PixelIDs      [cut]
        HEAT           = HEAT          [cut]

        if not HEAT.all(): 
            # Get Station Values
            Station_Theta      = Event.get_value('Station_Theta')*180/torch.pi
            Station_Phi        = Event.get_value('Station_Phi')*180/torch.pi
            Station_Chi_i      = Event.get_value('Station_Chi_i')
            Station_Centroid   = Event.get_value('Station_Time')
            Station_Charge     = -1
            Station_PulseWidth = -1
            
            
            Pix_PulseStart[HEAT]= 0.5 * Pix_PulseStart[HEAT]
            Pix_PulseStop [HEAT]= 0.5 * Pix_PulseStop [HEAT]

            Pix_PulseWidth = (Pix_PulseStop - Pix_PulseStart)/2
            
            Pix_Time_Diff  = torch.abs(Pix_Centroid  [None,:] - Pix_Centroid  [:,None])
            Pix_Chi_i_Diff = torch.abs(Pix_Chi_i[None,:] - Pix_Chi_i[:,None])
            Pix_Ang_Vel    = Pix_Chi_i_Diff/Pix_Time_Diff
            Pix_Time_Gap   = torch.abs(Pix_PulseWidth[None,:] + Pix_PulseWidth[:,None])
            Pix_Ang_Diff   = torch.sqrt((Pix_Phi[None,:]-Pix_Phi[:,None])**2 + (Pix_Theta[None,:]-Pix_Theta[:,None])**2)
            
            Edges = (Pix_Time_Diff < leniency*Pix_Time_Gap) & (Pix_Ang_Diff<max_ang_diff)
            # Compute the median Angular Velocity for each node using Edges as mask before it is overwritten
            # print(Pix_Ang_Vel)
            Ang_Vel_masked = torch.where(Edges,Pix_Ang_Vel,torch.full_like(Pix_Ang_Vel,float('nan')))
            # print(Ang_Vel_masked)
            Pix_Ang_Vel_median = torch.nanmedian(Ang_Vel_masked,dim=1).values
            Pix_Ang_Vel_median[torch.isnan(Pix_Ang_Vel_median)] = 0
            Pix_Ang_Vel_median[torch.isinf(Pix_Ang_Vel_median)] = 0
            Pix_Ang_Vel_median = Pix_Ang_Vel_median*100/1.5
            Edges = torch.stack(torch.where(Edges),dim=1)
            Edges = Edges[Edges[:,0]!=Edges[:,1]]
            # Station will be the last node
            # It will connect to all other nodes
            Station_Edges = torch.stack([torch.arange(len(Pix_Theta)),torch.ones(len(Pix_Theta))*len(Pix_Theta)],dim=1).to(torch.long)

            Edge_ang_div  = Pix_Ang_Diff [Edges[:,0],Edges[:,1]]
            Edge_time_div = Pix_Time_Diff[Edges[:,0],Edges[:,1]]
            Edge_ang_Vel  = Pix_Ang_Vel  [Edges[:,0],Edges[:,1]]
            Edge_ang_div  = Edge_ang_div/1.5
            Edge_time_div = Edge_time_div/100
            Edge_ang_Vel  = Edge_ang_Vel*100/1.5
            # Find any nans or infs in Angular Velocity and replace with 0
            Edge_ang_Vel[torch.isnan(Edge_ang_Vel)] = 0
            Edge_ang_Vel[torch.isinf(Edge_ang_Vel)] = 0

            Edge_values   = torch.stack([Edge_ang_div,Edge_time_div,Edge_ang_Vel],dim=1)

            Station_Edge_ang_div  = torch.sqrt((Station_Phi-Pix_Phi)**2 + (Station_Theta-Pix_Theta)**2) / 1.5
            Station_Edge_time_div = torch.abs(Station_Centroid-Pix_Centroid)                            / 100
            Station_Edge_Ang_Vel  =-(Station_Chi_i-Pix_Chi_i) / (Station_Centroid-Pix_Centroid)
            Station_Edge_values = torch.stack([Station_Edge_ang_div,Station_Edge_time_div,Station_Edge_Ang_Vel],dim=1)
            
            Station_Ang_Vel = torch.median(Station_Edge_Ang_Vel)

            Edges = torch.cat([Edges,Station_Edges],dim=0)
            Edge_values = torch.cat([Edge_values,Station_Edge_values],dim=0)

            # Append Station Node
            Pix_Theta    = torch.cat([Pix_Theta   ,torch.tensor([Station_Theta])])
            Pix_Phi      = torch.cat([Pix_Phi     ,torch.tensor([Station_Phi])])
            Pix_Chi_i    = torch.cat([Pix_Chi_i   ,torch.tensor([Station_Chi_i])])
            Pix_Centroid = torch.cat([Pix_Centroid,torch.tensor([Station_Centroid])])
            Pix_Ang_Vel_median = torch.cat([Pix_Ang_Vel_median,torch.tensor([Station_Ang_Vel])])

            # Normalise Node Values
            Pix_Theta    = (90-Pix_Theta)/30
            Central_Phi  = torch.mean(Pix_Phi)
            Pix_Phi      = (Pix_Phi-Central_Phi)/40

            Pix_Centroid = Pix_Centroid/1000
            Pix_Charge   = torch.log10(Pix_Charge+1)/2.5-1
            Pix_PulseWidth = Pix_PulseWidth/50
            
            # Pix_Chi_i    -= torch.min(Pix_Chi_i)
            Pix_Centroid -= torch.min(Pix_Centroid)

            # append the Non-Normalised Station Values
            Pix_Charge   = torch.cat([Pix_Charge,torch.tensor([Station_Charge])])
            Pix_PulseWidth = torch.cat([Pix_PulseWidth,torch.tensor([Station_PulseWidth])])
            Node_values   = torch.stack([Pix_Theta,Pix_Phi,Pix_Chi_i,Pix_Centroid,Pix_Charge,Pix_PulseWidth,Pix_Ang_Vel_median],dim=1)


            # Meta Values 
            Primary  = Event.get_value('Primary')
            LogE     = Event.get_value('Gen_LogE')
            Xmax     = Event.get_value('Gen_Xmax')
            # Truth Values
            # Gen_SDP_Theta = Event.get_value('Gen_SDPTheta')
            # Gen_SDP_Phi   = Event.get_value('Gen_SDPPhi')
            Gen_Chi0      = Event.get_value('Gen_Chi0')
            Gen_CEDist    = Event.get_value('Gen_CoreEyeDist')
            # Rec Values
            Rec_SDP_Theta = Event.get_value('Rec_SDPTheta')
            Rec_SDP_Phi   = Event.get_value('Rec_SDPPhi')
            Rec_Chi0      = Event.get_value('Rec_Chi0')
            Rec_CEDist    = Event.get_value('Rec_CoreEyeDist')
            
            Gen_SDP_Theta -= torch.pi/2
            Gen_Chi0      -= torch.pi/2
            Rec_SDP_Theta -= torch.pi/2
            Rec_Chi0      -= torch.pi/2
            
            Gen_x = torch.sin(Gen_Chi0)*torch.cos(Gen_SDP_Theta)
            Gen_y = torch.sin(Gen_SDP_Theta)
            Gen_z = torch.cos(Gen_Chi0)*torch.cos(Gen_SDP_Theta)

            Rec_x = torch.sin(Rec_Chi0)*torch.cos(Rec_SDP_Theta)
            Rec_y = torch.sin(Rec_SDP_Theta)
            Rec_z = torch.cos(Rec_Chi0)*torch.cos(Rec_SDP_Theta)

            
            TruthValues = torch.stack([Gen_x,Gen_y,Gen_z,Gen_SDP_Phi,Gen_CEDist],dim=0)
            RecValues   = torch.stack([Rec_x,Rec_y,Rec_z,Rec_SDP_Phi,Rec_CEDist],dim=0)
            MetaValues  = torch.stack([Primary,LogE,Xmax])

            # Append to the Storage Lists
            Graph  .append([Node_values,Edges,Edge_values])
            AuxData.append(Central_Phi)
            Truth  .append(TruthValues)
            Rec    .append(RecValues)
            Meta   .append(MetaValues)
        else:
            Node_values = torch.tensor([[0,0,0,0,0,0]])
            Edges = torch.tensor([[0,0]])
            # Edge_values = torch.tensor([[0,0]])
            Graph.append([Node_values,Edges])
            AuxData.append(0)
            Truth.append(torch.tensor([0,0,0,0,0]))
            Rec.append(torch.tensor([0,0,0,0,0]))
            Meta.append(torch.tensor([0,0,0]))


    # Stack Data
    Meta    = torch.stack(Meta,dim=0)
    Truth   = torch.stack(Truth,dim=0)
    Rec     = torch.stack(Rec,dim=0)            
    AuxData = torch.tensor(AuxData).unsqueeze(1)

    
    # Normalise Truth and Rec out of Loop
    Truth[:,3][Truth[:,3]<0] += 2*torch.pi
    Truth[:,3] -= torch.pi/2
    Truth[:,3] -= AuxData[:,0]/180*torch.pi
    Truth[:,3] = torch.sin(Truth[:,3])
    Truth[:,4] /= lightspeed*1000

    Rec[:,3][Rec[:,3]<0] += 2*torch.pi
    Rec[:,3] -= torch.pi/2
    Rec[:,3] -= AuxData[:,0]/180*torch.pi
    Rec[:,3] = torch.sin(Rec[:,3])
    Rec[:,4] /= lightspeed*1000





    if ProcessingDataset is None:
        return Graph,AuxData,Truth,Rec,Meta,IDsList
    else:
        ProcessingDataset.GraphData = True
        ProcessingDataset._Graph    = Graph
        ProcessingDataset._Aux      = AuxData
        ProcessingDataset._Truth    = Truth
        ProcessingDataset._Rec      = Rec
        ProcessingDataset._MetaData = Meta
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Graph_Axis
        ProcessingDataset.Truth_Keys        = ['x','y','z','SDPPhi','CEDist']
        ProcessingDataset.Truth_Units       = ['','','','deg','km']
        
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'