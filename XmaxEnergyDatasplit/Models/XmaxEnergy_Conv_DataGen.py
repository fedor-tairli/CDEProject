import torch
# import numpy as np
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
        # Traces = torch.log1p((Traces).clip(min=0))
        Traces = Traces.clip(min=0)
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



def Unnormalise_XmaxEnergy(XmaxEnergy):
    '''Will unnormalise Xmax and Energy'''
    # Normalise Xmax
    XmaxMean = 591
    XmaxStd  = 72
    XmaxEnergy[:,0] = XmaxEnergy[:,0] * XmaxStd + XmaxMean

    # Normalise Energy
    EnergyMean = 16.15
    EnergyStd  = 0.475
    XmaxEnergy[:,1] = XmaxEnergy[:,1] * EnergyStd + EnergyMean

    return XmaxEnergy

def Truth_XmaxEnergy(Dataset,ProcessingDataset):
    '''Will provide Xmax and Energy for each event'''
    IDsList = ()
    Gen_Xmax   = []
    Gen_Energy = []
    Rec_Xmax   = []
    Rec_Energy = []

    for i,Event in enumerate(Dataset):
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)
        if i%100 == 0:
            print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        Xmax   = Event.get_value('Gen_Xmax')
        Energy = Event.get_value('Gen_LogE')
        Gen_Xmax  .append(Xmax)
        Gen_Energy.append(Energy)

        Xmax   = Event.get_value('Rec_Xmax')
        Energy = Event.get_value('Rec_LogE')
        Rec_Xmax  .append(Xmax)
        Rec_Energy.append(Energy)
    

    Gen_Xmax   = torch.stack(Gen_Xmax)
    Gen_Energy = torch.stack(Gen_Energy)
    Rec_Xmax   = torch.stack(Rec_Xmax)
    Rec_Energy = torch.stack(Rec_Energy)
    

    # Normalise Xmax
    XmaxMean = 591
    XmaxStd  = 72
    Gen_Xmax = (Gen_Xmax - XmaxMean) / XmaxStd
    Rec_Xmax = (Rec_Xmax - XmaxMean) / XmaxStd

    # Normalise Energy
    EnergyMean = 16.15
    EnergyStd  = 0.475
    Gen_Energy = (Gen_Energy - EnergyMean) / EnergyStd
    Rec_Energy = (Rec_Energy - EnergyMean) / EnergyStd

    if ProcessingDataset is None:
        return torch.stack((Gen_Xmax,Gen_Energy),dim =1) , torch.stack((Rec_Xmax,Rec_Energy),dim =1)
    ProcessingDataset._Truth = torch.stack((Gen_Xmax,Gen_Energy),dim =1)
    ProcessingDataset._Rec   = torch.stack((Rec_Xmax,Rec_Energy),dim =1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_XmaxEnergy
    ProcessingDataset.Truth_Keys  = ('Xmax','LogE')
    ProcessingDataset.Truth_Units = ('g/cm^2','')
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
    
    Angular_Velocity = torch.zeros(len(Dataset),1)


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

        # Calculate Angular Velocity
        pix_Thetas    = Event.get_pixel_values('Theta')
        pix_Phis      = Event.get_pixel_values('Phi')
        pix_Centroids = Event.get_pixel_values('PulseCentroid')
        pix_Status    = Event.get_pixel_values('Status')

        Cut        = pix_Status == 4
        pix_Thetas = pix_Thetas[Cut]
        pix_Phis   = pix_Phis  [Cut]
        pix_Centroids = pix_Centroids[Cut]

        if pix_Thetas.shape[0] < 2:
            Angular_Velocity[i] = 0
        else:
            Gen_SDPTheta = Event.get_value('Gen_SDPTheta').item()
            Gen_SDPPhi   = Event.get_value('Gen_SDPPhi').item()

            # Angular Span
            theta_r = torch.deg2rad(torch.tensor(Gen_SDPTheta, dtype=torch.float32))
            phi_r   = torch.deg2rad(torch.tensor(Gen_SDPPhi,   dtype=torch.float32))
            SDP_Vec = torch.stack([
                torch.sin(theta_r) * torch.cos(phi_r),
                torch.sin(theta_r) * torch.sin(phi_r),
                torch.cos(theta_r)
            ])
            SDP_Vec = SDP_Vec / torch.linalg.norm(SDP_Vec)

            z_axis   = torch.tensor([0., 0., 1.])
            SDP_Orth = torch.linalg.cross(SDP_Vec, z_axis) if not torch.allclose(SDP_Vec, z_axis) else torch.tensor([1., 0., 0.])
            SDP_Orth = SDP_Orth / torch.linalg.norm(SDP_Orth)
            SDP_Perp = torch.linalg.cross(SDP_Vec, SDP_Orth)

            pix_az   = torch.deg2rad(pix_Phis)
            pix_el   = torch.deg2rad(pix_Thetas)
            pix_vecs = torch.stack([
                torch.cos(pix_el) * torch.cos(pix_az),
                torch.cos(pix_el) * torch.sin(pix_az),
                torch.sin(pix_el)
            ], dim=1)

            proj   = pix_vecs - torch.outer(pix_vecs @ SDP_Vec, SDP_Vec)
            proj   = proj / torch.linalg.norm(proj, dim=1, keepdim=True)
            angles = torch.sort(torch.arctan2(proj @ SDP_Perp, proj @ SDP_Orth) % (2 * torch.pi)).values
            gaps   = torch.cat([torch.diff(angles), (angles[0] + 2 * torch.pi - angles[-1]).unsqueeze(0)])
            Angular_Span = torch.rad2deg(2 * torch.pi - gaps.max())

            # Duration
            if pix_Centroids.numel() == 0:
                Duration = 0.0
            else:
                Duration = pix_Centroids.max().item() - pix_Centroids.min().item()

            # Angular velocity
            Angular_Velocity[i] = Angular_Span.item() / Duration if Duration > 0 else 0

    if ProcessingDataset is None:
        return torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Angular_Velocity),dim=1)
    else:
        ProcessingDataset._Aux = torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Angular_Velocity),dim=1)
        ProcessingDataset.Aux_Keys = ('Event_Class','Primary','LogE','CosZenith','Xmax','Chi0','Rp','AngularVelocity')
        ProcessingDataset.Aux_Units = ('','','','','g/cm^2','rad','m','Deg/100ns')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'



def High_Angular_Velocity_Cut(ProcessingDataset,**kwargs):

    All_Angular_Velocities = ProcessingDataset._Aux[:,7] # Assuming Angular Velocity is the 8th column in Aux
    Threshold = kwargs.get('Angular_Velocity_Threshold', 3.0) # Default threshold value
    Good_Event_Mask = All_Angular_Velocities >= Threshold
    Good_Event_Mask &= All_Angular_Velocities <20 # Also remove the very high angular velocity events, which are likely to be noise
    ProcessingDataset._Good_Event_Mask = Good_Event_Mask.squeeze()
    print(f'Applied Angular Velocity cut at {Threshold} Deg/100ns, keeping {Good_Event_Mask.sum().item()} out of {len(Good_Event_Mask)} events.')


def Low_Angular_Velocity_Cut(ProcessingDataset,**kwargs):

    All_Angular_Velocities = ProcessingDataset._Aux[:,7] # Assuming Angular Velocity is the 8th column in Aux
    Threshold = kwargs.get('Angular_Velocity_Threshold', 3.0) # Default threshold value
    Good_Event_Mask = All_Angular_Velocities <= Threshold
    Good_Event_Mask &= All_Angular_Velocities <20 # Also remove the very high angular velocity events, which are likely to be noise
    ProcessingDataset._Good_Event_Mask = Good_Event_Mask.squeeze()
    print(f'Applied Angular Velocity cut at {Threshold} Deg/100ns, keeping {Good_Event_Mask.sum().item()} out of {len(Good_Event_Mask)} events.')

def All_Angular_Velocity(ProcessingDataset,**kwargs):
    All_Angular_Velocities = ProcessingDataset._Aux[:,7] # Assuming Angular Velocity is the 8th column in Aux
    Good_Event_Mask = (All_Angular_Velocities >= 0) & (All_Angular_Velocities <20) # Keep all events with valid angular velocity
    ProcessingDataset._Good_Event_Mask = Good_Event_Mask.squeeze()
    print(f'Keeping all events with valid Angular Velocity, keeping {Good_Event_Mask.sum().item()} out of {len(Good_Event_Mask)} events.')

    