import torch
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly


# Once Defined, the functions are to be unchanged
########################################################################################################################

############################################################
# Supporting Values and functions for Geometry
############################################################


# Pixel Indexing 
def IndexToXY(indices,return_tensor=False):
    indices -=1
    Xs = indices//22
    Ys = indices%22
    if return_tensor: return Xs.int(),Ys.int()
    else:             return Xs.int().tolist(),Ys.int().tolist()


# Telescope Offsets
HE_1_OA_Phi   = (   44.45)/180*torch.pi
HE_1_OA_Theta = (90-45.07)/180*torch.pi
HE_2_OA_Phi   = (   89.87)/180*torch.pi
HE_2_OA_Theta = (90-45.58)/180*torch.pi
HE_3_OA_Phi   = (  132.83)/180*torch.pi
HE_3_OA_Theta = (90-44.85)/180*torch.pi

HE_OA_Phi   = {1: HE_1_OA_Phi  , 2: HE_2_OA_Phi  , 3: HE_3_OA_Phi  }
HE_OA_Theta = {1: HE_1_OA_Theta, 2: HE_2_OA_Theta, 3: HE_3_OA_Theta}

# Camera Transormation aux functions
def get_telescope_optical_axis(TelID):
    Tel_OA_Theta = torch.tensor([ HE_OA_Theta[TelID] ])
    Tel_OA_Phi   = torch.tensor([ HE_OA_Phi  [TelID] ])

    return torch.tensor([ torch.cos(Tel_OA_Theta)*torch.cos(Tel_OA_Phi),
                            torch.cos(Tel_OA_Theta)*torch.sin(Tel_OA_Phi),
                            torch.sin(Tel_OA_Theta) 
                        ], dtype=torch.float32)

def make_SDP_vector(SDP_Theta,SDP_Phi):
    if not isinstance(SDP_Theta,torch.Tensor):
        SDP_Theta = torch.tensor(SDP_Theta)
    if not isinstance(SDP_Phi,torch.Tensor):
        SDP_Phi = torch.tensor(SDP_Phi)

    SDP_vec =  torch.tensor([ torch.cos(SDP_Theta)*torch.cos(SDP_Phi),
                              torch.cos(SDP_Theta)*torch.sin(SDP_Phi),
                              torch.sin(SDP_Theta) 
                            ], dtype=torch.float32) 
    return SDP_vec / torch.norm(SDP_vec,dim=0,keepdim=True)

def make_camera_plane_basis(TelOA, Global_Up = None):
    if Global_Up is None:
        Global_Up = torch.tensor([0,0,1],dtype=torch.float32)
    
    # Construct the Camera Plane Basis Vectors
    Z_camera = TelOA
    Z_camera = Z_camera / torch.norm(Z_camera,dim=0,keepdim=True) # Should be already normalised, but just in case
    X_camera = Global_Up - torch.dot(Global_Up,Z_camera) * Z_camera
    X_camera = X_camera / torch.norm(X_camera,dim=0,keepdim=True)
    Y_camera = torch.cross(Z_camera,X_camera)
    Y_camera = Y_camera / torch.norm(Y_camera,dim=0,keepdim=True)

    return X_camera, Y_camera, Z_camera

def produce_camera_plane_geometry_Angles(SDP_Theta,SDP_Phi,Chi0,Rp,TelID):
    '''Normalise geometry to the camera plane
    Return the Intersection Coords and Shower Axis via spherical angles
    '''
    # If the inputs are not tensors, convert them to tensors
    if not isinstance(SDP_Theta,torch.Tensor):
        SDP_Theta = torch.tensor(SDP_Theta)
    if not isinstance(SDP_Phi,torch.Tensor):
        SDP_Phi = torch.tensor(SDP_Phi)
    if not isinstance(Chi0,torch.Tensor):
        Chi0 = torch.tensor(Chi0)
    if not isinstance(Rp,torch.Tensor):
        Rp = torch.tensor(Rp)
    # Starting Values
    SDP_vec = make_SDP_vector(SDP_Theta,SDP_Phi)
    Tel_OA = get_telescope_optical_axis(TelID)
    Global_Up = torch.tensor([0,0,1],dtype=torch.float32)

    # Construct the SDP
    Ground_normal = torch.tensor([0,0,1],dtype=torch.float32)
    Ground_in_SDP = torch.cross(SDP_vec, torch.cross(Ground_normal,SDP_vec))
    Ground_in_SDP = Ground_in_SDP / torch.norm(Ground_in_SDP,dim=0,keepdim=True)
    # Construct perpendicular point
    Ground_Perp_in_SDP = torch.cross(SDP_vec, Ground_in_SDP)    
    Ground_Perp_in_SDP = Ground_Perp_in_SDP / torch.norm(Ground_Perp_in_SDP,dim=0,keepdim=True)
    Shower_Axis_dir = torch.cos(Chi0) * SDP_vec + torch.sin(Chi0) * Ground_in_SDP
    Shower_Axis_dir = Shower_Axis_dir / torch.norm(Shower_Axis_dir,dim=0,keepdim=True)
    Shower_Perp_dir = torch.cross(SDP_vec, Shower_Axis_dir)
    Shower_Perp_dir = Shower_Perp_dir / torch.norm(Shower_Perp_dir,dim=0,keepdim=True)
    Shower_Perp_Point = - Rp * Shower_Perp_dir

    # Construct the Camera Plane Intersection (using Tel_OA and 0,0,0)
    t_intersection = -torch.dot(Tel_OA,Shower_Perp_Point) / torch.dot(Tel_OA,Shower_Axis_dir)
    intersection_point = Shower_Perp_Point + t_intersection * Shower_Axis_dir

    # Construct the Camera Plane Basis Vectors
    X_camera, Y_camera, Z_camera = make_camera_plane_basis(Tel_OA, Global_Up)

    # Produce the Azimuth and Zenith angles Shower axis makes with Camera Normal
    Shower_Axis_camera  = torch.tensor([ torch.dot(Shower_Axis_dir,X_camera),
                                        torch.dot(Shower_Axis_dir,Y_camera),
                                        torch.dot(Shower_Axis_dir,Z_camera) ], dtype=torch.float32)
    Shower_Axis_camera = Shower_Axis_camera / torch.norm(Shower_Axis_camera,dim=0,keepdim=True) # By definition it should be normalised, but just in case
    
    Shower_Azimuth = torch.atan2(Shower_Axis_camera[1],Shower_Axis_camera[0])
    Shower_Zenith  = torch.acos(Shower_Axis_camera[2])

    # Produce the intersection point in Camera Plane
    X_Intersection = torch.dot(intersection_point,X_camera)
    Y_Intersection = torch.dot(intersection_point,Y_camera)
    Z_Intersection = torch.dot(intersection_point,Z_camera)

    # # Make an assertion that the Z_intersection is 0 , # Kinda sometimes its not zero actually, so skipping this for now
    # assert torch.isclose(Z_Intersection, torch.tensor(0.0, dtype=torch.float32)), \
    #     f'Z Intersection is not 0, it is {Z_Intersection.item()}'
    
    return X_Intersection, Y_Intersection, Shower_Zenith, Shower_Azimuth


def produce_camera_plane_geometry_Axis(SDP_Theta,SDP_Phi,Chi0,Rp,TelID):
    '''Normalise geometry to the camera plane
    Return the Intersection Coords and Shower Axis in Camera Plane
    '''
    # If the inputs are not tensors, convert them to tensors
    if not isinstance(SDP_Theta,torch.Tensor):
        SDP_Theta = torch.tensor(SDP_Theta)
    if not isinstance(SDP_Phi,torch.Tensor):
        SDP_Phi = torch.tensor(SDP_Phi)
    if not isinstance(Chi0,torch.Tensor):
        Chi0 = torch.tensor(Chi0)
    if not isinstance(Rp,torch.Tensor):
        Rp = torch.tensor(Rp)

    # Starting Values
    SDP_vec = make_SDP_vector(SDP_Theta,SDP_Phi)
    Tel_OA = get_telescope_optical_axis(TelID)
    Global_Up = torch.tensor([0,0,1],dtype=torch.float32)

    # Construct the SDP
    Ground_normal = torch.tensor([0,0,1],dtype=torch.float32)
    Ground_in_SDP = torch.cross(SDP_vec, torch.cross(Ground_normal,SDP_vec))
    Ground_in_SDP = Ground_in_SDP / torch.norm(Ground_in_SDP,dim=0,keepdim=True)
    # Construct perpendicular point
    Ground_Perp_in_SDP = torch.cross(SDP_vec, Ground_in_SDP)    
    Ground_Perp_in_SDP = Ground_Perp_in_SDP / torch.norm(Ground_Perp_in_SDP,dim=0,keepdim=True)
    Shower_Axis_dir = torch.cos(Chi0) * SDP_vec + torch.sin(Chi0) * Ground_in_SDP
    Shower_Axis_dir = Shower_Axis_dir / torch.norm(Shower_Axis_dir,dim=0,keepdim=True)
    Shower_Perp_dir = torch.cross(SDP_vec, Shower_Axis_dir)
    Shower_Perp_dir = Shower_Perp_dir / torch.norm(Shower_Perp_dir,dim=0,keepdim=True)
    Shower_Perp_Point = - Rp * Shower_Perp_dir

    # Construct the Camera Plane Intersection (using Tel_OA and 0,0,0)
    t_intersection = -torch.dot(Tel_OA,Shower_Perp_Point) / torch.dot(Tel_OA,Shower_Axis_dir)
    intersection_point = Shower_Perp_Point + t_intersection * Shower_Axis_dir

    # Construct the Camera Plane Basis Vectors
    X_camera, Y_camera, Z_camera = make_camera_plane_basis(Tel_OA, Global_Up)

    # Produce the Azimuth and Zenith angles Shower axis makes with Camera Normal
    Shower_Axis_camera  = torch.tensor([ torch.dot(Shower_Axis_dir,X_camera),
                                        torch.dot(Shower_Axis_dir,Y_camera),
                                        torch.dot(Shower_Axis_dir,Z_camera) ], dtype=torch.float32)
    Shower_Axis_camera = Shower_Axis_camera / torch.norm(Shower_Axis_camera,dim=0,keepdim=True) # By definition it should be normalised, but just in case
    
    # Shower_Azimuth = torch.atan2(Shower_Axis_camera[1],Shower_Axis_camera[0])
    # Shower_Zenith  = torch.acos(Shower_Axis_camera[2])

    # Produce the intersection point in Camera Plane
    X_Intersection = torch.dot(intersection_point,X_camera)
    Y_Intersection = torch.dot(intersection_point,Y_camera)
    Z_Intersection = torch.dot(intersection_point,Z_camera)

    # # Make an assertion that the Z_intersection is 0 # Kinda sometimes its not zero actually, so skipping this for now
    # assert torch.isclose(Z_Intersection, torch.tensor(0.0, dtype=torch.float32),atol = 1e-3), \
    #     f'Z Intersection is not 0, it is {Z_Intersection.item()}'
    
    return X_Intersection, Y_Intersection, Shower_Axis_camera[0], Shower_Axis_camera[1], Shower_Axis_camera[2]


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
    Gen_Cherenkov = torch.zeros(len(Dataset),1) 
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
        Gen_Cherenkov[i] = Event.get_value('Gen_CherenkovFraction')

    
    if ProcessingDataset is None:
        return torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_Cherenkov),dim=1)
    else:
        ProcessingDataset._Aux = torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_Cherenkov),dim=1)
        ProcessingDataset.Aux_Keys = ('Event_Class','Primary','LogE','CosZenith','Xmax','Chi0','Rp','ChrenkovFraction')
        ProcessingDataset.Aux_Units = ('','','','','g/cm^2','rad','m','')
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


def Unnormalise_Geometry_CameraPlane_Axis(Data):
    ''' Unnormalise the geometry of the events withing the axis based camera plane'''

    Data[0] = Data[0] * 1.83 # X
    Data[1] = Data[1] * 4.89 # Y
    # Data[2] = Data[2] # Axis_X
    # Data[3] = Data[3] # Axis_Y
    # Data[4] = Data[4] # Axis_Z

    return Data

def Geometry_InCameraPlane_Axis(Dataset,ProcessingDataset):

    ''' Transform the geometry of the events to be referenced in the camera plane'''

    IDsList = ()
    Gen_X       = torch.zeros(len(Dataset),1)
    Gen_Y       = torch.zeros(len(Dataset),1)
    # Gen_Z       = torch.zeros(len(Dataset),1) # This one should always 0 in camera plane
    Gen_Axis_X  = torch.zeros(len(Dataset),1)
    Gen_Axis_Y  = torch.zeros(len(Dataset),1)
    Gen_Axis_Z  = torch.zeros(len(Dataset),1)

    Rec_X       = torch.zeros(len(Dataset),1)
    Rec_Y       = torch.zeros(len(Dataset),1)
    # Rec_Z       = torch.zeros(len(Dataset),1) 
    Rec_Axis_X  = torch.zeros(len(Dataset),1)
    Rec_Axis_Y  = torch.zeros(len(Dataset),1)
    Rec_Axis_Z  = torch.zeros(len(Dataset),1)

    # Loop through the events and get the geometry
    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Geometry {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values of the standard geometry
        This_TelID    = Event.get_pixel_values('TelID')[0].item()

        This_Gen_SDPTheta = Event.get_value('Gen_SDPTheta')
        This_Gen_SDPPhi   = Event.get_value('Gen_SDPPhi')
        This_Gen_Chi0     = Event.get_value('Gen_Chi0')
        This_Gen_Rp       = Event.get_value('Gen_Rp') /1000 # Convert to km
        
        This_Rec_SDPTheta = Event.get_value('Rec_SDPTheta')
        This_Rec_SDPPhi   = Event.get_value('Rec_SDPPhi')
        This_Rec_Chi0     = Event.get_value('Rec_Chi0')
        This_Rec_Rp       = Event.get_value('Rec_Rp') /1000 # Convert to km


        Gen_X[i] , Gen_Y[i] , Gen_Axis_X[i], Gen_Axis_Y[i], Gen_Axis_Z[i] = produce_camera_plane_geometry_Axis(This_Gen_SDPTheta, This_Gen_SDPPhi, This_Gen_Chi0, This_Gen_Rp, This_TelID)
        Rec_X[i] , Rec_Y[i] , Rec_Axis_X[i], Rec_Axis_Y[i], Rec_Axis_Z[i] = produce_camera_plane_geometry_Axis(This_Rec_SDPTheta, This_Rec_SDPPhi, This_Rec_Chi0, This_Rec_Rp, This_TelID)


    # Some Normalisation here:
    # Gen_Axis_X, Gen_Axis_Y, Gen_Axis_Z  are already normalised

    # Since Rp is in KM: use /1.83 for X and 4.89 for Y

    Gen_X = Gen_X / 1.83
    Gen_Y = Gen_Y / 4.89

    Rec_X = Rec_X / 1.83
    Rec_Y = Rec_Y / 4.89


    # Slap into the dataset
    if ProcessingDataset is None:
        return torch.stack((Gen_X,Gen_Y,Gen_Axis_X,Gen_Axis_Y,Gen_Axis_Z),dim=1), \
               torch.stack((Rec_X,Rec_Y,Rec_Axis_X,Rec_Axis_Y,Rec_Axis_Z),dim=1)
    else:
        ProcessingDataset._Truth = torch.stack((Gen_X,Gen_Y,Gen_Axis_X,Gen_Axis_Y,Gen_Axis_Z),dim=1)
        ProcessingDataset._Rec   = torch.stack((Rec_X,Rec_Y,Rec_Axis_X,Rec_Axis_Y,Rec_Axis_Z),dim=1)

        ProcessingDataset.Truth_Keys = ('X','Y','Axis_X','Axis_Y','Axis_Z')
        ProcessingDataset.Truth_Units = ('km','km','','','','')

        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'

