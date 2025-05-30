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



def Unnormalise_Conv_SDP(Truth,AuxData=None):
    Truth[:,0] = torch.acos(Truth[:,0])/torch.pi*180
    Truth[:,1] = (torch.asin(Truth[:,1])+torch.pi/2)/torch.pi*180
    Truth[:,1][Truth[:,1]>180] -= 360
    if AuxData is None:
        return Truth
    else:
        Truth[:,1] += AuxData
        return Truth


def Conv_TripleTelescope_SDP(Dataset, ProcessingDataset = None):
    '''
    Function provides the triple telescope Data for the SDP Reconstruction
    For use in CNNs
    '''
    PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    IDsList = ()
    Main    = []
    Truth   = []
    Rec     = []
    AuxData = []
    Meta    = []

    for i, Event in enumerate(Dataset):

        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        Pix_Phi      = Event.get_pixel_values('Phi'            )
        Pix_Theta    = Event.get_pixel_values('Theta'          )
        Pix_ID       = Event.get_pixel_values('PixelID'        )
        Pix_TelID    = Event.get_pixel_values('TelID'          )
        Pix_Charge   = Event.get_pixel_values('Charge'         )
        Pix_Centroid = Event.get_pixel_values('PulseCentroid'  )
        Pix_PStart   = Event.get_pixel_values('PulseStart'     )
        Pix_PStop    = Event.get_pixel_values('PulseStop'      )
        
        Pix_PWidth   = Pix_PStop - Pix_PStart

        HEAT = (Pix_TelID == 7)+(Pix_TelID == 8)+(Pix_TelID == 9)+(Event.get_pixel_values('EyeID') == 5)  # For sanity check really
        
        # Get Truth and Rec
        Gen_SDPTheta = Event.get_value('Gen_SDPTheta')
        Gen_SDPPhi   = Event.get_value('Gen_SDPPhi'  )

        Rec_SDPTheta = Event.get_value('Rec_SDPTheta')
        Rec_SDPPhi   = Event.get_value('Rec_SDPPhi'  )

        # Get MetaData
        Primary = Event.get_value('Primary'    )
        LogE    = Event.get_value('Gen_LogE'   )
        Xmax    = Event.get_value('Gen_Xmax'   )

        # Process the Main data array

        if not HEAT.all(): # Shouldnt happen, but just in case
            # Get Telescope with most pixels
            Unique_Telescopes, counts = torch.unique(Pix_TelID,return_counts=True)
            Unique_Telescopes = Unique_Telescopes[torch.argsort(counts,descending=True)]

            Central_Tel = Unique_Telescopes[0].item()
            Left_Tel    = Central_Tel+1
            Right_Tel   = Central_Tel-1

            Main_Central = torch.zeros(5,20,22)
            Main_Left    = torch.zeros(5,20,22)
            Main_Right   = torch.zeros(5,20,22)

            
            # Central Telescope
            Mask  = Pix_TelID == Central_Tel

            Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
            Main_Central[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
            Main_Central[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
            Main_Central[2,Tel_Pix_X,Tel_Pix_Y] = Pix_PWidth  [Mask]/50
            Main_Central[3,Tel_Pix_X,Tel_Pix_Y] = (90-Pix_Theta   [Mask])/30
            Main_Central[4,Tel_Pix_X,Tel_Pix_Y] = (Pix_Phi      [Mask] - PhiOffsets[Central_Tel])/40

            # Left Telescope
            if Left_Tel == 7:
                Main_Left[...] = -1
            else:
                Mask  = Pix_TelID == Left_Tel

                Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
                Main_Left[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
                Main_Left[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
                Main_Left[2,Tel_Pix_X,Tel_Pix_Y] = Pix_PWidth  [Mask]/50
                Main_Left[3,Tel_Pix_X,Tel_Pix_Y] = (90-Pix_Theta   [Mask])/30
                Main_Left[4,Tel_Pix_X,Tel_Pix_Y] = (Pix_Phi      [Mask] - PhiOffsets[Left_Tel])/40
            
            # Right Telescope
            if Right_Tel == 0:
                Main_Right[...] = -1
            else:
                Mask  = Pix_TelID == Right_Tel

                Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
                Main_Right[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
                Main_Right[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
                Main_Right[2,Tel_Pix_X,Tel_Pix_Y] = Pix_PWidth  [Mask]/50
                Main_Right[3,Tel_Pix_X,Tel_Pix_Y] = (90-Pix_Theta   [Mask])/30
                Main_Right[4,Tel_Pix_X,Tel_Pix_Y] = (Pix_Phi      [Mask] - PhiOffsets[Right_Tel])/40

            This_Main  = torch.cat((Main_Left,Main_Central,Main_Right),dim=1)
            This_Truth = torch.tensor([Gen_SDPTheta,Gen_SDPPhi])
            This_Rec   = torch.tensor([Rec_SDPTheta,Rec_SDPPhi])
            This_Meta  = torch.tensor([Primary,LogE,Xmax])
            This_AuxData = torch.tensor([PhiOffsets[Central_Tel]*180/torch.pi])

        else:
            This_Main    = torch.zeros(3,60,22)
            This_Truth   = torch.zeros(2)
            This_Rec     = torch.zeros(2)
            This_Meta    = torch.zeros(3)
            This_AuxData = torch.zeros(1)

        Main.append(This_Main)
        Truth.append(This_Truth)
        Rec.append(This_Rec)
        Meta.append(This_Meta)
        AuxData.append(This_AuxData)


    # Normalisation of Data
    Main    = torch.stack(Main)
    Truth   = torch.stack(Truth)    
    Rec     = torch.stack(Rec)
    Meta    = torch.stack(Meta)
    AuxData = torch.stack(AuxData)

    # Main Normalised in Loop

    # SDPTheta
    Truth[:,0] = torch.cos(Truth[:,0])
    Rec[:,0]   = torch.cos(Rec[:,0])

    # SDPPhi
    Truth[:,1][Truth[:,1]<0] += 2*torch.pi
    Rec  [:,1][Rec  [:,1]<0] += 2*torch.pi

    Truth[:,1] -= torch.pi/2
    Rec  [:,1] -= torch.pi/2
    
    Truth[:,1] -= AuxData.squeeze()/180*torch.pi
    Rec  [:,1] -= AuxData.squeeze()/180*torch.pi

    Truth[:,1]  = torch.sin(Truth[:,1])
    Rec  [:,1]  = torch.sin(Rec  [:,1])
 
    

    # Slap into Dataset
    if ProcessingDataset is None:
        return Main, AuxData, Truth, Rec, Meta, IDsList
    
    else:
        ProcessingDataset._Main.append(Main) # Must append only main for multiple Mains possibility
        ProcessingDataset._Aux   = AuxData
        ProcessingDataset._Truth = Truth
        ProcessingDataset._Rec   = Rec
        ProcessingDataset._Meta  = Meta

        ProcessingDataset.Unnormalise_Truth = Unnormalise_Conv_SDP
        ProcessingDataset.Truth_Keys        = ['SDPTheta','SDPPhi']
        ProcessingDataset.Truth_Units       = ['deg','deg']

        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'



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

def Main_TripleTelescope_Charge_and_Time(Dataset, ProcessingDataset = None):
    '''
    Function provides the triple telescope Data for the SDP Reconstruction
    For use in CNNs
    '''
    PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    IDsList = ()
    Main    = []
    AuxData = []
    
    for i, Event in enumerate(Dataset):

        ID = (Event.get_value('EventID_1/2').int()*100000 + Event.get_value('EventID_2/2').int()).item()
        IDsList += (ID,)

        Pix_ID       = Event.get_pixel_values('PixelID'        )
        Pix_TelID    = Event.get_pixel_values('TelID'          )
        Pix_Charge   = Event.get_pixel_values('Charge'         )
        Pix_Centroid = Event.get_pixel_values('PulseCentroid'  )
        HEAT = (Pix_TelID == 7)+(Pix_TelID == 8)+(Pix_TelID == 9)+(Event.get_pixel_values('EyeID') == 5)  # For sanity check really
        
        if not HEAT.all(): # Shouldnt happen, but just in case
            # Get Telescope with most pixels
            Unique_Telescopes, counts = torch.unique(Pix_TelID,return_counts=True)
            Unique_Telescopes = Unique_Telescopes[torch.argsort(counts,descending=True)]

            Central_Tel = Unique_Telescopes[0].item()
            Left_Tel    = Central_Tel+1
            Right_Tel   = Central_Tel-1

            Main_Central = torch.zeros(2,20,22)
            Main_Left    = torch.zeros(2,20,22)
            Main_Right   = torch.zeros(2,20,22)

            
            # Central Telescope
            Mask  = Pix_TelID == Central_Tel

            Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
            Main_Central[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
            Main_Central[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
            
            # Left Telescope
            if Left_Tel == 7:
                Main_Left[...] = -1
            else:
                Mask  = Pix_TelID == Left_Tel

                Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
                Main_Left[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
                Main_Left[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
                
            # Right Telescope
            if Right_Tel == 0:
                Main_Right[...] = -1
            else:
                Mask  = Pix_TelID == Right_Tel

                Tel_Pix_X, Tel_Pix_Y = IndexToXY(Pix_ID[Mask],return_tensor=True)
                Main_Right[0,Tel_Pix_X,Tel_Pix_Y] = torch.log10(Pix_Charge  [Mask] +1) /2.5 -1
                Main_Right[1,Tel_Pix_X,Tel_Pix_Y] = Pix_Centroid[Mask]/1000
                
            This_Main  = torch.cat((Main_Left,Main_Central,Main_Right),dim=1)
            This_AuxData = torch.tensor([PhiOffsets[Central_Tel]*180/torch.pi])

        else:
            This_Main    = torch.zeros(2,60,22)
            This_AuxData = torch.zeros(1)

        Main.append(This_Main)
        AuxData.append(This_AuxData)


    # Normalisation of Data
    Main    = torch.stack(Main)
    AuxData = torch.stack(AuxData)

    # Main Normalised in Loop

    # Slap into Dataset
    if ProcessingDataset is None:
        return Main, AuxData, IDsList
    
    else:
        ProcessingDataset._Main.append(Main) # Must append only main for multiple Mains possibility
        ProcessingDataset._Aux   = AuxData
        
        ProcessingDataset.Unnormalise_Truth = Unnormalise_Conv_SDP
        ProcessingDataset.Truth_Keys        = ['SDPTheta','SDPPhi']
        ProcessingDataset.Truth_Units       = ['deg','deg']

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


