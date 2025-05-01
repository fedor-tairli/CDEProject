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



def Unnormalise_SDP(Truth): 
    # Just for Truth Func below
        Gen_SDPTheta = torch.acos(Truth[:,0])*180/torch.pi
        Gen_SDPPhi   = torch.asin(Truth[:,1])*180/torch.pi
        return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)


def Unnormalise_SDP_NoTrig(Truth):
    Gen_SDPTheta = Truth[:,0]*180/torch.pi
    Gen_SDPPhi   = Truth[:,1]*180/torch.pi
    return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)



def Unnormalise_SDP(Truth,AuxData = None): 
    # Just for Truth Func below
    Gen_SDPTheta = torch.acos(Truth[:,0])*180/torch.pi
    Gen_SDPPhi   = torch.asin(Truth[:,1])*180/torch.pi if AuxData is None else torch.asin(Truth[:,1])*180/torch.pi+AuxData
    return torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1)

def LSTM_SDP(Dataset, ProcessingDataset = None):
    '''
    Function to construct the SDP Dataset for the LSTM reconstruction
    '''
    PhiOffsets = {1:-75/180*torch.pi,2:-45/180*torch.pi,3:-15/180*torch.pi,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}
    IDsList = ()
    Main    = []
    Main_st = []
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
        Pix_PStart   = Event.get_pixel_values('PulseStart'     )
        Pix_PStop    = Event.get_pixel_values('PulseStop'      )
        Pix_TOffset  = Event.get_pixel_values('TimeOffset'     )        
        Pix_Status   = Event.get_pixel_values('Status'         )

        Pix_Traces   = Event.get_trace_values()

        Pix_Phi     = Pix_Phi[Pix_Status>2]
        Pix_Theta   = Pix_Theta[Pix_Status>2]
        Pix_ID      = Pix_ID[Pix_Status>2]
        Pix_TelID   = Pix_TelID[Pix_Status>2]
        Pix_PStart  = Pix_PStart[Pix_Status>2]
        Pix_PStop   = Pix_PStop[Pix_Status>2]
        Pix_TOffset = Pix_TOffset[Pix_Status>2]
        Pix_Status  = Pix_Status[Pix_Status>2]

        Pix_Pstart  = Pix_PStart.int()
        Pix_Pstop   = Pix_PStop.int()
        Pix_TOffset = Pix_TOffset.int()
        


        Comb_Trace_Start = (Pix_PStart+Pix_TOffset).min().item()
        Comb_Trace_Stop  = (Pix_PStop +Pix_TOffset).max().item()
        Comb_Trace_Start = int(Comb_Trace_Start)
        Comb_Trace_Stop  = int(Comb_Trace_Stop)
        
        CombTrace = torch.zeros(Comb_Trace_Stop-Comb_Trace_Start)
        CombTheta = torch.zeros(Comb_Trace_Stop-Comb_Trace_Start)
        CombPhi   = torch.zeros(Comb_Trace_Stop-Comb_Trace_Start)

        CombWeights = torch.zeros(Comb_Trace_Stop-Comb_Trace_Start)
        CombNpix    = torch.zeros(Comb_Trace_Stop-Comb_Trace_Start)

        for iPix in range(len(Pix_Phi)):
            This_Comb_PStart =      Pix_PStart[iPix]+Pix_TOffset[iPix]-Comb_Trace_Start
            This_Comb_PStop  = min([Pix_PStop [iPix]+Pix_TOffset[iPix]-Comb_Trace_Start,This_Comb_PStart+100])
            This_Comb_PStart = int(This_Comb_PStart.item())
            This_Comb_PStop  = int(This_Comb_PStop .item())
            This_PulseLenght = min([100,(Pix_Pstop[iPix]-Pix_Pstart[iPix]).item()])
            This_Trace = Pix_Traces[iPix,:This_PulseLenght]
            CombTrace  [This_Comb_PStart:This_Comb_PStop] += This_Trace
            CombTheta  [This_Comb_PStart:This_Comb_PStop] += Pix_Theta[iPix]#*torch.abs(This_Trace)
            CombPhi    [This_Comb_PStart:This_Comb_PStop] += Pix_Phi[iPix]  #*torch.abs(This_Trace)
            CombWeights[This_Comb_PStart:This_Comb_PStop] += 1#torch.abs(This_Trace)
            CombNpix   [This_Comb_PStart:This_Comb_PStop] += 1

        CombTheta = CombTheta/CombWeights
        CombPhi   = CombPhi  /CombWeights

        # Normalize
        NoSigMask = (CombTrace==0)|(CombWeights==0)      # care - it is possible to have zero total signal
        CombTrace = torch.log10(torch.clip((CombTrace+1),min=1))
        CombTheta = 90-CombTheta
        CentralCombPhi = CombPhi[~NoSigMask].mean()
        CombPhi   = CombPhi - CentralCombPhi

        # CombTheta = torch.cos(CombTheta*torch.pi/180)
        # CombPhi   = torch.sin(CombPhi  *torch.pi/180)

        CombTheta[NoSigMask] = -1              
        CombPhi  [NoSigMask] = -1
        
        This_Main = torch.zeros(1000,3)
        This_Main[:CombTrace.shape[0],0] = CombTrace[-1000:] # In case the trace is longer than 1000
        This_Main[:CombTheta.shape[0],1] = CombTheta[-1000:]
        This_Main[:CombPhi  .shape[0],2] = CombPhi  [-1000:]

        # The end of Main is now zeros, need to move these zeros to the front
        This_Main = torch.roll(This_Main,shifts=1000-CombTrace.shape[0],dims=0)

        # Get Truth and Rec
        Gen_SDPTheta = Event.get_value('Gen_SDPTheta')
        Gen_SDPPhi   = Event.get_value('Gen_SDPPhi'  )
        Rec_SDPTheta = Event.get_value('Rec_SDPTheta')
        Rec_SDPPhi   = Event.get_value('Rec_SDPPhi'  )
        # Normalise Phi to be around the central pixel
        # print(Gen_SDPPhi,CentralCombPhi)
        Gen_SDPPhi = Gen_SDPPhi #- CentralCombPhi/180*torch.pi
        Rec_SDPPhi = Rec_SDPPhi #- CentralCombPhi/180*torch.pi
        

        # This_Truth = torch.tensor([torch.cos(Gen_SDPTheta),torch.sin(Gen_SDPPhi)])
        # This_Rec   = torch.tensor([torch.cos(Rec_SDPTheta),torch.sin(Rec_SDPPhi)])
        
        This_Truth = torch.tensor([torch.cos(Gen_SDPTheta),Gen_SDPPhi])
        This_Rec   = torch.tensor([torch.cos(Rec_SDPTheta),Rec_SDPPhi])

        # Get MetaData
        Primary = Event.get_value('Primary'    )
        LogE    = Event.get_value('Gen_LogE'   )
        Xmax    = Event.get_value('Gen_Xmax'   )
        This_Meta = torch.tensor([Primary,LogE,Xmax])
        
        # Get AuxData
        Station_Distance = Event.get_value('Station_Distance')/30000
        # Station_Theta    = Event.get_value('Station_Theta'   )
        # Station_Phi      = Event.get_value('Station_Phi'     )
        Station_Time     = Event.get_value('Station_Time'    )/1000

        # This_AuxData = torch.tensor([Station_Distance,Station_Theta,Station_Phi,Station_Time])
        This_MainSt = torch.tensor([Station_Distance,Station_Time])
        
        Main   .append(This_Main)
        Main_st.append(This_MainSt)
        Truth  .append(This_Truth)
        Rec    .append(This_Rec)
        Meta   .append(This_Meta)
        AuxData.append(torch.tensor([CentralCombPhi]))

    
        if (ProcessingDataset is None) and  (len(Truth)>10000) : break

    Main    = torch.stack(Main   ,dim=0)
    Main_st = torch.stack(Main_st,dim=0)
    Truth   = torch.stack(Truth  ,dim=0)
    Rec     = torch.stack(Rec    ,dim=0)
    Meta    = torch.stack(Meta   ,dim=0)
    AuxData = torch.stack(AuxData,dim=0)

    
    # Renormalise phi
    Truth[:,1][Truth[:,1]<0] += 2*torch.pi 
    Truth[:,1] -= torch.pi/2
    Truth[:,1] -= AuxData.squeeze()/180*torch.pi
    Truth[:,1] = torch.sin(Truth[:,1])

    Rec[:,1][Rec[:,1]<0] += 2*torch.pi
    Rec[:,1] -= torch.pi/2
    Rec[:,1] -= AuxData.squeeze()/180*torch.pi
    Rec[:,1] = torch.sin(Rec[:,1])

    # Slap into Dataset
    if ProcessingDataset is None:
        return (Main,Main_st), AuxData, Truth, Rec, Meta, IDsList
    
    else:
        ProcessingDataset._Main.append(Main) # Must append only main for multiple Mains possibility
        ProcessingDataset._Main.append(Main_st)
        ProcessingDataset._Aux   = AuxData
        ProcessingDataset._Truth = Truth
        ProcessingDataset._Rec   = Rec
        ProcessingDataset._Meta  = Meta

        ProcessingDataset.Unnormalise_Truth = Unnormalise_SDP
        ProcessingDataset.Truth_Keys        = ['SDPTheta','SDPPhi']
        ProcessingDataset.Truth_Units       = ['deg','deg']

        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'