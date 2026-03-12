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




def Graph_Conv3d_Traces_and_RecValues(Dataset,ProcessingDataset):
    '''Will just provide 1 mirror array of pixel traces'''
    IDsList = ()
    Graph = [] # Will have Event in dim1, [Trace,X,Y,PulseStart,RecValues] in dim2, values of dim2 in dim3
    
    for i,Event in enumerate(Dataset):
        print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        Gen_Xmax     = Event.get_value('Gen_Xmax')
        Gen_LogE     = Event.get_value('Gen_LogE')
        Gen_Chi0     = Event.get_value('Gen_Chi0')
        Gen_Rp       = Event.get_value('Gen_Rp'  )
        Gen_SDPTheta = Event.get_value('Gen_SDPTheta')
        Gen_SDPPhi   = Event.get_value('Gen_SDPPhi'  )
        
        # Normalise reconstruction values here
        Gen_Xmax     = (Gen_Xmax - 591    ) / 72
        Gen_LogE     = (Gen_LogE - 16.15)   / 0.475
        Gen_Chi0     = (Gen_Chi0 - 1.65   ) / 0.597
        Gen_Rp       = (Gen_Rp   - 1269   ) / 713
        Gen_SDPTheta = (Gen_SDPTheta - 1.57) / 0.5
        Gen_SDPPhi   = (Gen_SDPPhi   - 0.0 ) / (3.14/2)


        Gen_RecValues = torch.stack((Gen_LogE,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi),dim=0)
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
        # Traces = Traces.clip(min=0)
        # Pstart normalised Here
        if len(Pstart)>0 : Pstart = Pstart - torch.min(Pstart)
        # Append to the GraphData
        Graph.append([Traces,Xs,Ys,Pstart,Gen_RecValues])
        
        
    
    if ProcessingDataset is None:
        return Graph
    ProcessingDataset._Graph = Graph
    ProcessingDataset.GraphData = True
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'    

    
def Unnormalise_FullRecValues(RecValues):
    ''' Function for unnormalisation of the FullRecValues
    But the expectation is that the model is expected to classify the wrongness of these values so return the same thing'''
    return RecValues

def Result_FullRecValues(Dataset, ProcessingDataset):
    ''' Provides a normalised version of all reconstruction variables as main data
    Does not do augmentations
    Variables: LogE, Xmax, Chi0, Rp, SDPTheta, SDPPhi
    Also Does Aux Values: Gen_LogE, Gen_Xmax, Gen_Chi0, Gen_Rp, Gen_SDPTheta, Gen_SDPPhi,
                          Gen_EventClass, Gen_Primary, Gen_CosZenith, Gen_CherenkovFraction

    '''

    IDsList = ()
    Gen_Xmax     = torch.zeros(len(Dataset),1)
    Gen_LogE     = torch.zeros(len(Dataset),1)
    Gen_Chi0     = torch.zeros(len(Dataset),1)
    Gen_Rp       = torch.zeros(len(Dataset),1)
    Gen_SDPTheta = torch.zeros(len(Dataset),1)
    Gen_SDPPhi   = torch.zeros(len(Dataset),1)

    Gen_EventClass = torch.zeros(len(Dataset),1)
    Gen_Primary    = torch.zeros(len(Dataset),1)
    Gen_CosZenith  = torch.zeros(len(Dataset),1)
    Gen_CherFrac   = torch.zeros(len(Dataset),1)


    Rec_Xmax     = torch.zeros(len(Dataset),1)
    Rec_LogE     = torch.zeros(len(Dataset),1)
    Rec_Chi0     = torch.zeros(len(Dataset),1)
    Rec_Rp       = torch.zeros(len(Dataset),1)
    Rec_SDPTheta = torch.zeros(len(Dataset),1)
    Rec_SDPPhi   = torch.zeros(len(Dataset),1)

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values
        Gen_Xmax     [i] = Event.get_value('Gen_Xmax')
        Gen_LogE     [i] = Event.get_value('Gen_LogE')
        Gen_Chi0     [i] = Event.get_value('Gen_Chi0')
        Gen_Rp       [i] = Event.get_value('Gen_Rp')
        Gen_SDPTheta [i] = Event.get_value('Gen_SDPTheta')
        Gen_SDPPhi   [i] = Event.get_value('Gen_SDPPhi')

        Gen_EventClass[i] = Event.get_value('Event_Class')
        Gen_Primary   [i] = Event.get_value('Primary')
        Gen_CosZenith [i] = Event.get_value('Gen_CosZenith')
        Gen_CherFrac  [i] = Event.get_value('Gen_CherenkovFraction')


        Rec_Xmax     [i] = Event.get_value('Rec_Xmax')
        Rec_LogE     [i] = Event.get_value('Rec_LogE')
        Rec_Chi0     [i] = Event.get_value('Rec_Chi0')
        Rec_Rp       [i] = Event.get_value('Rec_Rp')
        Rec_SDPTheta [i] = Event.get_value('Rec_SDPTheta')
        Rec_SDPPhi   [i] = Event.get_value('Rec_SDPPhi')

    # Normalisations - Everything must be normalised to mean 0 and std 1
    # Xmax
    XmaxMean = 591
    XmaxStd  = 72
    Gen_Xmax = (Gen_Xmax - XmaxMean) / XmaxStd
    Rec_Xmax = (Rec_Xmax - XmaxMean) / XmaxStd

    # Energy
    EnergyMean = 16.15
    EnergyStd  = 0.475
    Gen_LogE = (Gen_LogE - EnergyMean) / EnergyStd
    Rec_LogE = (Rec_LogE - EnergyMean) / EnergyStd

    # TODO:  Check Rp, Chi0, SDPTheta, SDPPhi Means and Stds
    # Chi0
    Chi0Mean = 1.65
    Chi0Std  = 0.597
    Gen_Chi0 = (Gen_Chi0 - Chi0Mean) / Chi0Std
    Rec_Chi0 = (Rec_Chi0 - Chi0Mean) / Chi0Std

    # Rp
    RpMean = 1269
    RpStd  = 713
    Gen_Rp = (Gen_Rp - RpMean) / RpStd
    Rec_Rp = (Rec_Rp - RpMean) / RpStd

    # SDPTheta
    SDPThetaMean = 1.57
    SDPThetaStd  = 0.5
    Gen_SDPTheta = (Gen_SDPTheta - SDPThetaMean) / SDPThetaStd
    Rec_SDPTheta = (Rec_SDPTheta - SDPThetaMean) / SDPThetaStd

    # SDPPhi
    SDPPhiMean = 0.0
    SDPPhiStd  = 3.14/2
    Gen_SDPPhi = (Gen_SDPPhi - SDPPhiMean) / SDPPhiStd
    Rec_SDPPhi = (Rec_SDPPhi - SDPPhiMean) / SDPPhiStd

    Gen_Result = torch.stack((Gen_LogE,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi),dim=1).squeeze()
    Rec_Result = torch.stack((Rec_LogE,Rec_Xmax,Rec_Chi0,Rec_Rp,Rec_SDPTheta,Rec_SDPPhi),dim=1).squeeze()

    Aux_Values = torch.stack((Gen_LogE,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi,
                              Gen_EventClass,Gen_Primary,Gen_CosZenith,Gen_CherFrac),dim=1).squeeze()
    
    # print(f'Rec_SDPPhi Shape: {Rec_SDPPhi.shape}, resulting Rec Shape: {Rec_Result.shape}')
    # print(f'Gen_SDPPhi Shape: {Gen_SDPPhi.shape}, resulting Gen Shape: {Gen_Result.shape}')
    
    if ProcessingDataset is None:
        return Gen_Result, Rec_Result
    else:
        ProcessingDataset._Truth = Gen_Result
        ProcessingDataset._Rec   = Rec_Result
        ProcessingDataset.Truth_Keys  = ('LogE','Xmax','Chi0','Rp','SDPTheta','SDPPhi')
        ProcessingDataset.Truth_Units = ('','g/cm^2','rad','m','rad','rad')
        ProcessingDataset.Unnormalise_Truth = Unnormalise_FullRecValues
        ProcessingDataset._Aux = Aux_Values
        ProcessingDataset.Aux_Keys  = ('Gen_LogE','Gen_Xmax','Gen_Chi0','Gen_Rp','Gen_SDPTheta','Gen_SDPPhi',
                                       'Gen_EventClass','Gen_Primary','Gen_CosZenith','Gen_CherenkovFraction')
        ProcessingDataset.Aux_Units = ('','g/cm^2','rad','m','rad','rad',
                                       '','', '','')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'
            


def Aux_FullRecValues(Dataset, ProcessingDataset):
    '''Assuming that the function is used with thte Result_FullRecValues, which i can just put into aux at the same time
    '''
    if ProcessingDataset is None:
        return torch.zeros(len(Dataset),1)
    else:
        pass




# For SDP Predictions

def Main_Conv2d_Grid_Charge_and_Time(Dataset,ProcessingDataset):
    ''' Will just provide 1 mirror array of pixel signals
    Main is a tensor of shape (N,C,20,22) where C is the number of channels
    Selects only the hottest telescope
    '''
    # Has to be done on Event-by-Event basis
    # Preinitialize the tensor
    IDsList = ()
    Main = torch.zeros((len(Dataset),2,20,22))
    if ProcessingDataset is None:
        Main = torch.zeros(10000,2,20,22)
    for i,Event in enumerate(Dataset):
        if i%100 == 0:
            print(f'    Processing Main {i} / {len(Dataset)}',end='\r')
        if i == 10000 and ProcessingDataset is None: break
        # ID Checks
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # There only should exist only one telescope. So no need to check for telescope selection
        PixelIDs      = Event.get_pixel_values('PixelID')
        TelIDs        = Event.get_pixel_values('TelID')
        Charge        = Event.get_pixel_values('Charge')
        PulseCentroid = Event.get_pixel_values('PulseCentroid')
        Status        = Event.get_pixel_values('Status')

        StatusThreshold = 4
        PixelIDs      = PixelIDs     [Status>=StatusThreshold]
        TelIDs        = TelIDs       [Status>=StatusThreshold]
        Charge        = Charge       [Status>=StatusThreshold]
        PulseCentroid = PulseCentroid[Status>=StatusThreshold]
        Status        = Status       [Status>=StatusThreshold]
        if Status.numel() == 0:
            continue
        PulseCentroid_ZeroMask = PulseCentroid != 0
        # Normalise
        Charge = torch.log10(torch.clamp_min(Charge,0)+1)/3.75
        Time   = PulseCentroid[PulseCentroid_ZeroMask]
        Time = Time - torch.min(Time)
        Time = Time / 40

        Xs,Ys         = IndexToXY(PixelIDs-(TelIDs-1)*440+1,return_tensor=True)
        Main[i,0,Xs,Ys] = Charge
        Xs = Xs[PulseCentroid_ZeroMask]
        Ys = Ys[PulseCentroid_ZeroMask]
        Main[i,1,Xs,Ys] = Time

    # Pass the data to the ProcessingDataset
    if ProcessingDataset is None:
        return Main
    ProcessingDataset._Main.append(Main)
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match'


def Result_SDPRecValues(Dataset, ProcessingDataset):
    ''' Provides a normalised version of all reconstruction variables as main data
    Does not do augmentations
    Variables: SDPTheta, SDPPhi
    Also Does Aux Values: Gen_LogE, Gen_Xmax, Gen_Chi0, Gen_Rp, Gen_SDPTheta, Gen_SDPPhi,
                          Gen_EventClass, Gen_Primary, Gen_CosZenith, Gen_CherenkovFraction

    '''

    Offsets = {1:44.45/180*torch.pi,2:89.87/180*torch.pi,3:132.83/180*torch.pi}#,4:15/180*torch.pi,5:45/180*torch.pi,6:75/180*torch.pi}

    IDsList = ()
    Gen_Xmax     = torch.zeros(len(Dataset),1)
    Gen_LogE     = torch.zeros(len(Dataset),1)
    Gen_Chi0     = torch.zeros(len(Dataset),1)
    Gen_Rp       = torch.zeros(len(Dataset),1)
    Gen_SDPTheta = torch.zeros(len(Dataset),1)
    Gen_SDPPhi   = torch.zeros(len(Dataset),1)

    Gen_EventClass = torch.zeros(len(Dataset),1)
    Gen_Primary    = torch.zeros(len(Dataset),1)
    Gen_CosZenith  = torch.zeros(len(Dataset),1)
    Gen_CherFrac   = torch.zeros(len(Dataset),1)


    Rec_Xmax     = torch.zeros(len(Dataset),1)
    Rec_LogE     = torch.zeros(len(Dataset),1)
    Rec_Chi0     = torch.zeros(len(Dataset),1)
    Rec_Rp       = torch.zeros(len(Dataset),1)
    Rec_SDPTheta = torch.zeros(len(Dataset),1)
    Rec_SDPPhi   = torch.zeros(len(Dataset),1)

    SelectedTelescopes = torch.zeros(len(Dataset),1).int()

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Truth {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        TelIDs = Event.get_pixel_values('TelID')
        SelectedTelescopes[i] = TelIDs.int().bincount().argmax()

        # Get the values
        Gen_Xmax     [i] = Event.get_value('Gen_Xmax')
        Gen_LogE     [i] = Event.get_value('Gen_LogE')
        Gen_Chi0     [i] = Event.get_value('Gen_Chi0')
        Gen_Rp       [i] = Event.get_value('Gen_Rp')
        Gen_SDPTheta [i] = Event.get_value('Gen_SDPTheta')
        Gen_SDPPhi   [i] = Event.get_value('Gen_SDPPhi')

        Gen_EventClass[i] = Event.get_value('Event_Class')
        Gen_Primary   [i] = Event.get_value('Primary')
        Gen_CosZenith [i] = Event.get_value('Gen_CosZenith')
        Gen_CherFrac  [i] = Event.get_value('Gen_CherenkovFraction')


        Rec_Xmax     [i] = Event.get_value('Rec_Xmax')
        Rec_LogE     [i] = Event.get_value('Rec_LogE')
        Rec_Chi0     [i] = Event.get_value('Rec_Chi0')
        Rec_Rp       [i] = Event.get_value('Rec_Rp')
        Rec_SDPTheta [i] = Event.get_value('Rec_SDPTheta')
        Rec_SDPPhi   [i] = Event.get_value('Rec_SDPPhi')


    # Make Aux Values before the Normalisations
    Aux_Values = torch.stack((Gen_LogE,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi,
                              Gen_EventClass,Gen_Primary,Gen_CosZenith,Gen_CherFrac),dim=1).squeeze()
    

    # SDPTheta - no normalisation, already should be in radians

    # SDPPhi
    # Gen_SDPPhi = Gen_SDPPhi+ 2*torch.pi*(Gen_SDPPhi<0)
    # Gen_SDPPhi -= torch.pi
    # Rec_SDPPhi = Rec_SDPPhi+ 2*torch.pi*(Rec_SDPPhi<0)
    # Rec_SDPPhi -= torch.pi

    for i in range(1,4):
        print(f'Sum of SelectedTelescopes == {i} is {torch.sum(SelectedTelescopes == i)}')
        Gen_SDPPhi[SelectedTelescopes == i] -= Offsets[i]
        Rec_SDPPhi[SelectedTelescopes == i] -= Offsets[i]

    Gen_Result = torch.stack((Gen_SDPTheta,Gen_SDPPhi),dim=1).squeeze()
    Rec_Result = torch.stack((Rec_SDPTheta,Rec_SDPPhi),dim=1).squeeze()

    
    if ProcessingDataset is None:
        return Gen_Result, Rec_Result
    else:
        ProcessingDataset._Main.append(Gen_Result)
        ProcessingDataset._Truth = Gen_Result
        ProcessingDataset._Rec   = Rec_Result
        ProcessingDataset.Truth_Keys  = ('SDPTheta','SDPPhi')
        ProcessingDataset.Truth_Units = ('rad','rad')
        ProcessingDataset.Unnormalise_Truth = Unnormalise_FullRecValues
        ProcessingDataset._Aux = Aux_Values
        ProcessingDataset.Aux_Keys  = ('Gen_LogE','Gen_Xmax','Gen_Chi0','Gen_Rp','Gen_SDPTheta','Gen_SDPPhi',
                                       'Gen_EventClass','Gen_Primary','Gen_CosZenith','Gen_CherenkovFraction')
        ProcessingDataset.Aux_Units = ('','g/cm^2','rad','m','rad','rad',
                                       '','', '','')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'
            
