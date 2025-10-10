import torch
import pickle
# from Dataset2 import DatasetContainer, ProcessingDataset # No actual need as i am not using them explicitly


# Once Defined, the functions are to be unchanged
########################################################################################################################


def Main_Hillas_Parameters(Dataset,ProcessingDataset):
    '''Dataset should be a list of filenames'''
    IDsList = ()
    All_H_Amplitude       = []
    All_H_Distance        = []
    All_H_Width           = []
    All_H_Length          = []
    All_H_Skewness        = []
    All_H_Kurtosis        = []
    All_H_GOF_Profile     = []
    All_H_Grad_Profile    = []
    All_H_GOF_Time_Major  = []
    All_H_Grad_Time_Major = []
    All_H_GOF_Time_Minor  = []
    All_H_Grad_Time_Minor = []
    All_H_Npix            = []
    All_H_alpha           = []

    for file in Dataset:
        with open(file,'rb') as f:
            # print(f'Opening file {file}')
            Data = pickle.load(f)
        for i,Event in enumerate(Data):
            
            if not 'HillasValues' in Event:
                continue
                # print(Event.keys())
                # raise ValueError('HillasValues have not been calculated')
        
            if i%100 == 0: print(f'    Processing Main {i} / {len(Data)} with {len(IDsList)} total Events',end='\r')
            ID = Event['Batch']+Event['Shower']    
            IDsList += (ID,)
        
            HillasValues = Event['HillasValues']

            All_H_Amplitude      .append(HillasValues['H_Amplitude']      )
            All_H_Distance       .append(HillasValues['H_Distance']       )
            All_H_Width          .append(HillasValues['H_Width']          )
            All_H_Length         .append(HillasValues['H_Length']         )
            All_H_Skewness       .append(HillasValues['H_Skewness']       )
            All_H_Kurtosis       .append(HillasValues['H_Kurtosis']       )
            All_H_GOF_Profile    .append(HillasValues['H_GOF_Profile']    )
            All_H_Grad_Profile   .append(HillasValues['H_Grad_Profile']   )
            All_H_GOF_Time_Major .append(HillasValues['H_GOF_Time_Major'] )
            All_H_Grad_Time_Major.append(HillasValues['H_Grad_Time_Major'])
            All_H_GOF_Time_Minor .append(HillasValues['H_GOF_Time_Minor'] )
            All_H_Grad_Time_Minor.append(HillasValues['H_Grad_Time_Minor'])
            All_H_Npix           .append(HillasValues['H_Npix']           )
            All_H_alpha          .append(HillasValues['H_alpha']          )
    
    All_H_Amplitude       = torch.tensor(All_H_Amplitude       ).unsqueeze(1)
    All_H_Distance        = torch.tensor(All_H_Distance        ).unsqueeze(1)
    All_H_Width           = torch.tensor(All_H_Width           ).unsqueeze(1)
    All_H_Length          = torch.tensor(All_H_Length          ).unsqueeze(1)
    All_H_Skewness        = torch.tensor(All_H_Skewness        ).unsqueeze(1)
    All_H_Kurtosis        = torch.tensor(All_H_Kurtosis        ).unsqueeze(1)
    All_H_GOF_Profile     = torch.tensor(All_H_GOF_Profile     ).unsqueeze(1)
    All_H_Grad_Profile    = torch.tensor(All_H_Grad_Profile    ).unsqueeze(1)
    All_H_GOF_Time_Major  = torch.tensor(All_H_GOF_Time_Major  ).unsqueeze(1)
    All_H_Grad_Time_Major = torch.tensor(All_H_Grad_Time_Major ).unsqueeze(1)
    All_H_GOF_Time_Minor  = torch.tensor(All_H_GOF_Time_Minor  ).unsqueeze(1)
    All_H_Grad_Time_Minor = torch.tensor(All_H_Grad_Time_Minor ).unsqueeze(1)
    All_H_Npix            = torch.tensor(All_H_Npix            ).unsqueeze(1)
    All_H_alpha           = torch.tensor(All_H_alpha           ).unsqueeze(1)

    All_H_Amplitude       = torch.log10(All_H_Amplitude + 1)  # Log scale the amplitude
    All_H_GOF_Profile     = torch.log10(All_H_GOF_Profile + 1)  # Log scale the GOF
    
    All_H_GOF_Time_Major  = torch.clip(torch.abs(All_H_GOF_Time_Major), 0, 1)  # Clip the GOF to avoid inf values
    All_H_Grad_Time_Major = torch.clip(torch.abs(All_H_Grad_Time_Major), 0, 10)  # Clip the gradient to avoid inf values
    All_H_GOF_Time_Minor  = torch.clip(torch.abs(All_H_GOF_Time_Minor), 0, 10)  # Clip the GOF to avoid inf values
    All_H_Grad_Time_Minor = torch.clip(torch.abs(All_H_Grad_Time_Minor), 0, 10)  # Clip the gradient to avoid inf values

    if ProcessingDataset is None:
        return torch.cat((All_H_Amplitude,All_H_Distance,All_H_Width,All_H_Length,All_H_Skewness,All_H_Kurtosis,All_H_GOF_Profile,All_H_Grad_Profile,All_H_GOF_Time_Major,All_H_Grad_Time_Major,All_H_GOF_Time_Minor,All_H_Grad_Time_Minor,All_H_Npix,All_H_alpha),dim=1)
    ProcessingDataset._Main.append(torch.cat((All_H_Amplitude,All_H_Distance,All_H_Width,All_H_Length,All_H_Skewness,All_H_Kurtosis,All_H_GOF_Profile,All_H_Grad_Profile,All_H_GOF_Time_Major,All_H_Grad_Time_Major,All_H_GOF_Time_Minor,All_H_Grad_Time_Minor,All_H_Npix,All_H_alpha),dim=1))
    ProcessingDataset.GraphData = False
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'
    


def Main_Hillas_Parameters_and_Geometry(Dataset,ProcessingDataset):
    '''Dataset should be a list of filenames'''
    IDsList = ()
    All_H_Amplitude       = []
    All_H_Distance        = []
    All_H_Width           = []
    All_H_Length          = []
    All_H_Skewness        = []
    All_H_Kurtosis        = []
    All_H_GOF_Profile     = []
    All_H_Grad_Profile    = []
    All_H_GOF_Time_Major  = []
    All_H_Grad_Time_Major = []
    All_H_GOF_Time_Minor  = []
    All_H_Grad_Time_Minor = []
    All_H_Npix            = []
    All_H_alpha           = []
    All_SDP_Theta      = []
    All_SDP_Phi        = []
    All_Chi0           = []
    All_Rp             = []

    for file in Dataset:
        with open(file,'rb') as f:
            # print(f'Opening file {file}')
            Data = pickle.load(f)
        for i,Event in enumerate(Data):
            
            if not 'HillasValues' in Event:
                continue
                # print(Event.keys())
                # raise ValueError('HillasValues have not been calculated')
        
            if i%100 == 0: print(f'    Processing Main {i} / {len(Data)} with {len(IDsList)} total Events',end='\r')
            ID = Event['Batch']+Event['Shower']    
            IDsList += (ID,)
        
            HillasValues = Event['HillasValues']

            All_H_Amplitude      .append(HillasValues['H_Amplitude']      )
            All_H_Distance       .append(HillasValues['H_Distance']       )
            All_H_Width          .append(HillasValues['H_Width']          )
            All_H_Length         .append(HillasValues['H_Length']         )
            All_H_Skewness       .append(HillasValues['H_Skewness']       )
            All_H_Kurtosis       .append(HillasValues['H_Kurtosis']       )
            All_H_GOF_Profile    .append(HillasValues['H_GOF_Profile']    )
            All_H_Grad_Profile   .append(HillasValues['H_Grad_Profile']   )
            All_H_GOF_Time_Major .append(HillasValues['H_GOF_Time_Major'] )
            All_H_Grad_Time_Major.append(HillasValues['H_Grad_Time_Major'])
            All_H_GOF_Time_Minor .append(HillasValues['H_GOF_Time_Minor'] )
            All_H_Grad_Time_Minor.append(HillasValues['H_Grad_Time_Minor'])
            All_H_Npix           .append(HillasValues['H_Npix']           )
            All_H_alpha          .append(HillasValues['H_alpha']          )

            All_SDP_Theta      .append(Event['Gen_SDPTheta']      )
            All_SDP_Phi        .append(Event['Gen_SDPPhi']        )
            All_Chi0           .append(Event['Gen_Chi0']           )
            All_Rp             .append(Event['Gen_Rp']             )

    
    All_H_Amplitude       = torch.tensor(All_H_Amplitude       ).unsqueeze(1)
    All_H_Distance        = torch.tensor(All_H_Distance        ).unsqueeze(1)
    All_H_Width           = torch.tensor(All_H_Width           ).unsqueeze(1)
    All_H_Length          = torch.tensor(All_H_Length          ).unsqueeze(1)
    All_H_Skewness        = torch.tensor(All_H_Skewness        ).unsqueeze(1)
    All_H_Kurtosis        = torch.tensor(All_H_Kurtosis        ).unsqueeze(1)
    All_H_GOF_Profile     = torch.tensor(All_H_GOF_Profile     ).unsqueeze(1)
    All_H_Grad_Profile    = torch.tensor(All_H_Grad_Profile    ).unsqueeze(1)
    All_H_GOF_Time_Major  = torch.tensor(All_H_GOF_Time_Major  ).unsqueeze(1)
    All_H_Grad_Time_Major = torch.tensor(All_H_Grad_Time_Major ).unsqueeze(1)
    All_H_GOF_Time_Minor  = torch.tensor(All_H_GOF_Time_Minor  ).unsqueeze(1)
    All_H_Grad_Time_Minor = torch.tensor(All_H_Grad_Time_Minor ).unsqueeze(1)
    All_H_Npix            = torch.tensor(All_H_Npix            ).unsqueeze(1)
    All_H_alpha           = torch.tensor(All_H_alpha           ).unsqueeze(1)

    All_SDP_Theta      = torch.tensor(All_SDP_Theta      ).unsqueeze(1)
    All_SDP_Phi        = torch.tensor(All_SDP_Phi        ).unsqueeze(1)
    All_Chi0           = torch.tensor(All_Chi0           ).unsqueeze(1)
    All_Rp             = torch.tensor(All_Rp             ).unsqueeze(1)



    All_H_Amplitude       = torch.log10(All_H_Amplitude + 1)  # Log scale the amplitude
    All_H_GOF_Profile     = torch.log10(All_H_GOF_Profile + 1)  # Log scale the GOF
    
    All_H_GOF_Time_Major  = torch.clip(torch.abs(All_H_GOF_Time_Major), 0, 1)  # Clip the GOF to avoid inf values
    All_H_Grad_Time_Major = torch.clip(torch.abs(All_H_Grad_Time_Major), 0, 10)  # Clip the gradient to avoid inf values
    All_H_GOF_Time_Minor  = torch.clip(torch.abs(All_H_GOF_Time_Minor), 0, 10)  # Clip the GOF to avoid inf values
    All_H_Grad_Time_Minor = torch.clip(torch.abs(All_H_Grad_Time_Minor), 0, 10)  # Clip the gradient to avoid inf values


    All_Rp = torch.log10(All_Rp + 1)  # Log scale the Rp

    if ProcessingDataset is None:
        return torch.cat((All_H_Amplitude,All_H_Distance,All_H_Width,All_H_Length,All_H_Skewness,All_H_Kurtosis,All_H_GOF_Profile,All_H_Grad_Profile,All_H_GOF_Time_Major,All_H_Grad_Time_Major,All_H_GOF_Time_Minor,All_H_Grad_Time_Minor,All_H_Npix,All_H_alpha,All_SDP_Theta,All_SDP_Phi,All_Chi0,All_Rp),dim=1)
    ProcessingDataset._Main.append(torch.cat((All_H_Amplitude,All_H_Distance,All_H_Width,All_H_Length,All_H_Skewness,All_H_Kurtosis,All_H_GOF_Profile,All_H_Grad_Profile,All_H_GOF_Time_Major,All_H_Grad_Time_Major,All_H_GOF_Time_Minor,All_H_Grad_Time_Minor,All_H_Npix,All_H_alpha,All_SDP_Theta,All_SDP_Phi,All_Chi0,All_Rp),dim=1))
    ProcessingDataset.GraphData = False
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'


def Unnormalise_XmaxEnergyCF(XmaxEnergy):
    '''Will unnormalise Xmax and Energy'''
    # Normalise Xmax
    XmaxMean = 591
    XmaxStd  = 72
    XmaxEnergy[:,0] = XmaxEnergy[:,0] * XmaxStd + XmaxMean

    # Normalise Energy
    EnergyMean = 16.15
    EnergyStd  = 0.475
    XmaxEnergy[:,1] = XmaxEnergy[:,1] * EnergyStd + EnergyMean

    # Cherenkov Fraction - nothing

    return XmaxEnergy

def Truth_XmaxEnergy(Dataset,ProcessingDataset):
    '''Will provide Xmax and Energy for each event'''
    IDsList = ()
    Gen_Xmax   = []
    Gen_Energy = []
    Gen_CF     = []
    
    for file in Dataset:
        
        with open(file,'rb') as f:
            Data = pickle.load(f)
        for i,Event in enumerate(Data):
            if not 'HillasValues' in Event:
                continue
            # ID Checks
            ID = Event['Batch']+Event['Shower']
            IDsList += (ID,)
            if i%100 == 0:print(f'    Processing Truth {i} / {len(Data)} with {len(IDsList)} total Events',end='\r')
            
            
            Gen_Xmax  .append(torch.tensor(Event['Gen_Xmax']))
            Gen_Energy.append(torch.tensor(Event['Gen_LogE']))
            Gen_CF    . append(torch.tensor(Event['Gen_CherenkovFraction']))

    
    Gen_Xmax   = torch.stack(Gen_Xmax)
    Gen_Energy = torch.stack(Gen_Energy)
    Gen_CF     = torch.stack(Gen_CF)
    Rec_Xmax   = torch.zeros_like(Gen_Xmax)
    Rec_Energy = torch.zeros_like(Gen_Energy)
    Rec_CF     = torch.zeros_like(Gen_CF)
    

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

    # Normalise Cherenkov Fraction
    Gen_CF = Gen_CF / 100

    if ProcessingDataset is None:
        return torch.stack((Gen_Xmax,Gen_Energy,Gen_CF),dim =1) , torch.stack((Rec_Xmax,Rec_Energy,Rec_CF),dim =1)
    ProcessingDataset._Truth = torch.stack((Gen_Xmax,Gen_Energy,Gen_CF),dim =1)
    ProcessingDataset._Rec   = torch.stack((Rec_Xmax,Rec_Energy,Rec_CF),dim =1)

    ProcessingDataset.Unnormalise_Truth = Unnormalise_XmaxEnergyCF
    ProcessingDataset.Truth_Keys  = ('Xmax','LogE','CherenkovFraction')
    ProcessingDataset.Truth_Units = ('g/cm^2','','')
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else:
        assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'


from collections import defaultdict

int_to_event_class= {
    0: "'Shower Candidate'",
    1: "'Close Shower'",
    2: "'Horizontal Shower'",
    3: "'Large Event'",
    4: "'Muon + Noise'",
    5: "'Long Muon'",
    6: "'Noise'",
    7: "'Muon'",
    8: "'Empty",
}

int_to_event_class = defaultdict(lambda: 'Unknown', int_to_event_class)

event_class_to_int = defaultdict(lambda: -1, {v: k for k, v in int_to_event_class.items()})


def Aux_Descriptors(Dataset, ProcessingDataset):
    ''' Will just provide some event descriptors for dependence inspection
    Values are : 'Event_Class', 'Primary', 'Gen_LogE', 'Gen_CosZenith', 'Gen_Xmax','Gen_Chi0', 'Gen_Rp', etc
    '''

    IDsList = ()
    Event_Class   = []
    Primary       = []
    Gen_LogE      = []
    Gen_CosZenith = []
    Gen_Xmax      = []
    Gen_Chi0      = []
    Gen_Rp        = []
    Gen_SDPTheta  = []
    Gen_SDPPhi    = []
    Gen_CherFrac  = []

    for file in Dataset:
        with open(file,'rb') as f:
            Data = pickle.load(f)
        
        for i, Event in enumerate(Data):
            if not 'HillasValues' in Event:
                continue

            if i%100 ==0: print(f'    Processing Aux {i} / {len(Data)} with {len(IDsList)} total Events',end='\r')
            # ID Checks
            ID = Event['Batch']+Event['Shower']
            IDsList += (ID,)


            # Get the values
            # print(event_class_to_int.keys)
            # print(Event['EventClass'])
            Event_Class   .append(float(event_class_to_int[Event['EventClass']]))
            Primary       .append(Event['Gen_Primary'])
            Gen_LogE      .append(Event['Gen_LogE'])
            Gen_CosZenith .append(Event['Gen_CosZen'])
            Gen_Xmax      .append(Event['Gen_Xmax'])
            Gen_Chi0      .append(Event['Gen_Chi0'])
            Gen_Rp        .append(Event['Gen_Rp'])
            Gen_SDPTheta  .append(Event['Gen_SDPTheta'])
            Gen_SDPPhi    .append(Event['Gen_SDPPhi'])
            Gen_CherFrac  .append(Event['Gen_CherenkovFraction'])

    Event_Class   = torch.tensor(Event_Class  ).unsqueeze(1)
    Primary       = torch.tensor(Primary      ).unsqueeze(1)
    Gen_LogE      = torch.tensor(Gen_LogE     ).unsqueeze(1)
    Gen_CosZenith = torch.tensor(Gen_CosZenith).unsqueeze(1)
    Gen_Xmax      = torch.tensor(Gen_Xmax     ).unsqueeze(1)
    Gen_Chi0      = torch.tensor(Gen_Chi0     ).unsqueeze(1)
    Gen_Rp        = torch.tensor(Gen_Rp       ).unsqueeze(1)
    Gen_SDPTheta  = torch.tensor(Gen_SDPTheta ).unsqueeze(1)
    Gen_SDPPhi    = torch.tensor(Gen_SDPPhi   ).unsqueeze(1)
    Gen_CherFrac  = torch.tensor(Gen_CherFrac ).unsqueeze(1)


    if ProcessingDataset is None:
        return torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi,Gen_CherFrac),dim=1)
    else:
        ProcessingDataset._Aux = torch.stack((Event_Class,Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_SDPTheta,Gen_SDPPhi,Gen_CherFrac),dim=1)
        ProcessingDataset.Aux_Keys = ('Event_Class','Primary','LogE','CosZenith','Xmax','Chi0','Rp','SDPTheta','SDPPhi','CherenkovFraction')
        ProcessingDataset.Aux_Units = ('','','','','g/cm^2','rad','m','rad','rad','%')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'




def Clean_Hillas_Data(Dataset):
    Main = Dataset._Main[0]
    Truth = Dataset._Truth
    Aux = Dataset._Aux
    Rec = Dataset._Rec

    IDsList = Dataset._EventIds

    New_Main = []
    New_Truth = []
    New_Aux = []
    New_Rec = []

    New_IDsList = []

    N_BadEvents = 0 
    for i in range(len(Main)):
        N_BadEvents += 1
        if torch.any(torch.isnan(Main[i])) or torch.any(torch.isinf(Main[i])): continue
        if torch.any(torch.isnan(Truth[i])) or torch.any(torch.isinf(Truth[i])): continue
        # if torch.any(torch.isnan(Aux[i])) or torch.any(torch.isinf(Aux[i])): continue
        # if torch.any(torch.isnan(Rec[i])) or torch.any(torch.isinf(Rec[i])): continue

        New_Main.append(Main[i])
        New_Truth.append(Truth[i])
        New_Aux.append(Aux[i])
        New_Rec.append(Rec[i])
        New_IDsList.append(IDsList[i])
        
        N_BadEvents -= 1
        print(f'    Cleaning Data {i} / {len(Main)} with {len(New_IDsList)} total Events, removed {N_BadEvents} bad events',end='\r')
    print()
    Dataset._Main = [torch.stack(New_Main),]
    Dataset._Truth = torch.stack(New_Truth)
    Dataset._Aux = torch.stack(New_Aux)
    Dataset._Rec = torch.stack(New_Rec)
    Dataset._EventIds = tuple(New_IDsList)



