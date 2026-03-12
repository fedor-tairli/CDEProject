import torch


def Unnormalise_OnePix(Truth):
    Truth[:,0] = torch.acos(Truth[:,0])
    Truth[:,1] = Truth[:,1] * 50 # 50 km # Not m, like original

    return Truth




def Main_OnePixel_TimeFit(Dataset,ProcessingDataset):

    '''Will provide pixels with trace longer than 10 bins and consider them individual events'''

    Traces = []
    Chi_is = []
    Gen_Chi_0s = []
    Gen_Rp_s   = []
    Rec_Chi_0s = []
    Rec_Rp_s   = []

    N_Events = 0
    N_pix    = 0
    N_short  = 0

    for event in Dataset:

        N_Events += 1

        Ev_Gen_Chi0 = event.get_value('Gen_Chi0')
        Ev_Gen_Rp   = event.get_value('Gen_Rp')
        Ev_Rec_Chi0 = event.get_value('Rec_Chi0')
        Ev_Rec_Rp   = event.get_value('Rec_Rp')

        pix_chi_is = event.get_pixel_values('Chi_i')
        pix_pulse_start = event.get_pixel_values('PulseStart')
        pix_pulse_stop  = event.get_pixel_values('PulseStop')
        pix_status = event.get_pixel_values('Status')
        pix_traces = event.get_trace_values()

        pix_durations = pix_pulse_stop - pix_pulse_start
        pix_duration_mask = pix_durations > 10
        pix_status_mask = pix_status == 4

        total_mask = pix_duration_mask & pix_status_mask

        pix_chi_is = pix_chi_is[total_mask]
        pix_traces = pix_traces[total_mask]

        N_pix += len(pix_chi_is)
        N_short += len(pix_duration_mask) - len(pix_chi_is)
        for i in range(len(pix_chi_is)):
            
            Traces.append(pix_traces[i]//torch.max(pix_traces[i])) # Normalise the trace
            Chi_is.append(pix_chi_is[i])
            
            Gen_Chi_0s.append(Ev_Gen_Chi0)
            Rec_Chi_0s.append(Ev_Rec_Chi0)
            Gen_Rp_s.append(Ev_Gen_Rp)
            Rec_Rp_s.append(Ev_Rec_Rp)

        if N_Events % 100 == 0:
            print(f'Processeding Main: {N_Events} events, {N_pix} pixels, {N_short} short traces', end='\r')
            
    
    Traces = torch.stack(Traces) # Add a channel dimension
    Chi_is = torch.tensor(Chi_is) # Add a channel dimension
    
    Gen_Chi_0s = torch.tensor(Gen_Chi_0s)
    Gen_Rp_s   = torch.tensor(Gen_Rp_s)
    Rec_Chi_0s = torch.tensor(Rec_Chi_0s)
    Rec_Rp_s   = torch.tensor(Rec_Rp_s)

    # Normalisations
    Gen_Chi_0s = torch.cos(Gen_Chi_0s)
    Rec_Chi_0s = torch.cos(Rec_Chi_0s)

    Gen_Rp_s = Gen_Rp_s / 50000 # 50 km
    Rec_Rp_s = Rec_Rp_s / 50000 # 50 km

    Truth = torch.stack([Gen_Chi_0s, Gen_Rp_s], dim=1)
    Rec   = torch.stack([Rec_Chi_0s, Rec_Rp_s], dim=1)
    
    # Pass the Data to the Processing Dataset
    if ProcessingDataset is None:
        return [Traces,Chi_is] , Truth, Rec
    

    ProcessingDataset._Main.append(Traces)
    ProcessingDataset._Main.append(Chi_is)
    ProcessingDataset._Truth = Truth
    ProcessingDataset._Rec   = Rec
    ProcessingDataset._Aux = torch.zeros(len(Traces)) # No aux data, just a placeholder

    ProcessingDataset.Unnormalise_Truth = Unnormalise_OnePix
    ProcessingDataset.Truth_Keys = ['Chi0','Rp']
    ProcessingDataset.Truth_Units = ['rad','km']

    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = torch.arange(len(Traces))
    else:
        assert len(ProcessingDataset._EventIds) == len(Traces), 'EventIds length does not match Main data length'





def Aux_Descriptors(Dataset, ProcessingDataset):
    ''' Will just provide some event descriptors for dependence inspection
    Values are : 'Primary', 'Gen_LogE', 'Gen_CosZenith', 'Gen_Xmax','Gen_Chi0', 'Gen_Rp'
    '''

    IDsList = ()
    Primary       = torch.zeros(len(Dataset),1)
    Gen_LogE      = torch.zeros(len(Dataset),1)
    Gen_CosZenith = torch.zeros(len(Dataset),1)
    Gen_Xmax      = torch.zeros(len(Dataset),1)
    Gen_Chi0      = torch.zeros(len(Dataset),1)
    Gen_Rp        = torch.zeros(len(Dataset),1)
    Gen_T0        = torch.zeros(len(Dataset),1)

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Aux {i} / {len(Dataset)}',end='\r')
        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values
        Primary[i]       = Event.get_value('Primary')
        Gen_LogE[i]      = Event.get_value('Gen_LogE')
        Gen_CosZenith[i] = Event.get_value('Gen_CosZenith')
        Gen_Xmax[i]      = Event.get_value('Gen_Xmax')
        Gen_Chi0[i]      = Event.get_value('Gen_Chi0')
        Gen_Rp[i]        = Event.get_value('Gen_Rp')
        Gen_T0[i]        = Event.get_value('Gen_T0')
    
    if ProcessingDataset is None:
        return torch.cat((Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp),dim=1)
    else:
        ProcessingDataset._Aux = torch.cat((Primary,Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_Chi0,Gen_Rp,Gen_T0),dim=1)
        ProcessingDataset.Aux_Keys = ('Primary','LogE','CosZenith','Xmax','Chi0','Rp','T0')
        ProcessingDataset.Aux_Units = ('','','','g/cm^2','rad','m','ns')
        if ProcessingDataset._EventIds is None:
            ProcessingDataset._EventIds = IDsList
        else:
            assert ProcessingDataset._EventIds == IDsList, 'EventIDs do not match'


def Graph_Autoencoder_TimeFit(Dataset, ProcessingDataset=None):
    ''' Provides sparse data for pixel time information
    includes station
    sparse data in form of dictionary
    '''
    IDsList = ()

    Sparse_Data = []

    for i,event in enumerate(Dataset): 
        if i%100 == 0: print(f'    Processing event {i}/{len(Dataset)}', end='\r')
        ID = (event.get_value('EventID_1/2').int()*10000 + event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)


        # traces
        pix_traces = event.get_trace_values()
        # other data
        pix_chiis  = event.get_pixel_values('Chi_i')
        pix_charge = event.get_pixel_values('Charge')
        pix_time   = event.get_pixel_values('PulseCentroid')

        # Cut on triggered pixels
        pix_status = event.get_pixel_values('Status')
        status_mask = pix_status == 4
        pix_traces  = pix_traces[status_mask]
        pix_chiis   = pix_chiis[status_mask]
        pix_charge  = pix_charge[status_mask]
        pix_time    = pix_time[status_mask]


        # Station Info
        station_chii   = event.get_value('Station_Chi_i')
        station_Time   = event.get_value('Station_Time')
        station_signal = event.get_value('Station_TotalSignal')
        
        event_data = {
            'traces': pix_traces,
            'chi_is': pix_chiis,
            'charge': pix_charge,
            'time': pix_time,
            'station_chii': station_chii,
            'station_time': station_Time,
            'station_signal': station_signal
        }
        Sparse_Data.append(event_data)

    if ProcessingDataset is None:
        return Sparse_Data
    ProcessingDataset._Graph = Sparse_Data
    ProcessingDataset.GraphData = True
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else: 
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match between Graph and Dataset'




def Return_as_is(Truth):
    return Truth

def Truth_Rec_Time_Fit(Dataset, ProcessingDataset=None):
    '''Provides truth data for time fit'''

    IDsList = ()
    Truth_Data = torch.zeros((len(Dataset),3)) # Gen_Chi0, Gen_Rp, Gen_T0
    Rec_Data   = torch.zeros((len(Dataset),3)) # Rec_Chi0, Rec_Rp, Rec_T0

    for i,event in enumerate(Dataset):
        if i%100 == 0: print(f'    Processing event {i}/{len(Dataset)}', end='\r')
        ID = (event.get_value('EventID_1/2').int()*10000 + event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        Gen_Chi0 = event.get_value('Gen_Chi0')
        Gen_Rp   = event.get_value('Gen_Rp')
        Gen_T0   = event.get_value('Gen_T0')

        Rec_Chi0 = event.get_value('Rec_Chi0')
        Rec_Rp   = event.get_value('Rec_Rp')
        Rec_T0   = event.get_value('Rec_T0')

        Truth_Data[i] = torch.tensor([Gen_Chi0, Gen_Rp, Gen_T0])
        Rec_Data[i]   = torch.tensor([Rec_Chi0, Rec_Rp, Rec_T0])

    # Normalization is not done, because predictions are made on pixel basis. 

    if ProcessingDataset is None:
        return Truth_Data, Rec_Data
    ProcessingDataset._Truth = Truth_Data
    ProcessingDataset._Rec   = Rec_Data
    ProcessingDataset.Unnormalise_Truth = Return_as_is
    ProcessingDataset.Truth_Keys = ['Chi_0', 'Rp', 'T0']
    ProcessingDataset.Truth_Units = ['rad','m','ns']

    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else: 
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match between Graph and Dataset'
