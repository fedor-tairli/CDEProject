import torch





def Aux_Descriptors(Dataset, ProcessingDataset,skip_HEAT=True):
    ''' Will just provide some event descriptors for dependence inspection
    Values are : 'Primary', 'Gen_LogE', 'Gen_CosZenith', 'Gen_Xmax','Gen_Chi0', 'Gen_Rp'
    '''

    IDsList = ()
    Primary       = []
    Gen_LogE      = []
    Gen_CosZenith = []
    Gen_Xmax      = []
    Gen_Chi0      = []
    Gen_Rp        = []
    Gen_T0        = []

    for i, Event in enumerate(Dataset):
        if i%100 ==0: print(f'    Processing Aux {i} / {len(Dataset)}',end='\r')

        if skip_HEAT and (Event.get_pixel_values('EyeID') == 6).any():
            continue

        ID = (Event.get_value('EventID_1/2').int()*10000 + Event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # Get the values
        Primary      .append(Event.get_value('Primary'))
        Gen_LogE     .append(Event.get_value('Gen_LogE'))
        Gen_CosZenith.append(Event.get_value('Gen_CosZenith'))
        Gen_Xmax     .append(Event.get_value('Gen_Xmax'))
        Gen_Chi0     .append(Event.get_value('Gen_Chi0'))
        Gen_Rp       .append(Event.get_value('Gen_Rp'))
        Gen_T0       .append(Event.get_value('Gen_T0'))

    Primary        = torch.tensor(Primary      ).unsqueeze(1)
    Gen_LogE       = torch.tensor(Gen_LogE     ).unsqueeze(1)
    Gen_CosZenith  = torch.tensor(Gen_CosZenith).unsqueeze(1)
    Gen_Xmax       = torch.tensor(Gen_Xmax     ).unsqueeze(1)
    Gen_Chi0       = torch.tensor(Gen_Chi0     ).unsqueeze(1)
    Gen_Rp         = torch.tensor(Gen_Rp       ).unsqueeze(1)
    Gen_T0         = torch.tensor(Gen_T0       ).unsqueeze(1)

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



def Return_as_is(Truth):
    return Truth

def Unnormalise_TimeFit_Geometry(Truth):
    Truth[:,0] = torch.acos(Truth[:,0])
    Truth[:,1] = Truth[:,1] * 50000 # 50 km
    Truth[:,2] = Truth[:,2] * 1e5 # 1000 ns

    return Truth

def Truth_Rec_Time_Fit(Dataset, ProcessingDataset=None, skip_HEAT=True):
    '''Provides truth data for time fit'''

    IDsList = ()
    Truth_Data = [] # Gen_Chi0, Gen_Rp, Gen_T0
    Rec_Data   = [] # Rec_Chi0, Rec_Rp, Rec_T0

    for i,event in enumerate(Dataset):
        if i%100 == 0: print(f'    Processing event {i}/{len(Dataset)}', end='\r')
        
        if skip_HEAT and (event.get_pixel_values('EyeID') == 6).any():
            continue
        
        ID = (event.get_value('EventID_1/2').int()*10000 + event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        Gen_Chi0 = event.get_value('Gen_Chi0')
        Gen_Rp   = event.get_value('Gen_Rp')
        Gen_T0   = event.get_value('Gen_T0')

        Rec_Chi0 = event.get_value('Rec_Chi0')
        Rec_Rp   = event.get_value('Rec_Rp')
        Rec_T0   = event.get_value('Rec_T0')

        Truth_Data.append(torch.tensor([Gen_Chi0, Gen_Rp, Gen_T0]))
        Rec_Data  .append(torch.tensor([Rec_Chi0, Rec_Rp, Rec_T0]))

    Truth_Data = torch.stack(Truth_Data)
    Rec_Data   = torch.stack(Rec_Data)

    Truth_Data[:,0] = torch.cos(Truth_Data[:,0]) # Normalise Chi0
    Truth_Data[:,1] = Truth_Data[:,1] / 50000 # Normalise Rp to 50 km
    Truth_Data[:,2] = Truth_Data[:,2] / 1e5 # Normalise T0 to 1000 ns

    if ProcessingDataset is None:
        return Truth_Data, Rec_Data
    ProcessingDataset._Truth = Truth_Data
    ProcessingDataset._Rec   = Rec_Data
    ProcessingDataset.Unnormalise_Truth = Unnormalise_TimeFit_Geometry
    ProcessingDataset.Truth_Keys = ['Chi_0', 'Rp', 'T0']
    ProcessingDataset.Truth_Units = ['rad','m','ns']

    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else: 
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match between Graph and Dataset'




def Graph_FlatTimeFit(Dataset, ProcessingDataset=None,Skip_HEAT=True):
    ''' Provides sparse data for pixel time information
    '''

    N_Vals_Stored = 0

    IDsList = ()
    Graph_Data = []

    for i,event in enumerate(Dataset):
        if i%100 == 0: print(f'    Processing event {i}/{len(Dataset)}, N_vals : {N_Vals_Stored/1e6} M ', end='\r')
        
        if Skip_HEAT and (event.get_pixel_values('EyeID') == 6).any():
            continue

        ID = (event.get_value('EventID_1/2').int()*10000 + event.get_value('EventID_2/2').int()%10000).item()
        IDsList += (ID,)

        # traces
        pix_traces      = event.get_trace_values()
        pix_time        = event.get_pixel_values("PulseCentroid")
        # pix_charge      = event.get_pixel_values("Charge")
        pix_chi_i       = event.get_pixel_values("Chi_i")
        pix_pulse_start = event.get_pixel_values("PulseStart")
        pix_pulse_stop  = event.get_pixel_values("PulseStop")
        pix_time_offset = event.get_pixel_values("TimeOffset")
        
        pix_tel_ids = event.get_pixel_values("TelID")
        if Skip_HEAT and (pix_tel_ids > 6).any() : # HEAT + CO Event -  can mask
            mask = pix_tel_ids <= 6
            pix_traces      = pix_traces     [mask]
            pix_time        = pix_time       [mask]
            pix_chi_i       = pix_chi_i      [mask]
            pix_pulse_start = pix_pulse_start[mask]
            pix_pulse_stop  = pix_pulse_stop [mask]
            pix_time_offset = pix_time_offset[mask]
            
        
        pix_main = torch.zeros((181, 2500))

        for i in range(len(pix_time)):
            T_pix_trace = pix_traces[i].numpy()
            T_pix_pulse_start = int(pix_pulse_start[i].item()) + int(pix_time_offset[i].item())
            T_pix_Pulse_stop = int(pix_pulse_stop[i].item())
            T_pix_Chi_i = int(pix_chi_i[i].item() /torch.pi * 180)
            
            pix_main[
                T_pix_Chi_i, T_pix_pulse_start : T_pix_pulse_start + len(T_pix_trace)
            ] += T_pix_trace

        pix_main = torch.log10(torch.clamp(pix_main, 0, torch.inf) + 1) / 5.0 # Norm to 5
        pix_main[0,:] = 0 # Clear Station Row

        # Min Non Zero Time
        if pix_main.nonzero().size(0) <= 0: # I have no idea how -ve can happen . . .
            # No triggered pixels, skip this event
            event_data = {
                'indices': torch.empty((0, 2), dtype=torch.long),
                'values': torch.empty((0,), dtype=torch.float),
                'norm_min_time': 0,
            }
        
        else:
            min_non_zero_time = pix_main.nonzero()[0,1].item()
            
            

            Station_Signal = event.get_value('Station_TotalSignal')
            Station_Time   = event.get_value('Station_Time')

            Station_Signal = torch.log10(torch.clamp(Station_Signal, 1, torch.inf) + 1)
            Station_Signal = Station_Signal / 4 # Normalise to max signal of 10^4 VEM

            Station_Time = int(Station_Time.item()) # Normalise to the first triggered pixel
            if Station_Time < 0: 
                Station_Time = 0
                Station_Signal = 0

            pix_main[0,Station_Time] = Station_Signal

            # Shorten the Time and Chi_i - <1% of data is above
            pix_main = pix_main[:75, min_non_zero_time:1000]
            # Now find non zero entries and their indices
            non_zero_indices = torch.nonzero(pix_main)
            non_zero_values = pix_main[non_zero_indices[:, 0], non_zero_indices[:, 1]]

            event_data = {
                'indices': non_zero_indices,
                'values': non_zero_values,
                'norm_min_time': min_non_zero_time,
            }
            N_Vals_Stored += non_zero_values.numel()
            N_Vals_Stored += non_zero_indices.numel()
            N_Vals_Stored += 1

        Graph_Data.append(event_data)

    if ProcessingDataset is None:
        return Graph_Data
    ProcessingDataset._Graph = Graph_Data
    ProcessingDataset.GraphData = True
    if ProcessingDataset._EventIds is None:
        ProcessingDataset._EventIds = IDsList
    else: 
        assert ProcessingDataset._EventIds == IDsList, 'Event IDs do not match between Graph and Dataset'

    