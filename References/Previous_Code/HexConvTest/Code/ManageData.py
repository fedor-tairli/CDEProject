##########################################
#         Take Data from pandas          #
#         Normalise and (Not)split       #
#         Store in torch tensors         #
##########################################





import torch
import numpy as np
import pandas as pd
import os
import paths

os.system('clear')
# from matplotlib import pyplot as plt

# Prepare the paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# print(dir(Paths))
# Paths.RawData = Paths.data_path + 'RawData/'
Paths.NormData = Paths.data_path + 'NormData/'

# Paths.PrintVariables(values=True)

if not os.path.exists(Paths.NormData):
    os.makedirs(Paths.NormData)





# find the array index
def valid_indices(X,Y):
    # Calculate the indices
    X_indices = np.round(X / np.cos(np.pi/6) / 2 + 5).astype(int)
    Y_indices = np.round(Y).astype(int)
    Y_indices[X_indices % 2 == 0] -= 1
    Y_indices = Y_indices / 2 + 5

    # Create boolean masks for valid indices
    valid_X = (X_indices >= 0) & (X_indices < 11)
    valid_Y = (Y_indices >= 0) & (Y_indices < 11)

    # Use masks to filter out invalid indices
    valid_X_indices = X_indices[valid_X & valid_Y].astype(int)
    valid_Y_indices = Y_indices[valid_X & valid_Y].astype(int)

    return valid_X_indices, valid_Y_indices, valid_X & valid_Y


# Read data 

mass   = ['helium','iron','oxygen','proton']
energy = ['180_185','185_190','190_195','195_200']
# Run    = ['Run010','Run030','Run080','Run090']
Run    = 'Run090'

Event = pd.DataFrame()
Stations =pd.DataFrame()

print(f'Reading Data in {Run}')
myEventId = 0
for m in mass:
    for e in energy:
        # Construct Names
        fname_event = f'{Paths.RawData}/SIB23c_{e}_{m}_Hybrid_CORSIKA76400_{Run}_Event.pt'
        fname_stations = f'{Paths.RawData}/SIB23c_{e}_{m}_Hybrid_CORSIKA76400_{Run}_Stations.pt'
        # ReadData 
        SingleEvent = pd.read_pickle(fname_event)
        SingleStations = pd.read_pickle(fname_stations)
        
        # Adjust the EventId
        for id in SingleEvent.EventId:
            print(f'\rReading Event : {myEventId}',end='')
            SingleEvent.loc[SingleEvent.EventId == id,'EventId'] = myEventId
            SingleStations.loc[SingleStations.EventId == id,'EventId'] = myEventId
            myEventId += 1
            # if myEventId > 1000:
            #     break
        # Append to the main dataframes
        Event    = pd.concat([Event,SingleEvent])
        Stations = pd.concat([Stations,SingleStations])
    
        # if myEventId > 1000:
        #     break
    # if myEventId > 1000:
    #     break
print()



# Delete events without Stations
common_EventId = set(Event['EventId']) & set(Stations['EventId']) # & set(p_EyesData['EventId'])
Event = Event[Event['EventId'].isin(common_EventId)]
Stations = Stations[Stations['EventId'].isin(common_EventId)]

total_events = len(common_EventId)

print('Total Number of events : ',total_events)

# initialise the global torch tensors for X,Y and then split them into train,val and test

# globX = total_events,3 channels(signal,times,states),11x11 grid
globX = torch.zeros((total_events, 3, 11, 11), dtype=torch.float)

# globY = total_events 4 vectors (E,Xmax,Core,Axis)

globY_E = torch.zeros((total_events, 1), dtype=torch.float)
globY_Xmax = torch.zeros((total_events, 1), dtype=torch.float)
globY_Core = torch.zeros((total_events, 2), dtype=torch.float)
globY_Axis = torch.zeros((total_events, 3), dtype=torch.float)
glob_id    = torch.zeros((total_events, 1), dtype=torch.float)
print('Tensors Allocated')




# iterate over events and fill the tensors
# Do preliminary calculations and normalisations using vectorised numpy for speed

# GLOBAL VALUES
Xmax_MEAN     = 750.0
Xmax_STD      = 66.80484050442804 
GlobalTimeSTD = 4094.8664907986326 # dont need to include seconds, as they are always same for simulations
E_MEAN        = 19.0
Norm_LEN      = 750.0
Norm_SIG      = np.log10(100+1) # Normalise for unity at 100 vem + for log10
# Rotate Everything to accomodate for the hexagDLy grid
theta = np.pi/2  #90 degree rotation

rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0],
                            [0            ,  0            , 1]])

def rotate_point(point):
    return np.dot(rotation_matrix, point)
Stations['Position'] = Stations.Position.apply(rotate_point)
Event['GenCoreSiteCS'] = Event.GenCoreSiteCS.apply(rotate_point)
Event['GenAxisSiteCS'] = Event.GenAxisSiteCS.apply(rotate_point)
globY_E[:,:]    = torch.from_numpy(np.stack(np.log10(Event['GenPrimaryEnergy'])-E_MEAN )).unsqueeze(1)
globY_Xmax[:,:] = torch.from_numpy((np.stack(Event['GenXmax'])-Xmax_MEAN)/Xmax_STD).unsqueeze(1) 
globY_Axis[:,:] = torch.from_numpy(np.stack(Event['GenAxisSiteCS'])) # already normalised




print('Rotated and Normalised')

print('Begining to fill tensors')
Nev = -1

for i,id in enumerate(common_EventId):
    
    Nev += 1
    print(f'\rProcessed {Nev} events out of {total_events}',end='')
    # Get Stations
    OneStations = Stations[Stations['EventId']==id]
    # Shift positions to zero
    # Normalise time for Simulations , real data needs to include Seconds
    OneStations = OneStations.assign(NormTime=(OneStations['TimeNSecond'] - OneStations['TimeNSecond'].iloc[0])/ GlobalTimeSTD)
    # print('Number of tanks : ',OneStations.shape[0])
    globY_Core[i,:] = torch.from_numpy((Event[Event['EventId']==id]['GenCoreSiteCS'].values[0] -OneStations.Position.iloc[0])/Norm_LEN)[:2]

    OneStations.loc[:,'Position'] = OneStations['Position'].apply(lambda x: (x - OneStations['Position'].iloc[0])/Norm_LEN)
    # Fill the X Tensor 0 - Signal, 1 - Time, 2 - State

    # Get valid indices
    X_positions = OneStations['Position'].apply(lambda x: x[0]).to_numpy()
    Y_positions = OneStations['Position'].apply(lambda x: x[1]).to_numpy()
    X_indices, Y_indices, valid_mask = valid_indices(X_positions, Y_positions)

    # Convert valid indices and mask to torch tensors
    X_indices_torch = torch.from_numpy(X_indices).long()
    Y_indices_torch = torch.from_numpy(Y_indices).long()
    valid_mask_torch = torch.from_numpy(valid_mask)

    # Fill the values
    globX[i,0,X_indices_torch,Y_indices_torch] = torch.tensor((OneStations['Signal'][valid_mask].apply(lambda x: np.log10(x+1)/Norm_SIG)).to_numpy(),dtype=torch.float)
    globX[i,1,X_indices_torch,Y_indices_torch] = torch.tensor(OneStations['NormTime'][valid_mask].to_numpy(),dtype=torch.float)
    globX[i,2,X_indices_torch,Y_indices_torch] = 1 # OneStations.State.iloc[j] # 1 for now, as we don't have the state information

    # print('______________________________________________________________________________')

    # Testing 
    # if Nev == 1000:
    #     break
print()


print('Splitting the data into train,val and test')
# # Testing
# total_events = 1000

# Save the data to be split later
print('Saving Data')

torch.save(globX, f'{Paths.NormData}/{Run}_X.pt')
torch.save(globY_E, f'{Paths.NormData}/{Run}_Y_E.pt')
torch.save(globY_Xmax, f'{Paths.NormData}/{Run}_Y_Xmax.pt')
torch.save(globY_Core, f'{Paths.NormData}/{Run}_Y_Core.pt')
torch.save(globY_Axis, f'{Paths.NormData}/{Run}_Y_Axis.pt')


print('Finished')

