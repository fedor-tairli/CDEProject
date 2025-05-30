###########################################################
#                For The TraceHexConv                     #
#                Read RawData from ADST S                 #
#                 Normalise on the spot                   #
#                  Store in TorchTensors                  #
###########################################################

import numpy as np
import paths
import os
import torch
import gc
os.system('clear')

# import pickle
import pprint 
pp = pprint.PrettyPrinter().pprint

from adst3 import RecEventProvider, GetDetectorGeometry


######### SETUP #########

def clearout():
    os.system('clear')
clearout()

# Initialise some values paths mainly
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
Paths.RawADSTs = Paths.data_path + 'RawADSTs'
Paths.RawData = f'{Paths.data_path}RawData'
Paths.NormData = f'{Paths.data_path}NormData'

if not os.path.exists(Paths.RawData):
    os.system(f'mkdir -p {Paths.RawData}')

if not os.path.exists(Paths.NormData):
    os.system(f'mkdir -p {Paths.NormData}')


# find list of files to go over: 
TP = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
AP = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
mass   = ['helium','iron','oxygen','proton'] #,'photon']
run = ['Run010','Run030','Run080','Run090']
# For Quad Read : 
run = ['Run010']
size =1/4


print('Initialising Files array')

files = []
# Iterate over the files
for e in energy:
    # the low energy files are on andromeda
    if e in ['18.0_18.5','18.5_19.0']:
        Path = AP
    else:
        Path = TP
    # Construct the filenames array
    for m in mass:
        for r in run:
            sub_e = e.replace('.','')
            filename = f'{Path}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{r}.root'
            files.append(filename)
# pp(files)

# Conditions and ...

testfile = False
StopEarly = False
SaveDense = False

if SaveDense:
    print('Warning: Will attempt to save Dense Tensor.')
if testfile:
    files = ['/remote/tychodata/ftairli/data/EG_File.root']
    

# Iterate over the files
theta = np.pi/2  #90 degree rotation
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0],
                            [0            ,  0            , 1]])

def rotate_point(point):
    return np.dot(rotation_matrix, point)

def norm_Energy(E):
    E_mean = 19
    return np.log10(E+1)-E_mean

def norm_Xmax(X):
    X_mean = 750
    X_std  = 66.80484050442804 
    return (X-X_mean)/X_std

def norm_Dist(X):
    return X/750    

def norm_trace(trace,charge,peak):
    out = np.array(trace)*peak/charge
    out = np.log10(out+1)/np.log10(101) # Calibrate to 100 vem gives unity
    return out


def Get_event_string(filename):
    try:
        info = filename.split('/')[-1].split('.')[0].split('_')
        IDstr = ['0','0','0']
        # Energy
        if '180' == info[1]:
            IDstr[0]='1'
        elif '185' == info[1]:
            IDstr[0]='2'
        elif '190' == info[1]:
            IDstr[0]='3'
        elif '195' == info[1]:
            IDstr[0]='4'
        else:
            IDstr[0]='5'
        # Mass
        if 'proton' == info[3]:
            IDstr[1]='1'
        elif 'helium' == info[3]:
            IDstr[1]='2'
        elif 'oxygen' == info[3]:
            IDstr[1]='3'
        elif 'iron' == info[3]:
            IDstr[1]='4'
        else:
            IDstr[1]='5'

        # Run
        if 'Run010' == info[6]:
            IDstr[2]='1'
        elif 'Run030' == info[6]:
            IDstr[2]='2'
        elif 'Run080' == info[6]:
            IDstr[2]='3'
        elif 'Run090' == info[6]:
            IDstr[2]='4'
        else:
            IDstr[2]='5'
        return ''.join(IDstr)
    except:
        return '999'

print('Initialising Arrays')

# Input data arrays
size = size*1e6
size = int(size)
# Define the expected size of the allocated array -> expect <8 tanks per event will cut later
expeced_tanks = int(8)
size = int(size*expeced_tanks) # 360 = 3*120 expected values per tank

# Targets
Aux   = np.zeros((size,4),dtype=np.float32)-999
Main  = np.zeros((size,3,120),dtype=np.float32)-999
Truth = np.zeros((size,120),dtype=np.float32)-999


# MetaData
# EventId = np.zeros((size),dtype=str)
EventMyId = np.zeros((size),dtype=int)

print('Begining to read files')
Expected_Events = size/expeced_tanks
Nevents = -1       # Number of events read
Ntanks  = 0        # Number of tanks read

for filename in files:
    Evstr = Get_event_string(filename)
    detGeom = GetDetectorGeometry(filename) # in case its different per file, its same per event for sure
    for i,ev in enumerate(RecEventProvider(filename,0)):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
        Nevents += 1

        print(f'\rCurrent Progress: {Nevents/Expected_Events:.5f}, Total Tanks: {Ntanks}', end='')
        
        # SubBranches
        GenShower = ev.GetGenShower()
        stations = ev.GetSDEvent().GetStationVector()

        # Event Observables
        logE   = norm_Energy(GenShower.GetEnergy())
        Xmax   = norm_Xmax(GenShower.GetXmaxInterpolated())
        CosZen = GenShower.GetCosZenith()

        Axis = GenShower.GetAxisSiteCS()
        Core = GenShower.GetCoreSiteCS()


        Event_Aux_Array = np.array([logE,Xmax,CosZen,-999],dtype=np.float32)

        # Loop over stations
        for station in stations:

            if station.IsDense() or station.IsAccidental(): continue
            

            StationId = station.GetId()
            SignalStart = station.GetSignalStartSlot()

            Event_Aux_Array[3] = norm_Dist(detGeom.GetStationAxisDistance(StationId,Axis,Core))
            
            Aux[Ntanks,:] = Event_Aux_Array

            # get traces
            AvMuonTrace = np.zeros(120,dtype=np.float32)
            for pmtNo in [1,2,3]:
                trace  = station.GetPMTTraces(0,pmtNo).GetVEMComponent() # 0 = total
                SignalEnd = min(SignalStart + 120, len(trace)) # Adjust SignalEnd to not exceed trace length
                trace = np.array(trace[SignalStart:SignalEnd])

                # Pad trace with zeros if shorter than 120
                if len(trace) < 120:
                    trace = np.pad(trace, (0, 120 - len(trace)), 'constant')
                
                
                
                MuonTrace = station.GetPMTTraces(3,pmtNo).GetVEMComponent() # 3 = muon
                MuonSignalEnd = min(SignalStart + 120, len(MuonTrace)) # Adjust SignalEnd to not exceed trace length
                MuonTrace = np.array(MuonTrace[SignalStart:MuonSignalEnd])
                if len(MuonTrace) < 120:
                    MuonTrace = np.pad(MuonTrace, (0, 120 - len(MuonTrace)), 'constant')
                

                charge = station.GetPMTTraces(0,pmtNo).GetCharge()
                peak   = station.GetPMTTraces(0,pmtNo).GetPeak()

                
                Main[Ntanks,pmtNo-1,:] = norm_trace(trace,charge,peak)
                AvMuonTrace += norm_trace(MuonTrace,charge,peak)

            # Average the muon trace
            AvMuonTrace /= 3
            Truth[Ntanks,:] = AvMuonTrace
            EventMyId[Ntanks] = int(f'{Evstr}{i:05d}')
            Ntanks += 1
            




print()
print('Finished Reading, Trimming Arrays')
# Shorten the arrays to the correct size in case the allocation was too much
Main = Main[:Ntanks,:,:]
Aux  = Aux[:Ntanks,:]
Truth = Truth[:Ntanks,:]
EventMyId = EventMyId[:Ntanks]



# Save Indices
torch.manual_seed(1234)
length = len(Aux)

indices = torch.randperm(length)

train_indices = indices[:int(0.8*length)]
val_indices   = indices[int(0.8*length):int(0.9*length)]
test_indices  = indices[int(0.9*length):]

print('Converting to torch tensors, for Targets and Auxillary data')
Main = torch.from_numpy(Main)
Aux  = torch.from_numpy(Aux)
Truth = torch.from_numpy(Truth)
EventMyId = torch.from_numpy(EventMyId)


print('Saving Main')
Main_train = Main[train_indices]
Main_val   = Main[val_indices]
Main_test  = Main[test_indices]

torch.save(Main_train,f'{Paths.NormData}/Main_train.pt')
torch.save(Main_val,f'{Paths.NormData}/Main_val.pt')
torch.save(Main_test,f'{Paths.NormData}/Main_test.pt')

del Main, Main_train, Main_val, Main_test
gc.collect()

print('Saving Aux')
Aux_train = Aux[train_indices]
Aux_val   = Aux[val_indices]
Aux_test  = Aux[test_indices]

torch.save(Aux_train,f'{Paths.NormData}/Aux_train.pt')
torch.save(Aux_val,f'{Paths.NormData}/Aux_val.pt')
torch.save(Aux_test,f'{Paths.NormData}/Aux_test.pt')

del Aux, Aux_train, Aux_val, Aux_test
gc.collect()

print('Saving Truth')
Truth_train = Truth[train_indices]
Truth_val   = Truth[val_indices]
Truth_test  = Truth[test_indices]

torch.save(Truth_train,f'{Paths.NormData}/Truth_train.pt')
torch.save(Truth_val,f'{Paths.NormData}/Truth_val.pt')
torch.save(Truth_test,f'{Paths.NormData}/Truth_test.pt')

del Truth, Truth_train, Truth_val, Truth_test
gc.collect()

print('Saving EventMyId')
EventMyId_train = EventMyId[train_indices]
EventMyId_val   = EventMyId[val_indices]
EventMyId_test  = EventMyId[test_indices]

torch.save(EventMyId_train,f'{Paths.NormData}/EventMyId_train.pt')
torch.save(EventMyId_val,f'{Paths.NormData}/EventMyId_val.pt')
torch.save(EventMyId_test,f'{Paths.NormData}/EventMyId_test.pt')

del EventMyId, EventMyId_train, EventMyId_val, EventMyId_test
gc.collect()

print('Done')
