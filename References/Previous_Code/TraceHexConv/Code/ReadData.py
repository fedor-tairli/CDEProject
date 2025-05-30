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

if not os.path.exists(Paths.RawData):
    os.system(f'mkdir -p {Paths.RawData}')

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

def norm_Axis(A):
    return rotate_point(A)

def norm_core(C,CentralTank):

    norm_distance = 750
    C = np.array(C-CentralTank)/norm_distance
    return(rotate_point(C))
    
def valid_indices(position,CentralTank):
    
    position = rotate_point((position-CentralTank))/750 # Norm Length
    X = np.round(position[0] / np.cos(np.pi/6) / 2 + 5).astype(int)
    Y = np.round(position[1]).astype(int)
    if X % 2 == 0: Y-=1
    Y = Y//2+5
    if X <0 or X>10 or Y<0 or Y>10: # out of bounds
        return None,None
    else:
        return X,Y

def norm_trace(trace,charge,peak):
    out = np.array(trace)*peak/charge
    out = np.log10(out+1)/np.log10(101) # Calibrate to 100 vem gives unity
    return out

def norm_time(time,time_CT):
    return (time-time_CT)/4094.8664907986326  # GlobalTimeSTD

def Get_event_string(filename):
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


PMT=   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
TRC =  [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119,   0,   1,   2,   3,   4,   5,
          6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
         20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
         34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
         62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
         76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
         90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
        104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
        118, 119,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
         12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
         40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
         54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,
         68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
         82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
         96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119]

print('Initialising Arrays')

# Input data arrays
size = size*1e6
size = int(size)
# Define the expected size of the allocated array -> expect <8 tanks per event will cut later
expeced_tanks = int(8)
coords_expected_size= int(size*expeced_tanks*360) # 360 = 3*120 expected values per tank

D_main_values = np.empty(shape = coords_expected_size, dtype=np.float32)
D_main_coords = np.empty(shape = (5,coords_expected_size), dtype=np.int32)

D_aux  = np.zeros((size,11,11),dtype=np.float32)
# Define container to be appended to the D_main_values and coords arrays
D_main_values_container = np.zeros(shape = (360),dtype=np.float32)
D_main_coords_container = np.zeros(shape = (5,360),dtype=np.int32)

# Dont ever need to change these
D_main_coords_container[1] = TRC
D_main_coords_container[2] = PMT
                   

# Targets
logE   = np.zeros((size,1),dtype=np.float32)
Xmax   = np.zeros((size,1),dtype=np.float32)
Axis   = np.zeros((size,3),dtype=np.float32)
Core   = np.zeros((size,2),dtype=np.float32)


# MetaData
# EventId = np.zeros((size),dtype=str)
EventMyId = np.zeros((size),dtype=int)

print('Begining to read files')

Nappends = 0 # Number of appends to the D_main_values and coords arrays
N = -1       # Number of events read
BadEv = 0
for filename in files:
    Evstr = Get_event_string(filename)
    detGeom = GetDetectorGeometry(filename) # in case its different per file, its same per event for sure
    for i,ev in enumerate(RecEventProvider(filename,0)):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
        N += 1
        if N > size: break # in case, shouldnt trigger

        print(f'\rCurrent Event : {N} / {size}   Bad Events : {BadEv}',end='')
        
        # SubBranches
        GenShower = ev.GetGenShower()
        stations = ev.GetSDEvent().GetStationVector()

        # Check for dense stations and I guess Stationless events
        all_Dense = True
        for j, station in enumerate(stations): # Stations come in order ot highest signal, look for real stations only.
            if not (station.IsDense() and station.IsAccidental()):
                all_Dense = False
                position_CT = detGeom.GetStationPosition(stations[j].GetId())
                time_CT     = stations[j].GetTimeNSecond()

                break

        if all_Dense: 
            BadEv += 1   # Count bad Events
            N     -= 1   # Need to undo the addition above so the zero events dont occur
            continue # No tanks in event: skip

        # Loop over stations
        for station in stations:
            if station.IsDense() or station.IsAccidental(): continue
            position = detGeom.GetStationPosition(station.GetId())
            X,Y = valid_indices(position,position_CT)
            if X is None: continue
            SignalStart = station.GetSignalStartSlot()
            time   = station.GetTimeNSecond()
            D_aux[N,X,Y] = norm_time(time,time_CT)

            # get tracesst
            for pmtNo in [1,2,3]:
                trace  = station.GetPMTTraces(0,pmtNo).GetVEMComponent() # 0 = total
                SignalEnd = min(SignalStart + 120, len(trace)) # Adjust SignalEnd to not exceed trace length
                trace = np.array(trace[SignalStart:SignalEnd])

                # Pad trace with zeros if shorter than 120
                if len(trace) < 120:
                    trace = np.pad(trace, (0, 120 - len(trace)), 'constant')

                charge = station.GetPMTTraces(0,pmtNo).GetCharge()
                peak   = station.GetPMTTraces(0,pmtNo).GetPeak()

                
                D_main_values_container[(pmtNo-1)*120:pmtNo*120] = norm_trace(trace,charge,peak)  
                # print()
                # print(f'PMT {pmtNo} : {trace[:30]}')
            D_main_coords_container[0] = N
            # D_main_coords_container[1] = TRC
            # D_main_coords_container[2] = PMT  # Already set
            D_main_coords_container[3] = X
            D_main_coords_container[4] = Y


            D_main_values[Nappends*360:(Nappends+1)*360] = D_main_values_container
            D_main_coords[:,Nappends*360:(Nappends+1)*360] = D_main_coords_container
            Nappends += 1






        # Targets
        logE[N] = norm_Energy(GenShower.GetEnergy())
        Xmax[N] = norm_Xmax(GenShower.GetXmaxInterpolated())
        Axis[N] = norm_Axis(GenShower.GetAxisSiteCS())
        Core[N] = norm_core(GenShower.GetCoreSiteCS(),position_CT)[:2]
        EventMyId[N] = int(f'{Evstr}{i:05d}')
        # for testing purposes - exit early
        if StopEarly and N>1000: break
    if StopEarly and N>1000: break

# print()
# print('Currently Loaded Values:')
# print()
# import sys
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
#                          key= lambda x: -x[1])[:30]:  # we only print top 10 entries
#     print("{:>30}: {:>8}".format(name, size/1e9))

print('Cleaning Before Save')
del valid_indices, detGeom,ev,GenShower,stations,station,position_CT,time_CT,position,time,X,Y,SignalEnd,SignalStart,trace,charge,peak


print()
print('Finished Reading, Trimming Arrays')
# Shorten the arrays to the correct size in case the allocation was too much
D_main_values = D_main_values[:Nappends*360]
D_main_coords = D_main_coords[:,:Nappends*360]

# print('Sizes of Arrays')
# print('Values: ', np.shape(D_main_values))
# print('Coords: ', np.shape(D_main_coords))

# Save Indices
torch.manual_seed(1234)
length = N+1
indices = torch.randperm(length)

train_indices = indices[:int(0.8*length)]
val_indices   = indices[int(0.8*length):int(0.9*length)]
test_indices  = indices[int(0.9*length):]

# Need to make sure the Dataloaders Provide Shuffled Data, during training and validation.
# This sorting is required because the indexing uses masks instead of indexing
train_indices,_ = train_indices.sort()
val_indices,_   = val_indices.sort()
test_indices,_  = test_indices.sort()


print('Converting to torch tensors, for Targets and Auxillary data')
logE = torch.from_numpy(logE[:length])
Xmax = torch.from_numpy(Xmax[:length])
Axis = torch.from_numpy(Axis[:length])
Core = torch.from_numpy(Core[:length])
# EventId = torch.from_numpy(EventId[:length])
EventMyId = torch.from_numpy(EventMyId[:length])
D_aux = torch.from_numpy(D_aux)

print('Saving logE')
logE_train = logE[train_indices]
logE_val   = logE[val_indices]
logE_test  = logE[test_indices]

torch.save(logE_train,f'{Paths.NormData}/logE_train.pt')
torch.save(logE_val,f'{Paths.NormData}/logE_val.pt')
torch.save(logE_test,f'{Paths.NormData}/logE_test.pt')
del logE, logE_train, logE_val, logE_test


print('Saving Xmax')
Xmax_train = Xmax[train_indices]
Xmax_val   = Xmax[val_indices]
Xmax_test  = Xmax[test_indices]

torch.save(Xmax_train,f'{Paths.NormData}/Xmax_train.pt')
torch.save(Xmax_val,f'{Paths.NormData}/Xmax_val.pt')
torch.save(Xmax_test,f'{Paths.NormData}/Xmax_test.pt')
del Xmax, Xmax_train, Xmax_val, Xmax_test


print('Saving Axis')
Axis_train = Axis[train_indices]
Axis_val   = Axis[val_indices]
Axis_test  = Axis[test_indices]

torch.save(Axis_train,f'{Paths.NormData}/Axis_train.pt')
torch.save(Axis_val,f'{Paths.NormData}/Axis_val.pt')
torch.save(Axis_test,f'{Paths.NormData}/Axis_test.pt')
del Axis, Axis_train, Axis_val, Axis_test

print('Saving Core')
Core_train = Core[train_indices]
Core_val   = Core[val_indices]
Core_test  = Core[test_indices]

torch.save(Core_train,f'{Paths.NormData}/Core_train.pt')
torch.save(Core_val,f'{Paths.NormData}/Core_val.pt')
torch.save(Core_test,f'{Paths.NormData}/Core_test.pt')
del Core, Core_train, Core_val, Core_test

print('Saving EventMyId')
EventMyId_train = EventMyId[train_indices]
EventMyId_val   = EventMyId[val_indices]
EventMyId_test  = EventMyId[test_indices]

torch.save(EventMyId_train,f'{Paths.NormData}/EventMyId_train.pt')
torch.save(EventMyId_val,f'{Paths.NormData}/EventMyId_val.pt')
torch.save(EventMyId_test,f'{Paths.NormData}/EventMyId_test.pt')
del EventMyId, EventMyId_train, EventMyId_val, EventMyId_test

print('Saving D_aux')
D_aux_train = D_aux[train_indices]
D_aux_val   = D_aux[val_indices]
D_aux_test  = D_aux[test_indices]

torch.save(D_aux_train,f'{Paths.NormData}/D_aux_train.pt')
torch.save(D_aux_val,f'{Paths.NormData}/D_aux_val.pt')
torch.save(D_aux_test,f'{Paths.NormData}/D_aux_test.pt')
del D_aux, D_aux_train, D_aux_val, D_aux_test
gc.collect()



# Massive Tensor Time
# Test New approach here, proven to work

train_mask = np.isin(D_main_coords[0],train_indices)
val_mask   = np.isin(D_main_coords[0],val_indices)
test_mask  = np.isin(D_main_coords[0],test_indices)

D_main_coords_train = D_main_coords[:,train_mask]
D_main_coords_val   = D_main_coords[:,val_mask]
D_main_coords_test  = D_main_coords[:,test_mask]

#Adjust the event indices to be sequential and good for new array
unique_vals, counts = np.unique(D_main_coords_train[0], return_counts=True)
new_indices = np.repeat(np.arange(len(unique_vals)), counts)
D_main_coords_train[0] = new_indices
unique_vals, counts = np.unique(D_main_coords_val[0], return_counts=True)
new_indices = np.repeat(np.arange(len(unique_vals)), counts)
D_main_coords_val[0] = new_indices
unique_vals, counts = np.unique(D_main_coords_test[0], return_counts=True)
new_indices = np.repeat(np.arange(len(unique_vals)), counts)
D_main_coords_test[0] = new_indices


D_main_values_train = D_main_values[train_mask]
D_main_values_val   = D_main_values[val_mask]
D_main_values_test  = D_main_values[test_mask]

D_main_train_new = torch.sparse_coo_tensor(torch.from_numpy(D_main_coords_train),torch.from_numpy(D_main_values_train),(len(train_indices),120,3,11,11))
D_main_val_new   = torch.sparse_coo_tensor(torch.from_numpy(D_main_coords_val),torch.from_numpy(D_main_values_val),(len(val_indices),120,3,11,11))
D_main_test_new  = torch.sparse_coo_tensor(torch.from_numpy(D_main_coords_test),torch.from_numpy(D_main_values_test),(len(test_indices),120,3,11,11))

del D_main_coords_train, D_main_coords_val, D_main_coords_test
del D_main_values_train, D_main_values_val, D_main_values_test
gc.collect()

print('Saving D_main')


torch.save(D_main_val_new,f'{Paths.NormData}/D_main_val.pt')
print('    Validation Saved')
del D_main_val_new
gc.collect()

torch.save(D_main_test_new,f'{Paths.NormData}/D_main_test.pt')
print('    Test Saved')
del D_main_test_new
gc.collect()

torch.save(D_main_train_new,f'{Paths.NormData}/D_main_train.pt')
print('    Train Saved')
del D_main_train_new
gc.collect()

print('Done')

# Old Approach
# print('Saving D_main')
# print('    Creating Sparse tensor')
# D_main = torch.sparse_coo_tensor(D_main_coords,D_main_values,(length,120,3,11,11))
# print('    Full tensor shape = ',D_main.shape)
# del D_main_coords, D_main_values
# gc.collect()

# print('    Saving Tensors')
# D_main_val   = D_main.index_select(0,val_indices)

# if torch.equal(D_main_val.to_dense(),D_main_val_new.to_dense()):
#     print('        Validation tensors are equal')
# else:
#     print('        Validation tensors are not equal')

# torch.save(D_main_val,f'{Paths.NormData}/D_main_val.pt')
# del D_main_val
# gc.collect()
# print('        Validation Saved')

# D_main_test  = D_main.index_select(0,test_indices)




# if torch.equal(D_main_test.to_dense(),D_main_test_new.to_dense()):
#     print('        Test tensors are equal')
# else:
#     print('        Test tensors are not equal')

# torch.save(D_main_test,f'{Paths.NormData}/D_main_test.pt')
# del D_main_test
# gc.collect()
# print('        Test Saved')
# D_main_train = D_main.index_select(0,train_indices)

# if torch.equal(D_main_train.to_dense(),D_main_train_new.to_dense()):
#     print('        Train tensors are equal')
# else:
#     print('        Train tensors are not equal')

# if not SaveDense:
#     del D_main
#     gc.collect()

# torch.save(D_main_train,f'{Paths.NormData}/D_main_train.pt')
# del D_main_train
# gc.collect()
# print('        Train Saved')

# # Save entire D_main as dense for testing purposes
# if SaveDense:
#     print('    Saving Full Dense Tensor')
#     torch.save(D_main.to_dense(),f'{Paths.NormData}/D_main_FullDense.pt')
#     del D_main
#     gc.collect()







