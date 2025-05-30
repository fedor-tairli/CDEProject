####################################
#                                  #
#       Merge the datatensors      #
#    Split into train,val,test     #
#     Save as torch tensors        #
#                                  #
####################################

import torch
import paths
import numpy as np
import glob
import os
import gc

# Paths Management
Paths = paths.load_ProjectPaths(paths.get_caller_dir())

Paths.NormData = Paths.data_path + 'NormData/'
if not os.path.exists(Paths.NormData):
    os.makedirs(Paths.NormData)



# Read file names
files = glob.glob(Paths.RawData+'/*.npz')
# files = [f'{Paths.RawData}TestFile.npz']

# Initialise the lists to append during the loop
D_main_list = []
D_aux_list  = []
logE_list   = []
Xmax_list   = []
Axis_list   = []
Core_list   = []
EventMyId_list = []



Nfiles = 0

print('#################################################################')
for file in files:
    if 'TestFile' in file: 
        # info = '123_123_123_123'.split('_')
        print('Hit the testfile')
        continue
    elif 'Run030' in file or 'Run080' in file or 'Run090' in file:
        print('Hit the cut file')
        continue
    else:
        Nfiles += 1
        info = file.split('/')[-1].split('.')[0].split('_')
        print('#################################################################')
    
    IDstr = ['0','0','0']

    # updates to IDstr to be appended into MyID
    # Energy
    if '180' == info[0]:
        IDstr[0]='1'
    elif '185' == info[0]:
        IDstr[0]='2'
    elif '190' == info[0]:
        IDstr[0]='3'
    elif '195' == info[0]:
        IDstr[0]='4'
    else:
        IDstr[0]='5'
    # Mass
    if 'proton' == info[2]:
        IDstr[1]='1'
    elif 'helium' == info[2]:
        IDstr[1]='2'
    elif 'oxygen' == info[2]:
        IDstr[1]='3'
    elif 'iron' == info[2]:
        IDstr[1]='4'
    else:
        IDstr[1]='5'

    # Run
    if 'Run010' == info[3]:
        IDstr[2]='1'
    elif 'Run030' == info[3]:
        IDstr[2]='2'
    elif 'Run080' == info[3]:
        IDstr[2]='3'
    elif 'Run090' == info[3]:
        IDstr[2]='4'
    else:
        IDstr[2]='5'

    IDstr = ''.join(IDstr)
    # print(IDstr)

    print('File Number   :', Nfiles)
    print('Reading file  : ', file.split('/')[-1].split('.')[0])
    f = np.load(file)

    D_main = f['D_main']
    D_aux  = f['D_aux']
    logE   = f['logE']
    Xmax   = f['Xmax']
    Axis   = f['Axis']
    Core   = f['Core']
    EventMyId = f['EventMyId']

    del f
    gc.collect()

    # print('D_main :',D_main.shape)
    # print('D_aux  :',D_aux.shape)
    # print('logE   :',logE.shape)
    # print('Xmax   :',Xmax.shape)
    # print('Axis   :',Axis.shape)
    # print('Core   :',Core.shape)
    # Find where data doesn't exist
    print('Original Size : ', Core.shape[0])
    empty = np.where( D_main.sum(axis=(1,2,3,4)) == 0 )

    D_main = np.delete(D_main, empty, axis=0)
    D_aux = np.delete(D_aux, empty, axis=0)
    logE = np.delete(logE, empty, axis=0)
    Xmax = np.delete(Xmax, empty, axis=0)
    Axis = np.delete(Axis, empty, axis=0)
    Core = np.delete(Core, empty, axis=0)
    EventMyId = np.delete(EventMyId, empty, axis=0)
    
    print('Final Size    : ', Core.shape[0])
    print('#################################################################')
    # Some minor edits
    
    EventMyId = np.array([f"{IDstr}{id:05d}" for id in EventMyId]) # Add the IDstr to the EventMyId
    D_main = np.transpose(D_main, (0,2,1,3,4))                     # Transpose the D_main to match input order
    

    # Append the lists
    D_main_list.append(D_main)
    D_aux_list.append(D_aux)
    logE_list.append(logE)
    Xmax_list.append(Xmax)
    Axis_list.append(Axis)
    Core_list.append(Core)
    EventMyId_list.append(EventMyId)

# concatenate the lists
print('Concatenating the lists')
D_main_tensor = torch.from_numpy(np.concatenate(D_main_list, axis=0))
del D_main_list
D_aux_tensor = torch.from_numpy(np.concatenate(D_aux_list, axis=0))
del D_aux_list
logE_tensor = torch.from_numpy(np.concatenate(logE_list, axis=0))
del logE_list
Xmax_tensor = torch.from_numpy(np.concatenate(Xmax_list, axis=0))
del Xmax_list
Axis_tensor = torch.from_numpy(np.concatenate(Axis_list, axis=0))
del Axis_list
Core_tensor = torch.from_numpy(np.concatenate(Core_list, axis=0))
del Core_list
EventMyId_tensor = torch.from_numpy(np.concatenate(EventMyId_list, axis=0).astype(int))
del EventMyId_list
gc.collect()

print('Splitting the tensors')

# Split into train,val,test (80,10,10)
torch.manual_seed(1234)
length = Core_tensor.shape[0]
indices = torch.randperm(length)

train_indices = indices[:int(0.8*length)]
val_indices   = indices[int(0.8*length):int(0.9*length)]
test_indices  = indices[int(0.9*length):]


D_main_train = D_main_tensor[train_indices]
D_aux_train  = D_aux_tensor[train_indices]
logE_train   = logE_tensor[train_indices]
Xmax_train   = Xmax_tensor[train_indices]
Axis_train   = Axis_tensor[train_indices]
Core_train   = Core_tensor[train_indices]
EventMyId_train = EventMyId_tensor[train_indices]

torch.save(D_main_train, f'{Paths.NormData}/D_main_train.pt')
torch.save(D_aux_train,  f'{Paths.NormData}/D_aux_train.pt')
torch.save(logE_train,   f'{Paths.NormData}/logE_train.pt')
torch.save(Xmax_train,   f'{Paths.NormData}/Xmax_train.pt')
torch.save(Axis_train,   f'{Paths.NormData}/Axis_train.pt')
torch.save(Core_train,   f'{Paths.NormData}/Core_train.pt')
torch.save(EventMyId_train, f'{Paths.NormData}/EventMyId_train.pt')

del D_main_train, D_aux_train, logE_train, Xmax_train, Axis_train, Core_train, EventMyId_train

D_main_val = D_main_tensor[val_indices]
D_aux_val  = D_aux_tensor[val_indices]
logE_val   = logE_tensor[val_indices]
Xmax_val   = Xmax_tensor[val_indices]
Axis_val   = Axis_tensor[val_indices]
Core_val   = Core_tensor[val_indices]
EventMyId_val = EventMyId_tensor[val_indices]

torch.save(D_main_val, f'{Paths.NormData}/D_main_val.pt')
torch.save(D_aux_val,  f'{Paths.NormData}/D_aux_val.pt')
torch.save(logE_val,   f'{Paths.NormData}/logE_val.pt')
torch.save(Xmax_val,   f'{Paths.NormData}/Xmax_val.pt')
torch.save(Axis_val,   f'{Paths.NormData}/Axis_val.pt')
torch.save(Core_val,   f'{Paths.NormData}/Core_val.pt')
torch.save(EventMyId_val, f'{Paths.NormData}/EventMyId_val.pt')

del D_main_val, D_aux_val, logE_val, Xmax_val, Axis_val, Core_val, EventMyId_val

D_main_test = D_main_tensor[test_indices]
D_aux_test  = D_aux_tensor[test_indices]
logE_test   = logE_tensor[test_indices]
Xmax_test   = Xmax_tensor[test_indices]
Axis_test   = Axis_tensor[test_indices]
Core_test   = Core_tensor[test_indices]
EventMyId_test = EventMyId_tensor[test_indices]

torch.save(D_main_test, f'{Paths.NormData}/D_main_test.pt')
torch.save(D_aux_test,  f'{Paths.NormData}/D_aux_test.pt')
torch.save(logE_test,   f'{Paths.NormData}/logE_test.pt')
torch.save(Xmax_test,   f'{Paths.NormData}/Xmax_test.pt')
torch.save(Axis_test,   f'{Paths.NormData}/Axis_test.pt')
torch.save(Core_test,   f'{Paths.NormData}/Core_test.pt')
torch.save(EventMyId_test, f'{Paths.NormData}/EventMyId_test.pt')

del D_main_test, D_aux_test, logE_test, Xmax_test, Axis_test, Core_test, EventMyId_test

print('Done')



