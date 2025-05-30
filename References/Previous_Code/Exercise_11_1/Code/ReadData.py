###########################################################
#        For The Exercise, read data from npz file        #
#                 Store in Torch Tensors                  #
###########################################################

import numpy as np
import torch
import paths
from matplotlib import pyplot as plt


Paths = paths.load_ProjectPaths(paths.get_caller_dir())
Paths.RawData = Paths.data_path + 'airshowers.npz'

f = np.load(Paths.RawData)

print(f.files)

# Load Data into Numpy

# time map
T = f['time']
T -= np.nanmean(T)
T /= np.nanstd(T)
T[np.isnan(T)] = 0

print('T: Loaded, Shape:',T.shape)

# signal map
S = f['signal']
S = np.log10(S+1)
S -= np.nanmean(S)
S /= np.nanmax(S)
S[np.isnan(S)] = 0

print('S: Loaded, Shape:',S.shape)

X = np.stack([T,S],axis=-1)
X = np.transpose(X,(0,3,1,2))
print('Input: Created, Shape:',X.shape)

# Load outputs

axis = f['showeraxis']

core = f['showercore'][:,0:2]
core /= 750

# energy - log10(E/eV) in range [18.5, 20]
logE = f['logE']
logE -= 19.25

# Split the data into 90/10 train/test and load into torch tensors

randI = np.arange(X.shape[0])
np.random.shuffle(randI)

X = X[randI,...]
logE = logE[randI,...]
core = core[randI,...]
axis = axis[randI,...]

X_train,X_test       = np.split(X,   [int(0.9*X.shape[0])]   )
logE_train,logE_test = np.split(logE,[int(0.9*logE.shape[0])])
core_train,core_test = np.split(core,[int(0.9*core.shape[0])])
axis_train,axis_test = np.split(axis,[int(0.9*axis.shape[0])])

X_train = torch.from_numpy(X_train).float()
X_test  = torch.from_numpy(X_test).float()

logE_train = torch.from_numpy(logE_train).float()
logE_test  = torch.from_numpy(logE_test).float()

core_train = torch.from_numpy(core_train).float()
core_test  = torch.from_numpy(core_test).float()

axis_train = torch.from_numpy(axis_train).float()
axis_test  = torch.from_numpy(axis_test).float()

print('Data Loaded')


# Use inbuilt torch tensor save function to store data
# Make Paths
Paths.X_train = Paths.data_path + 'X_train.pt'
Paths.X_test  = Paths.data_path + 'X_test.pt'

Paths.logE_train = Paths.data_path + 'logE_train.pt'
Paths.logE_test  = Paths.data_path + 'logE_test.pt'

Paths.core_train = Paths.data_path + 'core_train.pt'
Paths.core_test  = Paths.data_path + 'core_test.pt'

Paths.axis_train = Paths.data_path + 'axis_train.pt'
Paths.axis_test  = Paths.data_path + 'axis_test.pt'
# Save
torch.save(X_train,Paths.X_train)
torch.save(X_test,Paths.X_test)

torch.save(logE_train,Paths.logE_train)
torch.save(logE_test,Paths.logE_test)

torch.save(core_train,Paths.core_train)
torch.save(core_test,Paths.core_test)

torch.save(axis_train,Paths.axis_train)
torch.save(axis_test,Paths.axis_test)

print('Data Saved')

# plotX = np.abs(X_train[:,1,:,:].numpy())
# print(np.shape(plotX))
# plotX = np.sum(plotX,axis = 2)
# plotX = np.sum(plotX,axis = 1)
# print(plotX.shape)
# plotY = logE_train.numpy()
# print(plotY.shape)
# plt.scatter(plotX,plotY)
# plt.show()