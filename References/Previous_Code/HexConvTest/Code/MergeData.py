####################################
#                                  #
#       Merge the datatensors      #
#    Split into train,val,test     #
#     Save as torch tensors        #
#                                  #
####################################

import torch
import paths

Paths = paths.load_ProjectPaths(paths.get_caller_dir())



Runs = ['Run010','Run030','Run080','Run090']

# Load and merge Runs for X, Y_E,Y_Xmax,Y_Core,Y_Axis

X_tensors      = [torch.load(f'{Paths.NormData}/{Run}_X.pt') for Run in Runs]
Y_E_tensors    = [torch.load(f'{Paths.NormData}/{Run}_Y_E.pt') for Run in Runs]
Y_Xmax_tensors = [torch.load(f'{Paths.NormData}/{Run}_Y_Xmax.pt') for Run in Runs]
Y_Core_tensors = [torch.load(f'{Paths.NormData}/{Run}_Y_Core.pt') for Run in Runs]
Y_Axis_tensors = [torch.load(f'{Paths.NormData}/{Run}_Y_Axis.pt') for Run in Runs]

# Concatenate tensors

X      = torch.cat(X_tensors)
Y_E    = torch.cat(Y_E_tensors)
Y_Xmax = torch.cat(Y_Xmax_tensors)
Y_Core = torch.cat(Y_Core_tensors)
Y_Axis = torch.cat(Y_Axis_tensors)

# Split into train,val,test (80,10,10)
torch.manual_seed(1234)

indices = torch.randperm(X.shape[0])

train_indices = indices[:int(0.8*X.shape[0])]
val_indices   = indices[int(0.8*X.shape[0]):int(0.9*X.shape[0])]
test_indices  = indices[int(0.9*X.shape[0]):]

X_train      = X[train_indices]
Y_E_train    = Y_E[train_indices]
Y_Xmax_train = Y_Xmax[train_indices]
Y_Core_train = Y_Core[train_indices]
Y_Axis_train = Y_Axis[train_indices]

X_val      = X[val_indices]
Y_E_val    = Y_E[val_indices]
Y_Xmax_val = Y_Xmax[val_indices]
Y_Core_val = Y_Core[val_indices]
Y_Axis_val = Y_Axis[val_indices]

X_test      = X[test_indices]
Y_E_test    = Y_E[test_indices]
Y_Xmax_test = Y_Xmax[test_indices]
Y_Core_test = Y_Core[test_indices]
Y_Axis_test = Y_Axis[test_indices]

# Save as torch tensors

torch.save(X_train,      f'{Paths.NormData}/X_train.pt')
torch.save(Y_E_train,    f'{Paths.NormData}/Y_E_train.pt')
torch.save(Y_Xmax_train, f'{Paths.NormData}/Y_Xmax_train.pt')
torch.save(Y_Core_train, f'{Paths.NormData}/Y_Core_train.pt')
torch.save(Y_Axis_train, f'{Paths.NormData}/Y_Axis_train.pt')

torch.save(X_val,      f'{Paths.NormData}/X_val.pt')
torch.save(Y_E_val,    f'{Paths.NormData}/Y_E_val.pt')
torch.save(Y_Xmax_val, f'{Paths.NormData}/Y_Xmax_val.pt')
torch.save(Y_Core_val, f'{Paths.NormData}/Y_Core_val.pt')
torch.save(Y_Axis_val, f'{Paths.NormData}/Y_Axis_val.pt')

torch.save(X_test,      f'{Paths.NormData}/X_test.pt')
torch.save(Y_E_test,    f'{Paths.NormData}/Y_E_test.pt')
torch.save(Y_Xmax_test, f'{Paths.NormData}/Y_Xmax_test.pt')
torch.save(Y_Core_test, f'{Paths.NormData}/Y_Core_test.pt')
torch.save(Y_Axis_test, f'{Paths.NormData}/Y_Axis_test.pt')

print('Done')



