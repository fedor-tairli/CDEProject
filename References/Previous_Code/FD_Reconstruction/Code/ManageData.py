import torch
import numpy as np
import sys
import os

import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
sys.path.append(Paths.data_path)


# Chi0
def Chii0_to_net(Chi0):
    return torch.cos(Chi0) if torch.is_tensor(Chi0) else np.cos(Chi0)

def Chi0_to_val(Cos_chi0):
    return torch.acos(Cos_chi0) if torch.is_tensor(Cos_chi0) else np.arccos(Cos_chi0)

# Phi
def Phi_to_net(Phi):
    return torch.sin(Phi) if torch.is_tensor(Phi) else np.sin(Phi)

def Phi_to_val(Sin_phi):
    return torch.asin(Sin_phi) if torch.is_tensor(Sin_phi) else np.arcsin(Sin_phi)

# Theta
def Theta_to_net(Theta):
    return torch.cos(Theta) if torch.is_tensor(Theta) else np.cos(Theta)

def Theta_to_val(Cos_theta):
    return torch.acos(Cos_theta) if torch.is_tensor(Cos_theta) else np.arccos(Cos_theta)

# Phi Offsets
# Phi_Offset_by_Mirror_inDeg={1:105,2:135,3:165,4:195,5:225,6:255}
# Phi_Offset_by_Mirror = {id:val*np.pi/180 for id,val in Phi_Offset_by_Mirror_inDeg.items()}
# Phi_Offset_by_Mirror_sin = {id:np.sin(val) for id,val in Phi_Offset_by_Mirror.items()}

def Phi_Offset_by_Mirror(MirrorId):
    MirrorDeg = (MirrorId-1)*30+15
    return MirrorDeg*np.pi/180


def Phi_to_mirror(Phi, MirrorId):
    Phi[Phi<0] += 2*np.pi
    return Phi-Phi_Offset_by_Mirror(MirrorId)-np.pi/2


    
    
    
def Phi_to_site(Phi,MirrorId):
    Phi = Phi+Phi_Offset_by_Mirror(MirrorId)
    Phi[Phi>np.pi] -= 2*np.pi

def SDP_to_XYZ(SDP):
    Phi = SDP[:,0]
    Theta = SDP[:,1]

    X = torch.sin(Theta)*torch.cos(Phi)
    Y = torch.sin(Theta)*torch.sin(Phi)
    Z = torch.cos(Theta)

    return torch.stack((X,Y,Z),dim=1)

def XYZ_to_SDP(XYZ):
    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]

    Phi = torch.atan2(Y,X)
    Theta = torch.acos(Z)

    return torch.stack((Phi,Theta),dim=1)
# Rp
RpStd = 5269.0
RpMean = 13109.0
def Rp_to_net(Rp):
    return (Rp-RpMean)/RpStd
def Rp_to_val(Rp):
    return Rp*RpStd+RpMean


# Pixel Time
PixTimeStd = 133.0
PixTimeMean = 375.0
def PixTime_to_net(PixTime):
    return (PixTime-PixTimeMean)/PixTimeStd
def PixTime_to_val(PixTime):
    return PixTime*PixTimeStd+PixTimeMean

# Pixel Signal (Already Log10)
# Go for Unity at 10^5 total charge
def PixSig_to_net(PixSig):
    return PixSig/5
def PixSig_to_val(PixSig):
    return PixSig*5


def PixDur_to_net(PixDur):
    Percentile68 = 19.0
    return PixDur/Percentile68

def PixDur_to_val(PixDur):
    Percentile68 = 19.0
    return PixDur*Percentile68


if __name__ == '__main__':
    print('Loading Data')
    
    Savedir = Paths.data_path + 'NormData/'
    Paths.NormData = Savedir
    if not os.path.exists(Savedir):
        os.mkdir(Savedir)
    
    LoadDir = Paths.data_path + 'RawData/'
    Main        = torch.load(LoadDir+'Main.pt')
    Meta        = torch.load(LoadDir+'Meta.pt')
    PixDur      = torch.load(LoadDir+'PixDur.pt')
    GenGeometry = torch.load(LoadDir+'GenGeometry.pt')
    RecGeometry = torch.load(LoadDir+'RecGeometry.pt')

    print('Splitting Data')

    Length = Main.shape[0]
    # Split train-val-test : 0.8-0.1-0.1
    indices = torch.randperm(Length)

    train_indices = indices[:int(0.8*Length)]
    val_indices   = indices[int(0.8*Length):int(0.9*Length)]
    test_indices  = indices[int(0.9*Length):]


    Main_train = Main[train_indices]
    Main_val   = Main[val_indices]
    Main_test  = Main[test_indices]
    Meta_train = Meta[train_indices]
    Meta_val   = Meta[val_indices]
    Meta_test  = Meta[test_indices]
    PixDur_train = PixDur[train_indices]
    PixDur_val   = PixDur[val_indices]
    PixDur_test  = PixDur[test_indices]
    GenGeometry_train = GenGeometry[train_indices]
    GenGeometry_val   = GenGeometry[val_indices]
    GenGeometry_test  = GenGeometry[test_indices]
    RecGeometry_train = RecGeometry[train_indices]
    RecGeometry_val   = RecGeometry[val_indices]
    RecGeometry_test  = RecGeometry[test_indices]

    # Deleting NaNs
    Main_train_maks        = torch.isnan(Main_train.sum(dim=(1,2,3)))
    Main_val_maks          = torch.isnan(Main_val.sum(dim=(1,2,3)))
    Main_test_maks         = torch.isnan(Main_test.sum(dim=(1,2,3)))
    PixDur_train_maks      = torch.isnan(PixDur_train.sum(dim=(1,2,3)))
    PixDur_val_maks        = torch.isnan(PixDur_val.sum(dim=(1,2,3)))
    PixDur_test_maks       = torch.isnan(PixDur_test.sum(dim=(1,2,3)))

    GenGeometry_train_maks = torch.isnan(GenGeometry_train.sum(dim=1))
    GenGeometry_val_maks   = torch.isnan(GenGeometry_val.sum(dim=1))
    GenGeometry_test_maks  = torch.isnan(GenGeometry_test.sum(dim=1))

    RecGeometry_train_maks = torch.isnan(RecGeometry_train.sum(dim=1))
    RecGeometry_val_maks   = torch.isnan(RecGeometry_val.sum(dim=1))
    RecGeometry_test_maks  = torch.isnan(RecGeometry_test.sum(dim=1))

    Combined_train_maks    = Main_train_maks | PixDur_train_maks| GenGeometry_train_maks | RecGeometry_train_maks
    Combined_val_maks      = Main_val_maks   | PixDur_val_maks  | GenGeometry_val_maks   | RecGeometry_val_maks
    Combined_test_maks     = Main_test_maks  | PixDur_test_maks | GenGeometry_test_maks  | RecGeometry_test_maks

    Main_train             = Main_train[~Combined_train_maks]
    Main_val               = Main_val[~Combined_val_maks]
    Main_test              = Main_test[~Combined_test_maks]
    Meta_train             = Meta_train[~Combined_train_maks]
    Meta_val               = Meta_val[~Combined_val_maks]
    Meta_test              = Meta_test[~Combined_test_maks]
    PixDur_train           = PixDur_train[~Combined_train_maks]
    PixDur_val             = PixDur_val[~Combined_val_maks]
    PixDur_test            = PixDur_test[~Combined_test_maks]
    GenGeometry_train      = GenGeometry_train[~Combined_train_maks]
    GenGeometry_val        = GenGeometry_val[~Combined_val_maks]
    GenGeometry_test       = GenGeometry_test[~Combined_test_maks]
    RecGeometry_train      = RecGeometry_train[~Combined_train_maks]
    RecGeometry_val        = RecGeometry_val[~Combined_val_maks]
    RecGeometry_test       = RecGeometry_test[~Combined_test_maks]


    print('Saving')
    torch.save(Main_train,Savedir+'Main_train.pt')
    torch.save(Main_val,Savedir+'Main_val.pt')
    torch.save(Main_test,Savedir+'Main_test.pt')
    torch.save(Meta_train,Savedir+'Meta_train.pt')
    torch.save(Meta_val,Savedir+'Meta_val.pt')
    torch.save(Meta_test,Savedir+'Meta_test.pt')
    torch.save(PixDur_train,Savedir+'PixDur_train.pt')
    torch.save(PixDur_val,Savedir+'PixDur_val.pt')
    torch.save(PixDur_test,Savedir+'PixDur_test.pt')
    torch.save(GenGeometry_train,Savedir+'GenGeometry_train.pt')
    torch.save(GenGeometry_val,Savedir+'GenGeometry_val.pt')
    torch.save(GenGeometry_test,Savedir+'GenGeometry_test.pt')
    torch.save(RecGeometry_train,Savedir+'RecGeometry_train.pt')
    torch.save(RecGeometry_val,Savedir+'RecGeometry_val.pt')
    torch.save(RecGeometry_test,Savedir+'RecGeometry_test.pt')

    print('Done')


    


    




