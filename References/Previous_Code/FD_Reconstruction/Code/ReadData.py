###########################################################
#                For The FD Reconstruction                #
#                Read RawData from ADST S                 #
#                 Normalisations to be done after         #
#                  Store as per convinience               #
#           Use custom objects to work with DataSets?     #
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
import warnings
warnings.filterwarnings("ignore")


from adst3 import RecEventProvider, GetDetectorGeometry
from FD_Reconstruction_Definitions import Find_Pixel_Id, Get_Pixel_Pos_Dict_EBStyle

### Set 5h timer to wait for memory to be free, just to be sure
import time
print('Waiting 5h for memory to be free')
# time.sleep(5*60*60)




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
dir = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/'
energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
mass   = ['helium','iron','oxygen','proton'] #,'photon']
run = ['Run010']
# For Quad Read : 
size =1/4


print('Initialising Files array')

files = []
# Iterate over the files
for e in energy:
    # Construct the filenames array
    for m in mass:
        for r in run:
            sub_e = e.replace('.','')
            filename = f'{dir}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{r}.root'
            files.append(filename)


# Conditions and ...

testfile = False
StopEarly = False
SaveDense = False

if SaveDense:
    print('Warning: Will attempt to save Dense Tensor.')
if testfile:
    files = ['/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/19.5_20.0/proton/SIB23c_195_200_proton_Hybrid_CORSIKA76400_Run010.root']
    

# Iterate over the files

print('Initialising Arrays')

# Input data arrays
size = size*1e6
size = int(size)
if StopEarly: size = 10

print('Begining to read files')
Nevents = -1       # Number of events read


# Data Goes Here

FD_Pixel_Pos_Dict = Get_Pixel_Pos_Dict_EBStyle()
MetaNumVars = 10

# Mirror size = 22x20
Main        = torch.zeros((size,2,22,20),dtype=torch.float32)-999 # Pixel Signals and CentroidTime in the mirror
PulseDur    = torch.zeros((size,1,22,20),dtype=torch.float32)-999 # Pulse Duration in the mirror
Meta        = torch.zeros((size,MetaNumVars),dtype=torch.float32)-999 # [logE,Xmax,Axis[0],Axis[1],Axis[2],Core[0],Core[2],Core[3],EyeId,HottestMirrorId]
GenGeometry = torch.zeros((size,5))-999 # [Chi0,Rp,T0,Phi,Theta]
RecGeometry = torch.zeros((size,5))-999 # [Chi0,Rp,T0,Phi,Theta]

for filename in files:
    detGeom = GetDetectorGeometry(filename) # in case its different per file, its same per event for sure
    for i,ev in enumerate(RecEventProvider(filename,0)):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
        # SubBranches
        GenShower = ev.GetGenShower()
        try:
            FdEvent   = ev.GetHottestEye()
        except:
            # No FD Data, Skip
            continue

        if FdEvent.GetEyeId() == 5: # Skip if HEAT is the hottest Eye
            continue
        if FdEvent.GetEyeId() == 6: # if HeCo is the hottest Eye, Get the next Hottest, that doesnt have HEAT in it
            try:
                for eye in reversed(ev.GetFdEvents()):
                    if eye.GetEyeId() not in [5,6]:
                        FdEvent = eye
            except:
                continue 

        FdRecPixel = FdEvent.GetFdRecPixel()
        
        Nevents += 1

        print(f'\rCurrent Progress: {Nevents}', end='')
        

        # Event Observables
        Axis   = np.array(GenShower.GetAxisSiteCS())
        Core   = np.array(GenShower.GetCoreSiteCS())
        # MetaData
        HottestEyeId = ev.GetHottestEyeId()
        HottestMirrorId = FdEvent.GetHottestMirrorId()

        # Meta[Nevents,:] = [logE,Xmax,Axis[0],Axis[1],Axis[2],Core[0],Core[1],Core[2],HottestEyeId,HottestMirrorId]
        Meta[Nevents,0] = np.log10(GenShower.GetEnergy())
        Meta[Nevents,1] = GenShower.GetXmaxInterpolated()
        Meta[Nevents,2] = Axis[0]
        Meta[Nevents,3] = Axis[1]
        Meta[Nevents,4] = Axis[2]
        Meta[Nevents,5] = Core[0]
        Meta[Nevents,6] = Core[1]
        Meta[Nevents,7] = Core[2]
        Meta[Nevents,8] = FdEvent.GetEyeId()
        Meta[Nevents,9] = HottestMirrorId
        # Zero out the Main array
        Main[Nevents,...] = 0
        PulseDur[Nevents,...] = 0
        # Loop over pixels (only Triggered)
        TriggeredPixels = np.nonzero(np.array(FdRecPixel.GetStatus()))[0]
        for iPix in TriggeredPixels:
            iPix=int(iPix)
            if FdRecPixel.GetTelescopeId(iPix) != HottestMirrorId: continue # Skip if not the SAME mirror
            PixPos = FD_Pixel_Pos_Dict[FdRecPixel.GetPixelId(iPix)]
            PixTime = FdRecPixel.GetTime(iPix)
            PixSignal = np.log10(FdRecPixel.GetCharge(iPix))
            PixStart  = FdRecPixel.GetPulseStart(iPix)
            PixStop   = FdRecPixel.GetPulseStop(iPix)
            PixDur    = PixStop-PixStart
            Main[Nevents,0,PixPos[0],PixPos[1]] = PixTime
            Main[Nevents,1,PixPos[0],PixPos[1]] = PixSignal
            PulseDur[Nevents,0,PixPos[0],PixPos[1]] = PixDur
        FdGenGeom = FdEvent.GetGenGeometry()
        FdRecGeom = FdEvent.GetFdRecGeometry()
        
        # Get The GenGeometryData
        GenGeometry[Nevents,0] = FdGenGeom.GetChi0()
        GenGeometry[Nevents,1] = FdGenGeom.GetRp()
        GenGeometry[Nevents,2] = FdGenGeom.GetT0()
        GenGeometry[Nevents,3] = FdGenGeom.GetSDPPhi()
        GenGeometry[Nevents,4] = FdGenGeom.GetSDPTheta()

        # Get The RecGeometryData
        RecGeometry[Nevents,0] = FdRecGeom.GetChi0()
        RecGeometry[Nevents,1] = FdRecGeom.GetRp()
        RecGeometry[Nevents,2] = FdRecGeom.GetT0()
        RecGeometry[Nevents,3] = FdRecGeom.GetSDPPhi()
        RecGeometry[Nevents,4] = FdRecGeom.GetSDPTheta()

        if Nevents >= size-1:
            break

    if Nevents >= size-1:
        print('Number Of Events Expected Reached')
        break


# Trim Arrays
Main        = Main[:Nevents+1,...]
Meta        = Meta[:Nevents+1,...]
PulseDur    = PulseDur[:Nevents+1,...]
GenGeometry = GenGeometry[:Nevents+1,...]
RecGeometry = RecGeometry[:Nevents+1,...]

# Save Data
print('Saving Data')
if testfile:
    torch.save(Main,f'{Paths.RawData}/Main_test.pt')
    torch.save(Meta,f'{Paths.RawData}/Meta_test.pt')
    torch.save(PulseDur,f'{Paths.RawData}/PixDur_test.pt')
    torch.save(GenGeometry,f'{Paths.RawData}/GenGeometry_test.pt')
    torch.save(RecGeometry,f'{Paths.RawData}/RecGeometry_test.pt')
else:
    torch.save(Main,f'{Paths.RawData}/Main.pt')
    torch.save(Meta,f'{Paths.RawData}/Meta.pt')
    torch.save(PulseDur,f'{Paths.RawData}/PixDur.pt')
    torch.save(GenGeometry,f'{Paths.RawData}/GenGeometry.pt')
    torch.save(RecGeometry,f'{Paths.RawData}/RecGeometry.pt')

print('Done')

