from adst3 import RecEventProvider, GetFileInfo,GetDetectorGeometry
from paths import EGFile as EGFile
import os
import numpy as np
os.system('clear')

import pprint 
pp = pprint.PrettyPrinter().pprint


# for ev in RecEventProvider(EGFile):
#     for d in dir(ev.GetGenShower()):
#         if 'Get' in d:
#             print(d)
#     break




# ev = GetFileInfo(EGFile)
# for d in dir(ev):
#     if 'Get' in d:
#         print(d)


# for ev in RecEventProvider(EGFile):
#     d = ev.GetSDEvent()
#     print(dir(d))
#     print()
#     print()
#     print()
#     print()
#     print((d))
#     break

# for ev in RecEventProvider(EGFile):
#     for station in ev.GetSDEvent().GetStationVector():
#         print('50   ',station.GetTime50())
#         print('50rms',station.GetTime50RMS())
#         print('sec  ',station.GetTimeSecond())
#         print('Nsec ',station.GetTimeNSecond())
#         print('Var  ',station.GetTimeVariance())
#         print('SigSt',station.GetSignalStartSlot())
#         print('_____________________________________')
#         # break
#     break



# ev = GetDetectorGeometry(EGFile)
# # for d in dir(ev):
# #     if 'Get' in d:
# #         print(d)
# pp((np.asarray(ev.GetStationPosition(56))))



# for ev in RecEventProvider(EGFile):
#     eye=ev.GetFDEvents()[0]
#     GenShower = eye.GetGenShower()
#     for d in dir(GenShower):
#         print(d)
#     break

file = '/remote/teslaa/tesla/bmanning/data/DenseRings/QGSJETII-04/iron/19_19.5/ADST_DAT175518_10390.root'
# file = '/remote/kennya/data/ADST_MC/NormalXmax_StandardFD/SIB23c/17.6_20.2/pHeNFe/SIB23c_176_202_pHeNFe_NormalXmax_StandardFD_XmaxStandardSubset_NoEnergyCuts_NoFOVcuts_Run15.root'
for ev in RecEventProvider(file):
    SDEvent = ev.GetSDEvent()
    stations = SDEvent.GetStationVector()
    for station in stations:
        if station.IsDense() or station.IsAccidental(): continue # Skip Dense and Accidental Stations
        print(station.GetId())
        # pri
    break

def printGets(obj):
    for d in dir(obj):
        if 'Get' in d:
            if d not in ['GetDrawOption','GetOption','GetIconName','GetDtorOnly','GetObjectInfo','GetObjectStat','Getoption','GetUniqueID']:
                print(d)