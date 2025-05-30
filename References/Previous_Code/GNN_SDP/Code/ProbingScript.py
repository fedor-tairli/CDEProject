from adst3 import RecEventProvider, GetDetectorGeometry
import numpy as np

import os 
os.system('clear')
import torch
# filename = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/18.0_18.5/helium/SIB23c_180_185_helium_Hybrid_CORSIKA76400_Run010.root'
# EventId_OfInterest = 100000108
# # EventId_OfInterest = 100330103


# for ev in RecEventProvider(filename):
#     if ev.GetSDEvent().GetEventId() == EventId_OfInterest:
#         print('Found event')
#         break


# FdEvent = ev.GetHottestEye()
# print(f'HottestEyeId = {FdEvent.GetEyeId()}')

# SDEvent = ev.GetSDEvent()
# HottestStation = SDEvent.GetStationById(FdEvent.GetFdRecGeometry().GetHottestStation())
# FdRecPixel = FdEvent.GetFdRecPixel()
# TriggeredPixels = np.nonzero(np.array(FdRecPixel.GetStatus()))[0]


# print('Triggered Pixels')
# print('|  Ipix |  TelId  |  EyeId  |  PixelId   |  PixelShortId   | PixelCentroid   |')
# for iPix in TriggeredPixels:
#     iPix = int(iPix)
#     PixId = FdRecPixel.GetID(iPix)
#     PixShortId = FdRecPixel.GetPixelId(iPix)
#     TelId = FdRecPixel.GetTelescopeId(iPix)
#     EyeId = FdEvent.GetEyeId()
#     PixelCentroid = FdRecPixel.GetTime(iPix)
#     if TelId in [7,8,9]: 
#         pointingId = 'upward'
#     else:
#         pointingId = 'downward'

#     print(f'|  {str(iPix).ljust(3)}  |  {str(TelId).ljust(5)}  |  {str(EyeId).ljust(5)}  |  {str(PixId).ljust(8)}  |  {str(PixShortId).ljust(13)}  |  {str(PixelCentroid)[:6].ljust(13)}  |')


# HottestStationSecond = HottestStation.GetTimeSecond()
# HottestStationNSecond = HottestStation.GetTimeNSecond()
# print(f'HottestStationTime = {HottestStationSecond} S + {HottestStationNSecond} ns')

# FdEventSecond  = FdEvent.GetGPSSecond()
# FdEventNSecond = FdEvent.GetGPSNanoSecond()
# print(f'FdEventTime = {FdEventSecond} S + {FdEventNSecond} ns')
# print('Triggered Pixels')

# for iPix in TriggeredPixels:
#     iPix = int(iPix)
#     pixelId = FdRecPixel.GetID(iPix)
#     if pixelId == 409:
#         break
# LastPixelLoc = iPix
# print(f'TraceLen : {len(FdRecPixel.GetTrace(iPix))}')
# LastPixelTime = FdRecPixel.GetTime(LastPixelLoc)*50
# print(f'Last pixel Time = {LastPixelTime} 100 ns')
# # print(f'Last pixel Time = {LastPixelTime} 100 ns')

# # print(f'StationTime = {HottestStationSecond+HottestStationNSecond*10**-9} S')
# # print(f'Pixel  Time = {FdEventSecond+FdEventNSecond*10**-9 - 10**-4 + LastPixelTime*10**-9} S')

# # PixelTime   = FdEventSecond+FdEventNSecond*10**-9 - 10**-4 + LastPixelTime*10**-9
# # StationTime = HottestStationSecond+HottestStationNSecond*10**-9
# # print(f'TimeDifference = {(HottestStationSecond+HottestStationNSecond*10**-9 - (FdEventSecond+FdEventNSecond*10**-9 - 10**-4 + LastPixelTime*10**-9))*10**9} ns')

# # print(f'Station Id = {HottestStation.GetId()}')
# # print(f'Time50     = {HottestStation.GetTime50()}')


# StationTime = HottestStationNSecond - 5*10**8
# PixelTime   = FdEventNSecond-10**5 + LastPixelTime - 5*10**8

# print(f'Station Time : {StationTime}')
# print(f'Pixel  Time  : {PixelTime}')
# print(f'Time Diff    : {StationTime-PixelTime}')



dir = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/'
energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
mass   = ['helium','iron','oxygen','proton'] #,'photon']
run = ['Run010']#,'Run030','Run080','Run090']

files = []
# Iterate over the files
for e in energy:
    # Construct the filenames array
    for m in mass:
        for r in run:
            sub_e = e.replace('.','')
            filename = f'{dir}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{r}.root'
            files.append(filename)
            # print(filename)

AllIDs = []
Nevents = 0
for file in files:
    for ev in RecEventProvider(file,1):
        Nevents += 1
        # print(f'Got {Nevents} events')
        if Nevents % 1000 == 0: print(f'Got {Nevents} events')
        AllIDs.append(ev.GetSDEvent().GetEventId())

print(f'Got {len(AllIDs)} events')
AllIDs = torch.tensor(AllIDs)
torch.save(AllIDs,'AllEventIds.pt')

