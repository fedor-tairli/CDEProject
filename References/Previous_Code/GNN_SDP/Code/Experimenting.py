from adst3 import RecEventProvider,GetDetectorGeometry
import numpy as np


filepath = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/Selected_Events.root'
# filepath = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/19.0_19.5/iron/SIB23c_190_195_iron_Hybrid_CORSIKA76400_Run010.root'
DetGeom = GetDetectorGeometry(filepath)

def Calculate_Chii(Chi0,Rp,T0,Ti):
    Chi_i = 2*np.arctan((T0-Ti*100)*3*10**8/Rp) - Chi0
    return -Chi_i


printAll = True



for i,ev in enumerate(RecEventProvider(filepath)):
    EventPrintedTels = []
    FdEvent = ev.GetHottestEye()
    FdRecPixel = FdEvent.GetFdRecPixel()
    TriggeredPixels = np.nonzero(np.array(FdRecPixel.GetStatus()))[0]
    for iPix in TriggeredPixels:
        iPix = int(iPix)
        PulseCentroid = FdRecPixel.GetTime(iPix)
        PulseStart    = FdRecPixel.GetPulseStart(iPix)
        PulseStop     = FdRecPixel.GetPulseStop(iPix)
        EyeId = FdEvent.GetEyeId()
        TelId = FdRecPixel.GetTelescopeId(iPix)
        if TelId not in EventPrintedTels:
            MirrorTimeOffset = FdEvent.GetMirrorTimeOffset(TelId)/100
            EventPrintedTels.append(TelId)
            print(f'Event {i} | Eye {EyeId} | Tel {TelId} | MirrorTimeOffset = {MirrorTimeOffset}')
    
        
        
        # print(f'Eye {EyeId} | Tel {TelId} | Pix {PixId} | PulseStart = {PulseStart} | PulseStop = {PulseStop} | PulseCentroid = {PulseCentroid} | MirrorTimeOffset = {MirrorTimeOffset} | nFADCBins = {nFADCBins} | nFADCBinWidth = {nFADCBinWidth}')
                                 
    # break
    # Station = FdEvent.GetStation(0)
    














# for ev in RecEventProvider(filepath):
#     # Event = ev
#     # break
#     print(f'Event Id= {ev.GetSDEvent().GetEventId()}', end='\r')
#     if ev.GetSDEvent().GetEventId() == 100600101:
#         Event = ev
#         break
    
# print()
# SDEvent     = ev.GetSDEvent()
# FdEvent    = ev.GetHottestEye()
        
# FdRecPixel  = FdEvent.GetFdRecPixel()
# GenShower   = ev.GetGenShower()
# RecShower   = FdEvent.GetFdRecShower()
# RecGeometry = FdEvent.GetFdRecGeometry()
# GenGeometry = FdEvent.GetGenGeometry()
# HottestStation = SDEvent.GetStationById(RecGeometry.GetHottestStation())
# ReferenceTime = 5*10**8 # For simulations its always the same, for real data, make sure the second isnt crossed over

# print(f'Reference time : {ReferenceTime}')
# print(f'Station Time   : {HottestStation.GetTimeNSecond() - ReferenceTime}')
# print(f'Station is used for hybrid : {HottestStation.IsHybrid()}')
# FdEventReferenceTime = FdEvent.GetGPSNanoSecond()-ReferenceTime -10**5
# print(f'FdEvent GPSNanoSecond: {FdEvent.GetGPSNanoSecond()}')
# print(f'FdReferenceTime: {FdEventReferenceTime}')

# FdStations = FdEvent.GetStationVector()

# for i,station in enumerate(FdStations):
#     # if station.IsDense(): continue
#     if not station.IsCandidate(): continue
#     if not station.IsHybrid(): continue
#     print(f'Station {station.GetId()} time: {str(station.GetTimeEye())[:6].ljust(6)} | Is Hybrid: {station.IsHybrid()} | Is Candidate : {station.IsCandidate()}')
#     break


# Stations = SDEvent.GetStationVector()
# StationN = 0
# for i,station in enumerate(Stations):
#     if station.IsDense(): continue
#     if not station.IsCandidate(): continue

#     StationN += 1
#     print(f'{StationN} Station {station.GetId()} time: {str(station.GetTimeNSecond() - ReferenceTime)[:6].ljust(6)} | Is Hybrid: {station.IsHybrid()} | Is Candidate : {station.IsCandidate()}')

# TriggeredPixels = np.array(FdRecPixel.GetStatus()) == 4
# # print(TriggeredPixels)
# currentTel = 0 
# for iPix in range(len(TriggeredPixels)):
#     if TriggeredPixels[iPix] == False:
#         continue
#     # Initialise Pixel Object
#     PixelEyeID = FdEvent.GetEyeId()
#     PixelTelID = FdRecPixel.GetTelescopeId(iPix)
#     PixelID    = FdRecPixel.GetPixelId(iPix)
#     if PixelTelID != currentTel:
#         print()
#         currentTel = PixelTelID
#     if PixelEyeID == 5 or PixelTelID in [7,8,9]:
#         PixelTimeScale = 50
#         pointingId = 'upward'
#     else:
#         PixelTimeScale = 100
#         pointingId = 'downward'
#     # PixelTimeScale = 50
#     PixelPulseStart    =  FdEventReferenceTime+ FdRecPixel.GetPulseStart(iPix)*PixelTimeScale
#     PixelPulseStop     =  FdEventReferenceTime+ FdRecPixel.GetPulseStop(iPix) *PixelTimeScale
    
#     PixelPulseStart    =  FdRecPixel.GetPulseStart(iPix)#*PixelTimeScale
#     PixelPulseStop     =  FdRecPixel.GetPulseStop(iPix) #*PixelTimeScale
    
#     PixelPulseCentroid =  FdRecPixel.GetTime(iPix)      #*PixelTimeScale
    
#     print(f'Pixel in Tel {PixelTelID} of Eye {PixelEyeID} | ID : {str(PixelID).ljust(4)} | Pulse Start: {str(PixelPulseStart)[:7].ljust(7)} | Pulse Stop: {str(PixelPulseStop)[:7].ljust(7)} | Pulse Centroid: {str(PixelPulseCentroid)[:7].ljust(7)} |') 

