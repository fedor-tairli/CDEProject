import numpy as np
import paths
import os
import torch
from matplotlib import pyplot as plt

os.system('clear')

import warnings
warnings.filterwarnings("ignore")


from adst3 import RecEventProvider, GetDetectorGeometry
# from FD_Reconstruction_Definitions import Find_Pixel_Id, Get_Pixel_Pos_Dict_EBStyle
# from Dataset import DatasetContainer, EventContainer, PixelContainer

######### SETUP #########

def clearout():
    os.system('clear')
clearout()

testfile = True



def ReadFiles(files):
    # Counters
    Nevents = 0
    NeventsWithNoEyes     = 0
    NeventsWithNoStations = 0

    print('Reading Files')

    for filename in files:
        # Load the file
        detGeom = GetDetectorGeometry(filename)
        for ev in RecEventProvider(filename,0): # Read Mode 0=Full, 2=NoTraces 1=ShowerLevelObservables
            Nevents += 1 # Progress Bar
            if Nevents%1 == 0:
                print(f'Event = {Nevents}')

            # FdData
            try:
                FdEvent    = ev.GetHottestEye()
            except Exception as e:
                if 'No such eye' in str(e):
                    NeventsWithNoEyes += 1
                    # print(f'No Eye     in event {ev.GetSDEvent().GetEventId()}, Total FdEvents Missing = {NeventsWithNoEyes} ')
                    continue
                else:
                    raise e
            # Get Objects
            SDEvent     = ev.GetSDEvent()
            FdStations = FdEvent.GetStationVector()
            FdRecPixel  = FdEvent.GetFdRecPixel()
            GenShower   = ev.GetGenShower()
            RecShower   = FdEvent.GetFdRecShower()
            RecGeometry = FdEvent.GetFdRecGeometry()
            GenGeometry = FdEvent.GetGenGeometry()


            Stations = FdEvent.GetStationVector()
            HottestStation = None
            for station in Stations:
                if not station.IsHybrid(): continue
                HottestStation = station
                break
            if HottestStation == None:
                NeventsWithNoStations += 1
                continue
            

            # Now we get pixels and start slapping. 
            # Pick Event with not too many pixels
            RecPixelMask = torch.tensor(FdRecPixel.GetStatus()) >1
            if torch.sum(RecPixelMask) > 90:
                continue
            RecPixelIndices = torch.nonzero(RecPixelMask).flatten().to(torch.int32)
            EventTraceSum = torch.zeros(1000)
            
            for iPix in RecPixelIndices:
                iPix = int(iPix)
                Trace = torch.tensor(FdRecPixel.GetTrace(iPix))
                PulseStart = FdRecPixel.GetPulseStart(iPix)
                PulseStop  = FdRecPixel.GetPulseStop(iPix)
                TelId      = FdRecPixel.GetTelescopeId(iPix)
                print(f'iPix = {iPix} | PulseStart = {PulseStart} | PulseStop = {PulseStop}, TraceLen = {len(Trace)} | TelId = {TelId}')
                if Trace.shape[0] == 2000: # Sum Every pair of bins
                    Trace = Trace[::2] + Trace[1::2]
                EventTraceSum += Trace

            print()
            plt.figure(figsize = (50,10))
            # plot bars for the trace
            plt.bar(range(1000),EventTraceSum)
            plt.xlim(200,600)
            plt.savefig('test.png')
                


            break # End at the event 1
            

            
if __name__ == '__main__':

    if testfile:
        files = ['/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/Selected_Events_Extended.root']
        ReadFiles(files)
    else:
        print('Going Through All Runs')
        for RUN in ['Run010','Run030','Run080','Run090']:
            run = [RUN]
            print(f'Run = {run}')
            # find list of files to go over: 
            dir = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
            energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
            mass   = ['helium','iron','oxygen','proton'] #,'photon']
            files = []
            # Iterate over the files
            for e in energy:
                # Construct the filenames array
                for m in mass:
                    for r in run:
                        sub_e = e.replace('.','')
                        filename = f'{dir}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{r}.root'
                        files.append(filename)
            ReadFiles(files)



