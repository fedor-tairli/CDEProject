# Testing How to do depth calculations. 


import numpy as np
import os
import torch
from matplotlib import pyplot as plt


os.system('clear')

import warnings
warnings.filterwarnings("ignore")



from adst3 import RecEventProvider, GetDetectorGeometry
# from FD_Reconstruction_Definitions import Find_Pixel_Id, Get_Pixel_Pos_Dict_EBStyle # looks like i didnt use these
from Dataset2 import DatasetContainer

######### SETUP #########



def clearout():
    os.system('clear')
clearout()

testfile = False
SavePath = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/DatasetFiles/RawData'

# Extra Functions



def ReadFiles(files,ExpectedSize,DatasetName):
    print('Initialising Dataset')

    # Input data arrays
    ExpectedSize = ExpectedSize*1e6 # Total number of events in files = 1e6
    ExpectedSize = int(ExpectedSize)


    print('Begining to read files')
    Nevents = 0                # Number of events read
    NeventsWithNoStations = 0  # Number of events with SD Event Missing
    NeventsWithNoEyes     = 0  # Number of events with FD Event Missing
    print(f'Initialising Dataset : Expected Size = {ExpectedSize}')
    Dataset = DatasetContainer()
    

    # Add the following to the dataset
    EventVals = ['EventID_1/2','EventID_2/2','Primary',\
                 'Station_TotalSignal','Station_Time','Station_Theta','Station_Phi','Station_Distance','Station_Chi_i',\
                 'Gen_LogE','Gen_CosZenith','Gen_Xmax','Gen_dEdXmax',\
                 'Rec_LogE','Rec_CosZenith','Rec_Xmax','Rec_dEdXmax','Rec_UspL','Rec_UspR',\
                 'Gen_SDPPhi','Gen_SDPTheta','Gen_Chi0','Gen_Rp','Gen_T0','Gen_CoreEyeDist',\
                 'Rec_SDPPhi','Rec_SDPTheta','Rec_Chi0','Rec_Rp','Rec_T0','Rec_CoreEyeDist',\
                ]
    for Val in EventVals: Dataset.add_event_value(Val)
    PixelVals = ['PixelID','TelID','EyeID','Status','Charge','Chi_i','Theta','Phi',\
                 'TimeOffset','PulseStart','PulseCentroid','PulseStop']
    for Val in PixelVals: Dataset.add_pixel_value(Val)
    Dataset.add_name(DatasetName)
    Dataset.preallocate_data(ExpectedSize)
    print('Dataset Initialised')
    print(f'Will Be saved to {SavePath}')
    # Data reading

    for filename in files:
        detGeom = GetDetectorGeometry(filename) # in case its different per file, its same per event for sure
        print()
        for i,ev in enumerate(RecEventProvider(filename,0)):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
            # Progress Bar
            Nevents += 1
            if len(files) == 1 : print(f'Event {Nevents}')
            else:
                if Nevents%1000 == 0:
                    print(f'Event {Nevents}, skipped {NeventsWithNoEyes+NeventsWithNoStations} events',end='\r')

            
            
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
            FdStations  = FdEvent.GetStationVector()
            FdRecPixel  = FdEvent.GetFdRecPixel()
            GenShower   = ev.GetGenShower()
            RecShower   = FdEvent.GetFdRecShower()
            RecGeometry = FdEvent.GetFdRecGeometry()
            GenGeometry = FdEvent.GetGenGeometry()
            
            ReferenceTime = 5*10**8 # For simulations its always the same, for real data, make sure the second isnt crossed over

            # Find Hottest Station
            HottestStation = None
            for station in FdStations:
                if not station.IsHybrid(): continue
                HottestStation = station
                break
            if HottestStation == None:
                NeventsWithNoStations += 1
                # print(f'No Station in event {ev.GetSDEvent().GetEventId()}, Total Stations Missing = {NeventsWithNoStations} ')
                continue



            # Initialise Event
            Event = Dataset.get_blank_event()

            # Add Station Information
            Event.add_event_value('Station_TotalSignal',HottestStation.GetTotalSignal())
            Event.add_event_value('Station_Time'       ,HottestStation.GetTimeEye())
            HottestStationPos = detGeom.GetStationPosition(HottestStation.GetId())
            EyePos            = detGeom.GetEye(FdEvent.GetEyeId()).GetEyePos()
            BackWallAngle     = detGeom.GetEye(FdEvent.GetEyeId()).GetBackWallAngle()
            StationDistance,StationTheta,StationPhi = CalculateStationPosition(EyePos,HottestStationPos,BackWallAngle)
            Event.add_event_value('Station_Theta'      ,StationTheta)
            Event.add_event_value('Station_Phi'        ,StationPhi)
            Event.add_event_value('Station_Distance'   ,StationDistance)
            Event.add_event_value('Station_Chi_i'      ,HottestStation.GetChi_i())
            
            # Add Event Information
            Energy  = np.log10(GenShower.GetEnergy())
            Primary = AdjustPrimaryName(GenShower.GetPrimary())
            EventID_1,EventID_2 = FixId(SDEvent.GetEventId(),Energy,Primary)

            Event.add_event_value('EventID_1/2'        ,EventID_1)
            Event.add_event_value('EventID_2/2'        ,EventID_2)
            Event.add_event_value('Primary'            ,Primary)

            # Add Gen Event Info
            Event.add_event_value('Gen_LogE'           ,Energy)
            Event.add_event_value('Gen_CosZenith'      ,GenShower.GetCosZenith())
            Event.add_event_value('Gen_Xmax'           ,GenShower.GetXmaxInterpolated())
            Event.add_event_value('Gen_dEdXmax'        ,GenShower.GetdEdXmaxInterpolated())

            # Add Rec event  Info
            Event.add_event_value('Rec_LogE'           ,np.log10(RecShower.GetEnergy()))
            Event.add_event_value('Rec_CosZenith'      ,RecShower.GetCosZenith())
            Event.add_event_value('Rec_Xmax'           ,RecShower.GetXmax())
            Event.add_event_value('Rec_dEdXmax'        ,RecShower.GetdEdXmax())
            Event.add_event_value('Rec_UspL'           ,RecShower.GetUspL())
            Event.add_event_value('Rec_UspR'           ,RecShower.GetUspR())

            # Add Gen Geometry Info
            Event.add_event_value('Gen_SDPPhi'         ,GenGeometry.GetSDPPhi())
            Event.add_event_value('Gen_SDPTheta'       ,GenGeometry.GetSDPTheta())
            Event.add_event_value('Gen_Chi0'           ,GenGeometry.GetChi0())
            Event.add_event_value('Gen_Rp'             ,GenGeometry.GetRp())
            Event.add_event_value('Gen_T0'             ,GenGeometry.GetT0())
            Event.add_event_value('Gen_CoreEyeDist'    ,GenGeometry.GetCoreEyeDistance())

            # Add Rec Geometry Info
            Event.add_event_value('Rec_SDPPhi'         ,RecGeometry.GetSDPPhi())
            Event.add_event_value('Rec_SDPTheta'       ,RecGeometry.GetSDPTheta())
            Event.add_event_value('Rec_Chi0'           ,RecGeometry.GetChi0())
            Event.add_event_value('Rec_Rp'             ,RecGeometry.GetRp())
            Event.add_event_value('Rec_T0'             ,RecGeometry.GetT0())
            Event.add_event_value('Rec_CoreEyeDist'    ,RecGeometry.GetCoreEyeDistance())

            
            # Add Pixel Info
            # Dont Worry about Seconds, they are not crossed over in simulations
            TriggeredPixels = np.nonzero(np.array(FdRecPixel.GetStatus()))[0]
            for iPix in TriggeredPixels:
                iPix = int(iPix)
                # Initialise Pixel Object
                # Add Pixel Info
                PixelID = FdRecPixel.GetPixelId(iPix)
                TelID   = FdRecPixel.GetTelescopeId(iPix)
                EyeID   = FdEvent.GetEyeId()
                Event.add_pixel_value('PixelID'        ,PixelID)
                Event.add_pixel_value('TelID'          ,TelID)
                Event.add_pixel_value('EyeID'          ,EyeID)
                Event.add_pixel_value('Status'         ,FdRecPixel.GetStatus(iPix))
                Event.add_pixel_value('Charge'         ,FdRecPixel.GetCharge(iPix))
                Event.add_pixel_value('Chi_i'          ,FdRecPixel.GetChi(iPix))
                PulseStart = FdRecPixel.GetPulseStart(iPix)
                PulseStop  = FdRecPixel.GetPulseStop(iPix)
                Event.add_pixel_value('PulseStart'     ,PulseStart)
                Event.add_pixel_value('PulseCentroid'  ,FdRecPixel.GetTime(iPix))
                Event.add_pixel_value('PulseStop'      ,PulseStop)
                Event.add_pixel_value('TimeOffset'     ,FdEvent.GetMirrorTimeOffset(TelID)/100)
                pointingId = 'downward' if TelID>6 else 'upward'
                Phi = detGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelPhi(PixelID-1,pointingId)
                Omega = detGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelOmega(PixelID-1,pointingId)
                Event.add_pixel_value('Phi'            ,Phi)
                Event.add_pixel_value('Theta'          ,90-Omega)
                
                CutTrace = torch.zeros(1,100)
                CutTrace[0,:PulseStop-PulseStart] = torch.tensor(FdRecPixel.GetTrace(iPix))[PulseStart:PulseStop][:100]
                Event.add_trace_values(                 CutTrace)
                Event.next_pixel()

            # Append the event to the dataset
            # Event.ShowEvent(ShowPixels=True)
            Dataset.add_event(Event)


            
        print('-------------------------------File Done---------------------------------')
    print('Done Reading Files')
    print(f'Number of Events Read             = {Nevents}')
    print(f'Number of Events with No Eyes     = {NeventsWithNoEyes}')
    print(f'Number of Events with No Stations = {NeventsWithNoStations}')
    print('Saving Data')
    
    Dataset.Save(DirPath = SavePath)

    print('Finished the Data Reading')
    
def TestDepth():
    path_to_data = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/'
    filename     = 'Selected_Events_Extended.root'

    for i,ev in enumerate(RecEventProvider(path_to_data+filename,0)):
        # FdData
        if i == 6:
            try:
                FdEvent    = ev.GetHottestEye()
            except Exception as e:
                if 'No such eye' in str(e):
                    NeventsWithNoEyes += 1
                    # print(f'No Eye     in event {ev.GetSDEvent().GetEventId()}, Total FdEvents Missing = {NeventsWithNoEyes} ')
                    continue
                else:
                    raise e
            
            Shower = FdEvent.GetFdRecShower()

            Depth = Shower.GetDepth()
            dEdX  = Shower.GetEnergyDeposit()
            
            plt.figure()
            plt.scatter(Depth,dEdX)
            plt.savefig('Depth.png')

        

if __name__ == '__main__':
    TestDepth()
    
            

print()
print()
print()
# os.system('python3.9 ProduceGraphs.py')