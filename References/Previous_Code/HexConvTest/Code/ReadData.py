###########################################################
#                For The HexConvTesting                   #
#                read RawData from ADSTS      file        #
#                 Store As Pandas First                   #
###########################################################

import numpy as np
import paths
import os
import pandas as pd
import pprint 
# import pickle
pp = pprint.PrettyPrinter().pprint

from adst3 import RecEventProvider, GetDetectorGeometry


######### SETUP #########

def clearout():
    os.system('clear')
clearout()

# Initialise some values paths mainly
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
Paths.RawADSTs = Paths.data_path + 'RawADSTs'
Paths.RawData = f'{Paths.data_path}/RawData'

if not os.path.exists(Paths.RawData):
    os.system(f'mkdir -p {Paths.RawData}')

# find list of files to go over: 
TP = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
AP = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
mass   = ['helium','iron','oxygen','proton'] #,'photon']
run = ['Run010','Run030','Run080','Run090']
# For Quad Read : 
# run = ['Run090']

# cannot use the recursive function because of soft links in files stored
#  in andromeda also?

files = []
# Iterate over the files
for e in energy:
    # the low energy files are on vulcan
    if e in ['18.0_18.5','18.5_19.0']:
        Path = AP
    else:
        Path = TP
    
    for m in mass:
        for r in run:
            sub_e = e.replace('.','')
            filename = f'{Path}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{r}.root'
            files.append(filename)
# pp(files)


# Preset the dataframes columns
# Only grab the required data

event_data = ['EventId','UniqueEventId','GenPrimaryEnergy','GenXmax','GenAxisSiteCS','GenCoreSiteCS']
stations_data= ['EventId','StationId','Signal','Position','TimeSecond','TimeNSecond']
# dont need eyes data for this here



# Iterate over the files

for filename in files:
    # Preempt Save Filenames 
    file = (filename.split('/')[-1])[:-5]
    Save_Filename_Event    = Paths.RawData + '/' +  file + '_Event.pt'
    Save_Filename_Stations = Paths.RawData + '/' +  file + '_Stations.pt'
    # Skip if already exists
    if os.path.exists(Save_Filename_Event) and os.path.exists(Save_Filename_Stations):
        continue

    DetGeometry = GetDetectorGeometry(filename)
    # print(filename)
    Nevents = 0
    # For each file, create 2 dataframes
    Event    = pd.DataFrame(columns=event_data)
    EventRow = {key:None for key in event_data}
    Stations = pd.DataFrame(columns=stations_data)
    StationsRow = {key:None for key in stations_data}


    # Get the data

    for ev in RecEventProvider(filename,2):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
        Nevents += 1
        print(f'\rCurrent Event : {Nevents}',end='')
        EventRow['EventId'           ]= ev.GetEventId()
        EventRow['UniqueEventId'     ]= ev.GetSDEvent().GetEventId()
        GenShower                     = ev.GetGenShower() 
        EventRow['GenPrimaryEnergy'  ]= GenShower.GetEnergy()
        EventRow['GenXmax'           ]= GenShower.GetXmaxInterpolated()
        EventRow['GenAxisSiteCS'     ]= np.asarray(GenShower.GetAxisSiteCS())
        EventRow['GenCoreSiteCS'     ]= np.asarray(GenShower.GetCoreSiteCS())
        Event = Event.append(EventRow,ignore_index=True)

        for station in ev.GetSDEvent().GetStationVector():
            if station.IsDense() or station.IsAccidental(): continue # Skipping dense and accidental stations
            StationsRow['EventId'      ]= EventRow['EventId'           ]
            StationsRow['StationId'    ]= station.GetId()
            StationsRow['Signal'       ]= station.GetTotalSignal()
            StationsRow['Position'     ]= np.asarray(DetGeometry.GetStationPosition(station.GetId()))
            StationsRow['TimeSecond'   ]= station.GetTimeSecond()
            StationsRow['TimeNSecond'  ]= station.GetTimeNSecond()

            Stations = Stations.append(StationsRow,ignore_index=True)
        # if Nevents == 100: break

    # Save the data per file using pickle
    
    Event.to_pickle(Save_Filename_Event)
    Stations.to_pickle(Save_Filename_Stations)
    # break



