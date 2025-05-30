#########################################################################
#       In this file we collect base versions of the data               #
#       We do not edit the data, simply store as a raw version          #
#       The method of storage is a dictionarry with torch tensors       #
#########################################################################


# Base packages

import numpy as np
import paths
import os
# Other required packages
from adst3 import RecEventProvider, GetDetectorGeometry

def clearout():
    os.system('clear')




# Initialise some values
Paths           = paths.load_ProjectPaths(paths.get_caller_dir()) # Load Path manager
Paths.ADSTs_dir = paths.DenseRings   # Hard Path to where the data is (or relative or coment out if already defined)
Paths.RawData   = Paths.data_path+'RawData.pt'




required_data = ['EventId','GenPrimaryEnergy']  # List of data keys

data = {key:None for key in required_data}   # Creates an empty dictionary to later populate with data. 

if Paths.ADSTs_dir.endswith('.root'):        # Get an array of filepaths to ADSTs
    files = [Paths.ADSTs_dir]
else:
    files = paths.get_root_files_recursive(Paths.ADSTs_dir)  


# number_of_events = 0                        # Find number of events
# for filename in files:
#     for ev in RecEventProvider(filename,mode=1):
#         number_of_events+=1


ReadMode        = 0                         # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
# Run the data Collection
for filename in files:
    for ev in RecEventProvider(filename, ReadMode):
        EventId           = ev.GetEventId()
        GenShower         = ev.GetGenShower() 
        GenPrimaryEnergy  = GenShower.GetEnergy()
        GenXmax           = GenShower.GetXmaxInterpolated()
        GendEdXmax        = GenShower.GetdEdXmaxInterpolated()

        
        print(GendEdXmax)
        # for d in dir(ev.GetGenShower()):
        #     if 'Get' in d:
        #         print(d)

        break
    # clearout()
    break















