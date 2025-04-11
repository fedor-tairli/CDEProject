import torch
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
os.system('clear')

import warnings
warnings.filterwarnings("ignore")

from adst3 import RecEventProvider, GetDetectorGeometry

from Dataset2 import DatasetContainer



Testing = False




# Filenames are generated based on the pre-cut files
DataDir = '../Data/MC/low_cut'
files = os.listdir(DataDir)
files = [DataDir+'/'+f for f in files if f.endswith('.root')]
if Testing: files = [f for f in files if 'b45' in f]
files.sort()# Sort files alphabetically

DetGeom = GetDetectorGeometry('../Data/Selected_Events_Extended.root')
##########################################################################################
# Predefine functions here

ALL_Ev_Ids = []
##########################################################################################
# Initialize the dataset container

# Example of event ID String
# Batch_300089975:Shower_7069
print('Please check IDs work for "High" energy files')
# read the data
EvCounter = 0
for file in files:
    for Event in RecEventProvider(file,mode=1):
        EvCounter += 1
        if EvCounter%1000 == 0: print(f'Event {EvCounter}')
        EventId = Event.GetEventId()
        ALL_Ev_Ids.append(EventId)
        


        
        # Event Variables
        EventId_1 = int(EventId.split(':')[0].split('_')[1][-6:])
        EventId_2 = int(EventId.split(':')[1].split('_')[1])
        
    print()
ALL_Ev_Ids = np.array(ALL_Ev_Ids)
print('Total Events:',len(ALL_Ev_Ids))
print('Unique Events:',len(np.unique(ALL_Ev_Ids)))


