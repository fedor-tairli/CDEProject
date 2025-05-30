# Base packages

import numpy as np
import pandas as pd
import paths
import os
import pickle
# Other required packages
from adst3 import RecEventProvider, GetDetectorGeometry , GetFileInfo
import pprint 
pp = pprint.PrettyPrinter().pprint

DetGem = GetDetectorGeometry(paths.EGFile)

Ids = np.asarray(DetGem.GetStationIds())
StationPositions = pd.DataFrame(columns = ['StationId','X','Y','Z'])
row = {key:None for key in StationPositions.columns}

for id in Ids:
    row['StationId']= id
    Position = DetGem.GetStationPosition(int(id))

    row['X'] = Position[0]
    row['Y'] = Position[1]
    row['Z'] = Position[2]
    StationPositions = StationPositions.append(row,ignore_index=True)

StationPositions.to_csv('StationPositions.csv')