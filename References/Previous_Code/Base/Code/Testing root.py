# import pyroot
# import ROOT
import sys
import adst3
import paths
from time import sleep

# for p in sys.path:
#     print(p)


for ev in adst3.RecEventProvider(paths.EGFile):
    print(ev.GetEventId())

sleep(3)