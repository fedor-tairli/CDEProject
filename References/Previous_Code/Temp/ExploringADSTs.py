import numpy as np
from adst3 import RecEventProvider
import os

os.system('clear')


ADSTPath = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/18.0_18.5/helium/'
Filename = 'SIB23c_180_185_helium_Hybrid_CORSIKA76400_Run010.root'

for ev in RecEventProvider(ADSTPath + Filename):
    Event = ev
    break

print('Got Event Id',Event.GetEventId())

Eye = Event.GetEye(2)

SDEvent = Event.GetSDEvent()

for station in SDEvent.GetStationVector():
    Station = station
    break


for station in SDEvent.GetSimStationVector():
    SimStation = station
    break

for particle in SimStation.GetParticleVector():
    Particle = particle
    break