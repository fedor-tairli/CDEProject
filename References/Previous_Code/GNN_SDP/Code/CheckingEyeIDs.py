import numpy
from adst3 import RecEventProvider

filename = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/18.0_18.5/helium/SIB23c_180_185_helium_Hybrid_CORSIKA76400_Run010.root'

for ev in RecEventProvider(filename):
    for eye in ev.GetFDEvents():
        if eye.GetEyeId() in [5]:
            Eye = eye
        else:
            Eye = None
    if Eye is None:
        continue
    print(Eye.GetEyeId())
    FdRecPixel = Eye.GetFdRecPixel()
    TriggeredPixels = numpy.nonzero(numpy.array(FdRecPixel.GetStatus()))[0]

    for iPix in TriggeredPixels:
        iPix = int(iPix)
        
        print(f'Pixel {iPix} : Id {FdRecPixel.GetID(iPix)} : TelId {FdRecPixel.GetTelescopeId(iPix)}')

    print('____________________________________________________________________')


    