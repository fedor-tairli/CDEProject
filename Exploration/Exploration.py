import torch
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import seaborn as sns
os.system('clear')

import warnings
warnings.filterwarnings("ignore")

from adst3 import RecEventProvider, GetDetectorGeometry

print('Older code here has absolute paths sometimes, GOTTA FIX THAT')


DataDir = '../Data/MC'
files = os.listdir(DataDir+'/low_cut')
files = [f for f in files if f.endswith('.root')]
files = [f for f in files if 'b01' in f]
files.sort()# Sort files alphabetically
files = [DataDir+'/low_cut/'+f for f in files]

# DataDir = '../Data/MC'
# subdirs = os.listdir(DataDir+'/low')
# subdirs = [subdir for subdir in subdirs if subdir.startswith('b')]
# subdirs.sort()
# print(subdirs)
# files = []
# for subdir in subdirs:
#     subfiles = os.listdir(DataDir+'/'+subdir+'/PCGF')
#     subfiels = [f for f in files if f.endswith('.root')]
#     # subfiles = subfiles.sort()
#     files += [DataDir+'/'+subdir+'/PCGF/'+s for s in subfiles]
# files.sort()
# files = [f for f in files if f.endswith('.root')]   
# # for file in files:
#     print(file)
# exit()





DetGeom = GetDetectorGeometry('../Data/Selected_Events_Extended.root')

##########################################################################################
# # Try to plot the value of Rp vs Direct Cherenkov light Fraction
# All_Rp = []
# All_CherenkovFraction = []
# EvNumber = 0
# for file in files:
#     for Event in RecEventProvider(file):
#         EvNumber += 1
#         if EvNumber % 100 == 0: print(f'Event: {EvNumber}')
#         # print('Event ID:',Event.GetEventId())
#         FdEvent = Event.GetFDEvents()[-1]
#         All_Rp               .append(FdEvent.GetGenGeometry().GetRp())
#         eyeEvent = Event.GetEye(6)
#         for iTel in range(7,10):# Check which HeCo Telescope is in the event
#             if eyeEvent.MirrorIsInEvent(iTel):
#                 telEvent = eyeEvent.GetTelescopeData(iTel)
#                 break
#         if telEvent is None:continue
        
#         All_CherenkovFraction.append(telEvent.GetGenApertureLight().GetCherenkovFraction())



#         if EvNumber > 10000: break
#     if EvNumber > 10000: break

# All_Rp = np.array(All_Rp)
# All_CherenkovFraction = np.array(All_CherenkovFraction)

# plt.figure(figsize=(10,10))
# plt.hist2d(All_Rp,All_CherenkovFraction,bins=100)
# plt.xlabel('Rp')
# plt.ylabel('Cherenkov Fraction')
# plt.colorbar()
# plt.grid()
# plt.savefig('Rp_vs_CherenkovFraction.png')

##########################################################################################
# Working on the alignment of the geometry
# def compute_vector(phi,theta,BackwallAngle):
#     phi_rad = (phi+BackwallAngle)/180*np.pi
#     theta_rad = theta/180*np.pi

#     x = np.sin(theta_rad)*np.cos(phi_rad)
#     y = np.sin(theta_rad)*np.sin(phi_rad)
#     z = np.cos(theta_rad)
#     return np.array([x,y,z])

# def space_angle(v1, v2):
#     """
#     Computes the space angle between two 3-vectors.
    
#     Parameters:
#         v1 (array-like): First 3-vector.
#         v2 (array-like): Second 3-vector.
        
#     Returns:
#         float: Space angle in degrees.
#     """
#     v1 = np.array(v1)
#     v2 = np.array(v2)
    
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
    
#     cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
#     theta_rad = np.arccos(cos_theta)
    
#     return np.degrees(theta_rad)

# def minimum_distance_line_point(p, v, q):
#     """Compute the minimum distance between a 3D line and a point.
    
#     Parameters:
#     p (array-like): A point on the line (3D).               p = Core
#     v (array-like): Direction vector of the line (3D).      v = Axis
#     q (array-like): The point in space (3D).                q = HEPos
    
#     Returns:
#     float: Minimum distance.
#     """
#     p, v, q = np.array(p), np.array(v), np.array(q)
#     w = q - p
#     cross_prod = np.cross(v, w)
#     distance = np.linalg.norm(cross_prod) / np.linalg.norm(v)
#     return distance


# # Generate Available filenames
# HE_BackwallAngle = 273.0
# HE_OpticalAxisPhi   = {1:44.45,2:89.87,3:132.83}
# HE_OpticalAxisTheta = {1:44.45,2:45.58,3:44.85}
# HEPos = np.array([-31741.12427975,  15095.57420328,    210.54774754])

# HE_OpticalAxisVec = {i: compute_vector(HE_OpticalAxisPhi[i],HE_OpticalAxisTheta[i],HE_BackwallAngle) for i in range(1,4)}  

# All_SpaceAngles        = []
# All_CherenkovFractions = []
# All_Rps                = []
# All_Distances          = []


# EvNumber = -0
# for file in files:
#     for Event in RecEventProvider(file):
#         EvNumber += 1
#         if EvNumber % 100 == 0: print(f'Event: {EvNumber}')

#         # Get the axis and core in Site Coordinates
#         Core = np.array(Event.GetGenShower().GetCoreSiteCS())
#         Axis = np.array(Event.GetGenShower().GetAxisSiteCS())
        
#         FdEvent = Event.GetFDEvents()[-1]
#         All_Rps.append(FdEvent.GetGenGeometry().GetRp())
        
#         # Get Cherenkov Fraction
#         eyeEvent = Event.GetEye(6)
#         for iTel in range(7,10):# Check which HeCo Telescope is in the event
#             if eyeEvent.MirrorIsInEvent(iTel):
#                 telEvent = eyeEvent.GetTelescopeData(iTel)
#                 break
#         if telEvent is None:continue
#         All_CherenkovFractions.append(telEvent.GetGenApertureLight().GetCherenkovFraction())

#         # Get the Space angle between the Axis and Heco Telescope Optical Axis

#         this_SpaceAngle = space_angle(Axis,HE_OpticalAxisVec[iTel-6])
#         All_SpaceAngles.append(this_SpaceAngle)
        
#         # Get the distance between the core and the HEPos
#         this_distance = minimum_distance_line_point(Core,Axis,HEPos)
#         All_Distances.append(this_distance)
        
        
        
#         if EvNumber > 10000: break
#     if EvNumber > 10000: break
        



# All_CherenkovFractions = np.array(All_CherenkovFractions)
# All_SpaceAngles        = np.array(All_SpaceAngles)
# All_Rps                = np.array(All_Rps)
# All_Distances          = np.array(All_Distances)

# # Normalise the angles and distances to be between 0 and 1 (ie. /180 and /5000)
# All_SpaceAngles        = All_SpaceAngles/180
# All_Distances          = All_Distances/5000

# All_Scores = All_SpaceAngles*All_Distances

# # Plot a 2 Histogram
# plt.figure(figsize=(10,10))
# sns.jointplot(x=All_Scores, y=All_CherenkovFractions, kind='hist', cmap='Blues', color='blue', marginal_kws=dict(bins=100, fill=True),joint_kws=dict(bins=100, fill=True))
# plt.xlabel('Space Angle')
# plt.ylabel('Cherenkov Fraction')
# plt.savefig('SpaceAngle_vs_CherenkovFraction.png')

        


##########################################################################################

# # Get the data into numpy arrays and pickle so that i can use jypyter notebook
# from Convert_GEOM import ConvertGeometry
# filename = DataDir+'/low/b01/PCGF/ADST.PCGF.300000000.root'
# EvNumber = 0

# ALL_HE_GEOM = []
# ALL_HC_GEOM = []
# ALL_GE_GEOM = []

# for file in files:
#     for Event in RecEventProvider(file):
#         EvNumber += 1
#         # if EvNumber != SelectEvent: continue
#         if EvNumber % 100 == 0: print(f'Event: {EvNumber}')
#         EvGenShower = Event.GetGenShower()
#         FdEvents    = Event.GetFDEvents()
#         HeCo_FdEvent = FdEvents[-1]
#         Heat_FdEvent = FdEvents[-2]
#         if HeCo_FdEvent.GetEyeId() != 6: continue
#         if Heat_FdEvent.GetEyeId() != 5: continue



#         Core = np.array(EvGenShower.GetCoreSiteCS())
#         Axis = np.array(EvGenShower.GetAxisCoreCS())
#         ALL_GE_GEOM.append({'Core':Core,'Axis':Axis})

#         HeatSDP_Theta = Heat_FdEvent.GetGenGeometry().GetSDPTheta()
#         HeatSDP_Phi   = Heat_FdEvent.GetGenGeometry().GetSDPPhi()
#         HeatChi0      = Heat_FdEvent.GetGenGeometry().GetChi0()
#         HeatRp        = Heat_FdEvent.GetGenGeometry().GetRp()

#         HE_GEOM = {'SDP_Theta':HeatSDP_Theta,'SDP_Phi':HeatSDP_Phi,'Chi0':HeatChi0,'Rp':HeatRp}
#         ALL_HE_GEOM.append(HE_GEOM)

#         HC_SDP_Theta = HeCo_FdEvent.GetGenGeometry().GetSDPTheta()
#         HC_SDP_Phi   = HeCo_FdEvent.GetGenGeometry().GetSDPPhi()
#         HC_Chi0      = HeCo_FdEvent.GetGenGeometry().GetChi0()
#         HC_Rp        = HeCo_FdEvent.GetGenGeometry().GetRp()

#         HC_GEOM = {'SDP_Theta':HC_SDP_Theta,'SDP_Phi':HC_SDP_Phi,'Chi0':HC_Chi0,'Rp':HC_Rp}
#         ALL_HC_GEOM.append(HC_GEOM)



        

#         if EvNumber > 10000: break
#     if EvNumber > 10000: break

# with open('All_GE_GEOM.pkl','wb') as f:
#     pickle.dump(ALL_GE_GEOM,f)
# with open('All_HE_GEOM.pkl','wb') as f:
#     pickle.dump(ALL_HE_GEOM,f)
# with open('All_HC_GEOM.pkl','wb') as f:
#     pickle.dump(ALL_HC_GEOM,f)

##########################################################################################
# # Testing the geoemtry conversion function by getting core and axis instead of geometry values
# from Convert_GEOM import ConvertGeometry
# filename = '/remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.300000000.root'
# SelectEvent = 3
# EvNumber = 0
# All_SDP_Theta_Errors = []
# All_SDP_Phi_Errors   = []
# for file in files:
#     for Event in RecEventProvider(file):
#         EvNumber += 1
#         # if EvNumber != SelectEvent: continue
#         if EvNumber % 100 == 0: print(f'Event: {EvNumber}')
#         EvGenShower = Event.GetGenShower()
#         FdEvents = Event.GetFDEvents()
#         HeCo_FdEvent = FdEvents[-1]
#         Heat_FdEvent = FdEvents[-2]
#         if HeCo_FdEvent.GetEyeId() != 6: continue
#         if Heat_FdEvent.GetEyeId() != 5: continue

#         Core = np.array(EvGenShower.GetCoreUTMCS())
#         Axis = np.array(EvGenShower.GetAxisCoreCS())
#         # print('Core:',Core)
#         # print('Axis:',Axis)

#         HEPos = np.array([445498.14,6114209.8, 1707.34])
#         CoreDir = np.array([Core[0]-HEPos[0],Core[1]-HEPos[1],Core[2]-HEPos[2]])
#         CoreDir = CoreDir/np.linalg.norm(CoreDir)
#         Axis = -Axis

#         # SDPVect is the cross product of the core direction and the axis direction
#         SDPVect = np.cross(CoreDir,Axis)
#         SDPVect = SDPVect/np.linalg.norm(SDPVect)
#         # print('SDPVect:',SDPVect)

#         SDP_Phi = np.arctan2(SDPVect[1],SDPVect[0])
#         SDP_Theta = np.arccos(SDPVect[2])
        
#         T = (HEPos[2]-Core[2])/Axis[2]
#         Core = Core + T*Axis
#         HeatCore = Core - HEPos
#         HeatCoreAzimuth = np.arctan2(HeatCore[1],HeatCore[0])
#         HE_BackwallAngle = 273.0

#         SDP_Phi = SDP_Phi - HE_BackwallAngle/180*np.pi+np.pi/2        
#         SDP_Phi = SDP_Phi- np.pi/2
#         if SDP_Phi < -np.pi: SDP_Phi += 2*np.pi
#         if SDP_Phi >  np.pi: SDP_Phi -= 2*np.pi
#         HeatSDP_Theta = Heat_FdEvent.GetGenGeometry().GetSDPTheta()
#         HeatSDP_Phi   = Heat_FdEvent.GetGenGeometry().GetSDPPhi()

#         # Print Theta
#         # print('SDP_Theta    :',SDP_Theta*180/np.pi,'With error: ',SDP_Theta-HeatSDP_Theta)
#         # print('HeatSDP_Theta:',HeatSDP_Theta*180/np.pi)
#         # # Print Phi
#         # print('SDP_Phi      :',SDP_Phi*180/np.pi, 'With error: ',SDP_Phi-HeatSDP_Phi)
#         # print('HeatSDP_Phi  :',HeatSDP_Phi  *180/np.pi)
    
#         # print()

#         All_SDP_Theta_Errors.append(HeatSDP_Theta - SDP_Theta)
#         All_SDP_Phi_Errors  .append(HeatSDP_Phi   - SDP_Phi  )

#         if EvNumber > 1000: break
#     if EvNumber > 1000: break

# All_SDP_Theta_Errors = np.array(All_SDP_Theta_Errors)*180/np.pi
# All_SDP_Phi_Errors   = np.array(All_SDP_Phi_Errors  )*180/np.pi

# All_SDP_Theta_Errors = All_SDP_Theta_Errors[np.abs(All_SDP_Phi_Errors)<=30]
# All_SDP_Phi_Errors   = All_SDP_Phi_Errors  [np.abs(All_SDP_Phi_Errors)<=30]

# fig, axs = plt.subplots(1,2, figsize=(10,5))
# axs[0].hist(All_SDP_Theta_Errors, bins=100)
# axs[1].hist(All_SDP_Phi_Errors  , bins=100)
# axs[0].set_xlabel('SDP Theta Error')
# axs[1].set_xlabel('SDP Phi Error')
# axs[0].grid()
# axs[1].grid()
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')
# plt.tight_layout()
# plt.savefig('GeometryConversionErrors_GetCoreMethod.png')

##########################################################################################
# # Testing the geoemtry conversion function
# from Convert_GEOM import ConvertGeometry
# filename = '/remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.300000000.root'
# SelectEvent = 3
# EvNumber = 0

# All_SDP_Theta_Errors = []
# All_SDP_Phi_Errors   = []
# All_Chi0_Errors      = []
# All_Rp_Errors        = []
# for file in files:
#     for Event in RecEventProvider(file):
#         EvNumber += 1
        
#         if EvNumber % 100 == 0: print(f'Event: {EvNumber}')

#         FdEvents = Event.GetFDEvents()

#         HeCo_FdEvent = FdEvents[-1]
#         Heat_FdEvent = FdEvents[-2]
#         # Check Ids of the eyes
#         if HeCo_FdEvent.GetEyeId() != 6: continue
#         if Heat_FdEvent.GetEyeId() != 5: continue

#         # Collect Gen Geometry from HeCo
#         HeCo_SDP_Theta = HeCo_FdEvent.GetGenGeometry().GetSDPTheta()
#         HeCo_SDP_Phi   = HeCo_FdEvent.GetGenGeometry().GetSDPPhi()
#         HeCo_Chi0      = HeCo_FdEvent.GetGenGeometry().GetChi0()
#         HeCo_Rp        = HeCo_FdEvent.GetGenGeometry().GetRp()
#         Heat_SDP_Theta = Heat_FdEvent.GetGenGeometry().GetSDPTheta()
#         Heat_SDP_Phi   = Heat_FdEvent.GetGenGeometry().GetSDPPhi()
#         Heat_Chi0      = Heat_FdEvent.GetGenGeometry().GetChi0()
#         Heat_Rp        = Heat_FdEvent.GetGenGeometry().GetRp()

#         E_Heat_SDP_Theta, E_Heat_SDP_Phi, E_Heat_Chi0, E_Heat_Rp = ConvertGeometry(HeCo_SDP_Theta,HeCo_SDP_Phi,HeCo_Chi0,HeCo_Rp)

#         All_SDP_Theta_Errors.append(Heat_SDP_Theta - E_Heat_SDP_Theta)
#         All_SDP_Phi_Errors  .append(Heat_SDP_Phi   - E_Heat_SDP_Phi  )
#         All_Chi0_Errors     .append(Heat_Chi0      - E_Heat_Chi0     )
#         All_Rp_Errors       .append(Heat_Rp        - E_Heat_Rp       )

#         if EvNumber > 10000: break
#     if EvNumber > 10000: break

# All_SDP_Theta_Errors = np.array(All_SDP_Theta_Errors)*180/np.pi
# All_SDP_Phi_Errors   = np.array(All_SDP_Phi_Errors  )*180/np.pi
# All_Chi0_Errors      = np.array(All_Chi0_Errors     )*180/np.pi
# All_Rp_Errors        = np.array(All_Rp_Errors       )

# All_SDP_Theta_Errors = All_SDP_Theta_Errors[np.abs(All_SDP_Phi_Errors)<=30]
# All_SDP_Phi_Errors   = All_SDP_Phi_Errors  [np.abs(All_SDP_Phi_Errors)<=30]
# All_Chi0_Errors      = All_Chi0_Errors     [np.abs(All_SDP_Phi_Errors)<=30]
# All_Rp_Errors        = All_Rp_Errors       [np.abs(All_SDP_Phi_Errors)<=100]

# fig,axs = plt.subplots(1,4, figsize=(20,5))
# axs[0].hist(All_SDP_Theta_Errors, bins=100)
# axs[1].hist(All_SDP_Phi_Errors  , bins=100)
# axs[2].hist(All_Chi0_Errors     , bins=100)
# axs[3].hist(All_Rp_Errors       , bins=100)
# axs[0].set_xlabel('SDP Theta Error')
# axs[1].set_xlabel('SDP Phi Error')
# axs[2].set_xlabel('Chi0 Error')
# axs[3].set_xlabel('Rp Error')
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# axs[3].grid()
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')
# axs[2].set_yscale('log')
# axs[3].set_yscale('log')
# plt.tight_layout()
# plt.savefig('GeometryConversionErrors.png')

# print('Event ID:',Event.GetEventId())
# print('GlobalAxis CoreCS',np.array(Event.GetGenShower().GetAxisCoreCS()))
# # print('GlobalAxis SiteCS',np.array(Event.GetGenShower().GetAxisSiteCS()))
# print('HeCo CEDist:',HeCo_FdEvent.GetGenGeometry().GetCoreEyeDistance())
# print('HEAT CEDist:',Heat_FdEvent.GetGenGeometry().GetCoreEyeDistance())


# print()
# print()
# print('HeCo Geometry')
# print('SDP Theta:',HeCo_SDP_Theta/np.pi*180)
# print('SDP Phi  :',HeCo_SDP_Phi  /np.pi*180)
# print('Chi0     :',HeCo_Chi0     /np.pi*180)
# print('Rp       :',HeCo_Rp                 )
# print()
# # Convert to Heat Geometry
# Heat_SDP_Theta, Heat_SDP_Phi, Heat_Chi0, Heat_Rp = ConvertGeometry(HeCo_SDP_Theta,HeCo_SDP_Phi,HeCo_Chi0,HeCo_Rp)
# print()
# print('Expected Heat Geometry and Actual Heat Geometry')
# print('SDP Theta:',Heat_SDP_Theta                             /np.pi*180)
# print('SDP Theta:',Heat_FdEvent.GetGenGeometry().GetSDPTheta()/np.pi*180)
# print('SDP Phi  :',Heat_SDP_Phi                               /np.pi*180)
# print('SDP Phi  :',Heat_FdEvent.GetGenGeometry().GetSDPPhi()  /np.pi*180)
# print('Chi0     :',Heat_Chi0                                  /np.pi*180)
# print('Chi0     :',Heat_FdEvent.GetGenGeometry().GetChi0()    /np.pi*180)
# print('Rp       :',Heat_Rp                                              )
# print('Rp       :',Heat_FdEvent.GetGenGeometry().GetRp()                )
# print('CoreAzim :',Heat_SDP_Phi*180/np.pi-90)
# print('CoreAzim :',Heat_FdEvent.GetGenGeometry().GetSDPPhi()*180/np.pi-90)



##########################################################################################
# Single Event things
# Checking if the SDP is the same in HeCo and HEAT
# filename = '/remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.300000000.root'
# EvNumber = 0
# SelectedEvent = 4
# for Event in RecEventProvider(filename):
#     for FdEvent in Event.GetFDEvents():
#         print('EyeID:',FdEvent.GetEyeId())
#         # print('SDP Phi:',FdEvent.GetGenGeometry().GetSDPPhi()*180/np.pi)
#         # print('SDP Theta:',FdEvent.GetGenGeometry().GetSDPTheta()*180/np.pi)

#     print('Site:',np.array(Event.GetGenShower().GetAxisSiteCS()))
#     print('Core:',np.array(Event.GetGenShower().GetAxisCoreCS()))

#     print()
##########################################################################################
# # Single event things
# # Choose event
# EvNumber = 0
# SelectedEvent = 28
# for i, file in enumerate(files):
#     if not file.endswith('.root'):continue
#     FullPath = os.path.join(DataDir, file)
#     for Event in RecEventProvider(FullPath):
#         EvNumber += 1
#         if EvNumber == SelectedEvent: break
#     if EvNumber == SelectedEvent: break


# AllPixelPhi    = []
# AllPixelTheta  = []
# AllPixelSignal = []
# AllPixelStatus = []
# FdEvent = Event.GetFDEvents()[-1]
# FdRecPixel = FdEvent.GetFdRecPixel()

# FirstTriggerBin = 1e99
# LastTriggerBin = -1

# # Find the summation window
# TriggeredPixels = np.nonzero(np.array(FdRecPixel.GetStatus())==4)[0]
# for iPix in TriggeredPixels:
#     iPix = int(iPix)
#     PulseStart = FdRecPixel.GetPulseStart(iPix)
#     PulseStop  = FdRecPixel.GetPulseStop (iPix)
#     if PulseStart < FirstTriggerBin: FirstTriggerBin = PulseStart
#     if PulseStop  > LastTriggerBin:  LastTriggerBin  = PulseStop
# # Add 3 bins before Start and 3 bins after Stop
# FirstTriggerBin = max(0, FirstTriggerBin-3)
# LastTriggerBin  = min(2000, LastTriggerBin+3)
# print('TriggerDuration+6=',LastTriggerBin-FirstTriggerBin)

# for iPix in range(len(FdRecPixel.GetStatus())):
#     iPix = int(iPix)
#     PixelID = FdRecPixel.GetPixelId(iPix)
#     TelID   = FdRecPixel.GetTelescopeId(iPix)
#     EyeID   = FdEvent.GetEyeId()


#     Phi = DetGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelPhi(PixelID-1,'upward')
#     Omega = DetGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelOmega(PixelID-1,'upward')
#     Theta = Omega
#     Status = FdRecPixel.GetStatus(iPix)
#     Signal = np.array(FdRecPixel.GetTrace(iPix)[FirstTriggerBin:LastTriggerBin]).max()
    
#     AllPixelPhi   .append(Phi)
#     AllPixelTheta .append(Theta)
#     AllPixelSignal.append(Signal)
#     AllPixelStatus.append(Status)


# # find all expected pixel positions
# Expected_Pixel_Phi   = []
# Expected_Pixel_Theta = []
# for i in range(440):
#     Phi   = DetGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelPhi  (i,'upward')
#     Omega = DetGeom.GetEye(EyeID).GetTelescope(TelID).GetPixelOmega(i,'upward')
#     Theta = Omega
#     Expected_Pixel_Phi.append(Phi)
#     Expected_Pixel_Theta.append(Theta)

# Expected_Pixel_Phi   = np.array(Expected_Pixel_Phi  )
# Expected_Pixel_Theta = np.array(Expected_Pixel_Theta)

# AllPixelPhi    = np.array(AllPixelPhi)
# AllPixelTheta  = np.array(AllPixelTheta)
# AllPixelSignal = np.array(AllPixelSignal)
# AllPixelStatus = np.array(AllPixelStatus)

# color = np.log(AllPixelSignal)


# # Plot
# plt.figure(figsize=(10,10), facecolor='lightgrey')

# # Expected pixels (gray crosses)
# plt.scatter(Expected_Pixel_Phi, Expected_Pixel_Theta, s=1, c='grey', marker='x')

# # Normalize color values so both scatter plots use the same colormap scale
# norm = mcolors.Normalize(vmin=min(color), vmax=max(color))  # Ensure consistent scaling
# cmap = plt.cm.inferno  # Define colormap once

# # Scatter plot for pixels with AllPixelStatus != 0 (circle markers)
# sc = plt.scatter(AllPixelPhi[AllPixelStatus!=0], AllPixelTheta[AllPixelStatus!=0], 
#                  s=100, c=color[AllPixelStatus!=0], cmap=cmap, norm=norm, marker='o')

# # Scatter plot for pixels with AllPixelStatus == 0 (hexagon markers)
# plt.scatter(AllPixelPhi[AllPixelStatus==0], AllPixelTheta[AllPixelStatus==0], 
#             s=100, c=color[AllPixelStatus==0], cmap=cmap, norm=norm, marker='H')

# # Labels
# plt.xlabel('Phi [deg]')
# plt.ylabel('Theta [deg]')

# # Reverse X axis
# plt.gca().invert_xaxis()
# plt.gca().set_facecolor('lightgrey')

# # Add colorbar using the first scatter plot
# plt.colorbar(sc)

# plt.show()
# plt.savefig('CameraPlot.png')

    
##########################################################################################

ShortRun=True
AllCherenkovFraction = []
AllXmaxAngle = []
AllMeanAngle = []
AllEventClass = []


evCounter = 0
for i, file in enumerate(files):
    if not file.endswith('.root'):continue
    # FullPath = os.path.join(DataDir, file)
    for Event in RecEventProvider(file):
        evCounter += 1
        print(f'Event: {evCounter}', end='\r')
        eyeEvent = Event.GetEye(6)
        telEvent = None
        for iTel in range(7,10):# Check which HeCo Telescope is in the event
            if eyeEvent.MirrorIsInEvent(iTel):
                telEvent = eyeEvent.GetTelescopeData(iTel)
                break
        if telEvent is None:continue

        GenApLight = telEvent.GetGenApertureLight()
        AllCherenkovFraction.append(GenApLight.GetCherenkovFraction())
        AllXmaxAngle        .append(GenApLight.GetXmaxAngle()        )
        AllMeanAngle        .append(GenApLight.GetMeanAngle()        )
        AllEventClass       .append(eyeEvent.GetEventClass()          )
        
        if ShortRun and evCounter>1000:break
    print('\n')
    if ShortRun and evCounter>1000:break

AllCherenkovFraction = np.array(AllCherenkovFraction)
AllXmaxAngle         = np.array(AllXmaxAngle)/np.pi*180
AllMeanAngle         = np.array(AllMeanAngle)/np.pi*180
AllEventClass        = np.array(AllEventClass)

# Define shower candidate classes
Unique_Shower_Candidates = ["'Close Shower'", "'Horizontal Shower'","'Shower Candidate'"]

# Identify unique event classes and sort them with shower candidates first
unique_classes = np.unique(AllEventClass)
unique_classes = sorted(unique_classes, key=lambda x: x not in Unique_Shower_Candidates)

# Assign colors
colors = plt.cm.Wistia(np.linspace(0, 1, len(unique_classes)))
# print(dir(plt.cm))
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot stacked histograms with hatching
for ax, data, xlabel in zip(axs, [AllXmaxAngle, AllMeanAngle, AllCherenkovFraction], 
                            ['Xmax Angle', 'Mean Angle', 'Cherenkov Fraction']):
    hist_data = [data[AllEventClass == cls] for cls in unique_classes]
    bars = ax.hist(hist_data, bins=50, stacked=True, color=colors, label=unique_classes)

    # Apply hatching to shower candidates
    for rects, cls in zip(bars[2], unique_classes):  # bars[2] contains patches
        if cls in Unique_Shower_Candidates:
            for rect in rects:
                rect.set_hatch('xx')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig('Stacked_Histograms_Test.png')
plt.savefig('Stacked_Histograms_Test.pdf')

##########################################################################################

# AllGenChi0     = np.array(AllGenChi0)
# AllGenCEDist   = np.array(AllGenCEDist)
# AllGenRp       = np.array(AllGenRp)
# AllGenSDPPhi   = np.array(AllGenSDPPhi)
# AllGenSDPTheta = np.array(AllGenSDPTheta)

# fig,axs = plt.subplots(1,5, figsize=(20,5))
# axs[0].hist(AllGenChi0    , bins=100)
# axs[1].hist(AllGenCEDist  , bins=100)
# axs[2].hist(AllGenRp      , bins=100)
# axs[3].hist(AllGenSDPPhi  , bins=100)
# axs[4].hist(AllGenSDPTheta, bins=100)
# axs[0].set_xlabel('Chi0')
# axs[1].set_xlabel('Core-Eye Distance')
# axs[2].set_xlabel('Rp')
# axs[3].set_xlabel('SDP Phi')
# axs[4].set_xlabel('SDP Theta')
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
# axs[3].grid()
# axs[4].grid()
# plt.tight_layout()
# plt.savefig('Geometry.png')

        

##########################################################################################
# Single Event
# print('File: ', files[0])
# FullPath = os.path.join(DataDir, files[-1])
# print(FullPath)
# for Event in RecEventProvider(FullPath):
#     FdEvent = Event.GetFDEvents()[-1]
    


# print(Event.GetEventId())
# print()
# # Check Geometry
# FdGenGeometry = FdEvent.GetGenGeometry()
# print("Chi0:", FdGenGeometry.GetChi0()*180/np.pi)
# print("Core-Eye Distance:", FdGenGeometry.GetCoreEyeDistance())
# print("Rp:", FdGenGeometry.GetRp())
# # print("SDP:", FdGenGeometry.GetSDP())
# print("SDP Phi:", FdGenGeometry.GetSDPPhi()*180/np.pi)
# print("SDP Theta:", FdGenGeometry.GetSDPTheta()*180/np.pi)
# print("T0:", FdGenGeometry.GetT0())


# # # Check Pixels
# FdRecPixel = FdEvent.GetFdRecPixel()
# # print("Number of Pixels:", FdRecPixel.GetNumberOfPixels())
# # print("Number of Pulsed Pixels:", FdRecPixel.GetNumberOfPulsedPixels())
# # print("Number of SDP Fit Pixels:", FdRecPixel.GetNumberOfSDPFitPixels())
# # print("Number of Spot Trace Pixels:", FdRecPixel.GetNumberOfSpotTracePixels())
# # print("Number of Time Fit Pixels:", FdRecPixel.GetNumberOfTimeFitPixels())
# # print("Number of Trace Pixels:", FdRecPixel.GetNumberOfTracePixels())
# # print("Number of Triggered Pixels:", FdRecPixel.GetNumberOfTriggeredPixels())


