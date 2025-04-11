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

def FindDataDir():
    current_path = os.getcwd()
    index = current_path.find("CDEs")
    if index == -1:
        raise ValueError("The directory 'CDEs' was not found in the current path.")
    DataDir = os.path.join(current_path[:index + len("CDEs")], "Data")
    
    return DataDir

DataDir = FindDataDir()
# files = os.listdir(DataDir+'/low_cut')
# files = [f for f in files if f.endswith('.root')]
# files = [f for f in files if 'b01' in f]
# files.sort()# Sort files alphabetically
# files = [DataDir+'/low_cut/'+f for f in files]

Dir_inData = '/MC/low/'
subdirs = os.listdir(DataDir+Dir_inData)
subdirs = [subdir for subdir in subdirs if subdir.startswith('b')]
subdirs.sort()
files = []
for subdir in subdirs:
    subfiles = os.listdir(DataDir+Dir_inData+subdir+'/PCGF')
    subfiles = [f for f in subfiles if f.endswith('.root')]
    subfiles.sort()
    files += [DataDir+Dir_inData+subdir+'/PCGF/'+f for f in subfiles]

DetGeom = GetDetectorGeometry(f'{DataDir}/Selected_Events_Extended.root')





# Observatory coordinates
HE_BackwallAngle_1 = 273.2    * np.pi / 180.
HE_BackwallAngle_2 = 273.0    * np.pi / 180.
CO_BackwallAngle_1 = 243.233  * np.pi / 180.
CO_BackwallAngle_2 = 243.0219 * np.pi / 180.


HE_Position = np.array([-31741.12427975, 15095.57420328, 210.54774754])
CO_Position = np.array([-31895.75932067, 15026.12801882, 214.90194976])

HE_EyeThetaZ = 0.005507726051748029
HE_EyePhiZ   = 2.6959186109405837
CO_EyeThetaZ = 0.00552489736362741
CO_EyePhiZ   = 2.69959138256139

def spherical_to_cartesian(theta, phi, mag=1.0):
    x = mag * np.sin(theta) * np.cos(phi)
    y = mag * np.sin(theta) * np.sin(phi)
    z = mag * np.cos(theta)
    return np.array([x, y, z])

HE_EyeZ = spherical_to_cartesian(HE_EyeThetaZ, HE_EyePhiZ)
CO_EyeZ = spherical_to_cartesian(CO_EyeThetaZ, CO_EyePhiZ)

def a_rotate(p, v, a):
    ca = np.cos(a)
    sa = np.sin(a)
    t = 1 - ca
    x=v[0]
    y=v[1]
    z=v[2]

    r = [
        [ca + x*x*t, x*y*t - z*sa, x*z*t + y*sa],
        [x*y*t + z*sa, ca + y*y*t, y*z*t - x*sa],
        [z*x*t - y*sa, z*y*t + x*sa, ca + z*z*t]
    ]

    return np.dot(r, p)

def rotate_around_axis(v, angle, axis):
    # Rodrigues' rotation formula
    # Method 1 - Consistent with a_rotate
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (v * cos_a +
            np.cross(axis, v) * sin_a +
            axis * np.dot(axis, v) * (1 - cos_a))
    # Method 2
    # K = np.array([[0       , -axis[2] , axis[1] ],
    #               [axis[2] ,   0      , -axis[0]],
    #               [-axis[1], axis[0]  ,   0     ]])
    # I = np.eye(3)
    # R = I + np.sin(-angle) * K + (1 - np.cos(-angle)) * np.dot(K, K)

    # v_rotated = np.dot(R, v)
    # return v_rotated
    # Method 3
    # return a_rotate(v, axis, angle)

def rotate_vector_to_eyeCS(v_siteCS, eyeZ_siteCS):
    eyeZ_siteCS = eyeZ_siteCS / np.linalg.norm(eyeZ_siteCS)
    z_eyeCS    = np.array([0, 0, 1])
    cross     = np.cross(z_eyeCS, eyeZ_siteCS)
    # if np.allclose(cross, 0): return v_siteCS  # Already aligned
    dot       = np.dot(z_eyeCS, eyeZ_siteCS)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    axis = cross / np.linalg.norm(cross)

    return rotate_around_axis(v_siteCS, -angle, axis)


def rotate_z(v, angle):
    z = np.array([0, 0, 1])
    return rotate_around_axis(v, angle, z)
    # return rot @ v

def SDP_in_eyeCS(axis_coreCS, core_siteCS, Eye_BackwallAngle_1,Eye_BackwallAngle_2, EyePosition,EyeZ):
    eye_to_core_siteCS = core_siteCS - EyePosition

    eye_to_core_eyeCS = rotate_vector_to_eyeCS(eye_to_core_siteCS, EyeZ)
    eye_to_core_eyeCS = rotate_z(eye_to_core_eyeCS, -Eye_BackwallAngle_1)

    axis_eyeCS = rotate_z(axis_coreCS, -Eye_BackwallAngle_2)
    sdp_eyeCS = np.cross(axis_eyeCS, eye_to_core_eyeCS)
    return sdp_eyeCS / np.linalg.norm(sdp_eyeCS)

def angle_between_two_vectors(v1,v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angle

# now we check if that is true
EvNumber = 0
EyeOfInterest = 5

if EyeOfInterest == 5:
    Eye_BackwallAngle_1 = HE_BackwallAngle_1
    Eye_BackwallAngle_2 = HE_BackwallAngle_2
    Eye_Position = HE_Position
    Eye_EyeZ = HE_EyeZ
elif EyeOfInterest == 6 or EyeOfInterest == 4:
    Eye_BackwallAngle_1 = CO_BackwallAngle_1
    Eye_BackwallAngle_2 = CO_BackwallAngle_2
    Eye_Position = CO_Position
    Eye_EyeZ = CO_EyeZ
else:
    raise ValueError("EyeOfInterest must be 4 or 5 or 6")

All_Angles    = []
All_CoreDists = []
for file in files:
    for Event in RecEventProvider(file):

        #Get FdEvent - specifically HEAT, skip if not found        
        try:
            FdEvent = Event.GetEye(EyeOfInterest)
        except Exception as e:
            if "No such eye" in str(e):
                continue
        
        EvNumber += 1
        # Get Axis and core in Site CS
        Core_siteCS = np.array(Event.GetGenShower().GetCoreSiteCS())
        Axis_coreCS = np.array(Event.GetGenShower().GetAxisCoreCS())
        CoreDistance = np.linalg.norm(Core_siteCS - Eye_Position)
        
        GenSDP = np.array(FdEvent.GetGenGeometry().GetSDP())
        GenSDP /= np.linalg.norm(GenSDP)

        # See if the Transformation is correct
        
        SDP = SDP_in_eyeCS(Axis_coreCS, Core_siteCS, Eye_BackwallAngle_1, Eye_BackwallAngle_2, Eye_Position, Eye_EyeZ)
        # compute angle between the SDPs
        angle = angle_between_two_vectors(SDP, GenSDP)*180/np.pi

        # print(f'Event {EvNumber} Angle between SDP and GenSDP: {angle:.6f} deg')
        if EvNumber % 1000 == 0:
            print(f'Event {EvNumber} Angle between SDP and GenSDP: {angle:.6f} deg')

        All_Angles   .append(angle)
        All_CoreDists.append(CoreDistance)

        if EvNumber > 10000: break
    if EvNumber > 10000: break

print('Mean Angle:',np.mean(All_Angles))
print('68% Angle :' ,np.percentile(All_Angles,68))

All_Angles    = np.array(All_Angles)
All_CoreDists = np.array(All_CoreDists)

plt.figure(figsize=(10, 10))
# sns.jointplot(x=All_CoreDists, y=All_Angles,\
#               kind='hex', cmap='Blues',color='blue',\
#               marginal_kws=dict(bins=50    , fill=True),\
#               joint_kws   =dict(gridsize=50, cmap='Blues',color='blue'))
plt.scatter(All_CoreDists, All_Angles, c='blue', s=1)
plt.xlabel('Core Distance [m]')
plt.ylabel('Angle between GenSDP and SDP [deg]')
plt.title(f'Angle between GenSDP and SDP for Eye {EyeOfInterest}')
plt.savefig("Angle_between_SDPs_vs_Core_Distance.png")


        
