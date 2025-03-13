import numpy as np

# Define some values for telescope positions
COPos_UTM = np.array([445343.8,6114140.0,1712.3])
HEPos_UTM = np.array([445498.14,6114209.8, 1707.34])

COPos_Site = np.array([-31895.75932067,  15026.12801882,    214.90194976])
HEPos_Site = np.array([-31741.12427975,  15095.57420328,    210.54774754])


CO_BackwallAngle = 243.0219        
CO_OpticalAxisPhi   = {1:14.8,2:44.92,3:74.93,4:105.04,5:134.89,6:165.2}
CO_OpticalAxisTheta = {1:16.03,2:14.14,3:16.02,4:16.20,5:16.09,6:16.15 }

HE_BackwallAngle = 273.0
HE_OpticalAxisPhi   = {1:44.45,2:89.87,3:132.83}
HE_OpticalAxisTheta = {1:44.45,2:45.58,3:44.85}


PrintIntermediates = False

# Conversion Function
def ConvertGeometry(CO_SDP_Theta,CO_SDP_Phi,CO_Chi0,CO_Rp,**kwargs):
    if "CS" in kwargs:
        if kwargs["CS"] == 'UTM':
            COpos = COPos_UTM
            HEpos = HEPos_UTM
        elif kwargs["CS"] == 'Site':
            COpos = COPos_Site
            HEpos = HEPos_Site
    else:
        COpos = COPos_Site
        HEpos = HEPos_Site
    # Convert the geometry to axis vector and core position in the cartesian coordinates
    # Core position first
    CO_CoreEyeDist = CO_Rp/np.sin(np.pi-CO_Chi0)
    CO_CoreAzimuth = CO_SDP_Phi + CO_BackwallAngle/180*np.pi - np.pi/2
    if PrintIntermediates:  print('-------->CO_CoreEyeDist:',CO_CoreEyeDist)
    if PrintIntermediates:  print('-------->CO_CoreAzimuth:',CO_CoreAzimuth/np.pi*180)
    
    # Core Position in Coihueco Relative Frame
    CO_Core_X = CO_CoreEyeDist*np.cos(CO_CoreAzimuth)
    CO_Core_Y = CO_CoreEyeDist*np.sin(CO_CoreAzimuth)
    CO_Core_Z = 0
    if PrintIntermediates:  print('-------->CO_Core:',CO_Core_X,CO_Core_Y,CO_Core_Z)
    
    # Axis Vector in Coihueco Relative Frame
    CO_Axis_X = -np.sin(CO_SDP_Theta)*np.cos(CO_Chi0)
    CO_Axis_Y =  np.cos(CO_SDP_Theta)
    CO_Axis_Z = -np.sin(CO_SDP_Theta)*np.sin(CO_Chi0)
    if PrintIntermediates: print('-------->CO_Axis :',CO_Axis_X,CO_Axis_Y,CO_Axis_Z)
    
    # Core_Position in Global Frame_relative to Coihueco
    Core_X = CO_Core_X + COpos[0]
    Core_Y = CO_Core_Y + COpos[1]
    Core_Z = CO_Core_Z + COpos[2]
    # These dont mean much so we dont print then

    # Axis Vector in Global Frame # Rotate the axis vector about vertical by the azimuth angle
    Axis_X = CO_Axis_X*np.cos(CO_CoreAzimuth) - CO_Axis_Y*np.sin(CO_CoreAzimuth)
    Axis_Y = CO_Axis_X*np.sin(CO_CoreAzimuth) + CO_Axis_Y*np.cos(CO_CoreAzimuth)
    Axis_Z = CO_Axis_Z
    
    if PrintIntermediates:  print('-------->Global_Axis:',Axis_X,Axis_Y,Axis_Z)
    # Now that we have core and Axis in the global frame, we can calculate the Heat frame geometry
    # First propagate the core to the HE height
    T = (HEPos[2]-Core_Z)/Axis_Z
    Core_X = Core_X + T*Axis_X
    Core_Y = Core_Y + T*Axis_Y
    Core_Z = HEPos[2]
    HE_Core_X = Core_X - HEPos[0]
    HE_Core_Y = Core_Y - HEPos[1]
    HE_Core_Z = Core_Z - HEPos[2]
    if PrintIntermediates:  print('-------->HE_Core:',HE_Core_X,HE_Core_Y,HE_Core_Z)
    
    # Find new Aziumuth of the core in the HE frame
    HE_CoreAzimuth = np.arctan2(HE_Core_Y,HE_Core_X)
    

    # Rotate the axis by -ve of the backwall angle
    HE_Axis_X = Axis_X*np.cos(-HE_CoreAzimuth) - Axis_Y*np.sin(-HE_CoreAzimuth)
    HE_Axis_Y = Axis_X*np.sin(-HE_CoreAzimuth) + Axis_Y*np.cos(-HE_CoreAzimuth)
    HE_Axis_Z = Axis_Z
    # Now we can calculate the angles
    if PrintIntermediates:  print('-------->HE_Axis:',HE_Axis_X,HE_Axis_Y,HE_Axis_Z)

    
    HE_SDP_Theta = np.arccos(HE_Axis_Y)
    
    HE_SDP_Phi = HE_CoreAzimuth - HE_BackwallAngle/180*np.pi + np.pi/2
    if HE_SDP_Phi > np.pi: HE_SDP_Phi -= 2*np.pi
    if HE_SDP_Phi < -np.pi: HE_SDP_Phi += 2*np.pi
    
    HE_Chi0      = np.pi+np.arctan2(HE_Axis_Z,HE_Axis_X)
    
    HE_CoreEyeDist = np.sqrt(HE_Core_X**2 + HE_Core_Y**2)
    if PrintIntermediates:  print('-------->HE_CoreEyeDist:',HE_CoreEyeDist)
    HE_Rp = HE_CoreEyeDist*np.sin(np.pi-HE_Chi0)
    
    # Placeholders
    # HE_SDP_Theta = 0
    # HE_SDP_Phi   = 0
    # HE_Chi0      = 0
    # HE_Rp        = 0
    return HE_SDP_Theta,HE_SDP_Phi,HE_Chi0,HE_Rp
    
# Conversion Function
