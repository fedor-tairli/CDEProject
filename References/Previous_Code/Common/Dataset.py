##########################################################
#                                                         #
#          This File Defines the Dataset Object           #
#                                                         #
###########################################################


import torch
import os
from DatasetAuxStructures import GetProcEvent
import time

# TODO:  This might be wrong, cause some values have been added
# Pixel Observables : ID, TelID, EyeID, Status, Charge, Chi, PulseStart, PulseCentroid, PulseStop, PixelPhi, PixelTheta, PixelTimeOffset, PixelTrace
# Station Data      : TotalSignal, Time, X, Y, Z
# Shower Info       : UniqueIDFirstHalf, UniqueIDSecondHalf, Primary
# Gen Event Info    : UniqueID, Primary, LogE, CosZenith, Xmax, dEdXmax
# Rec Event Info    : LogE, CosZenith, Xmax, dEdXmax, UspL, UspR
# Gen    Geometry   : SDPPhi, SDPTheta, Chi0, Rp, T0
# Rec    Geometry   : SDPPhi, SDPTheta, Chi0, Rp, T0, SDPPhiError, SDPThetaError, Chi0Error, RpError, T0Error



# To be Improved by predefining expected dtype?
# This is basically only internal for adding pixels i think
class PixelContainer:
    '''Class to hold the pixel observables for a single pixel
    '''
    def __init__(self):
        self.ID              = None
        self.TelID           = None
        self.EyeID           = None
        self.Status          = None
        self.Charge          = None
        self.Chi             = None
        self.PulseStart      = None
        self.PulseCentroid   = None
        self.PulseStop       = None
        self.PixelPhi        = None
        self.PixelTheta      = None
        self.PixelTimeOffset = None
        self.PixelTrace      = None

    def add_ID(self,ID):
        assert ID <= 440, 'Tel Pixel ID not Eye Pixel ID'
        self.ID = ID
    def add_TelID(self,TelID):
        self.TelID = TelID
    def add_EyeID(self,EyeID):
        self.EyeID = EyeID
    def add_Status(self,Status):
        self.Status = Status
    def add_Charge(self,Charge):
        self.Charge = Charge
    def add_Chi(self,Chi):
        self.Chi = Chi
    def add_PulseStart(self,PulseStart):
        self.PulseStart = PulseStart
    def add_PulseCentroid(self,PulseCentroid):
        self.PulseCentroid = PulseCentroid
    def add_PulseStop(self,PulseStop):
        self.PulseStop = PulseStop
    def add_SDP(self,n):
        '''Usage : calculates theta and phi for a pixle given pixel id'''
        j = (n-1)%22+1
        i = ((n-j)//22+1)*2 if j%2 else ((n-j)//22)*2+1
        alpha = (j-11.5)*1.5/180*torch.pi*torch.sqrt(3)/2
        alpha0= 16/180*torch.pi
        beta  = (i-20)*0.5/180*torch.pi
        delta = torch.arcsin(torch.sin(alpha+alpha0)*torch.cos(beta))
        phi   = torch.arcsin(torch.sin(beta)/torch.cos(delta))
        self.PixelPhi = phi
        self.PixelTheta = 90-delta
    def add_PixelPhi(self,PixelPhi):
        self.PixelPhi = PixelPhi
    def add_PixelTheta(self,PixelTheta):
        self.PixelTheta = PixelTheta
    def add_PixelTimeOffset(self,TimeOffset):
        self.PixelTimeOffset = TimeOffset
    def add_PixelTrace(self,Trace):
        if type(Trace) != torch.Tensor:
            Trace = torch.tensor(Trace)
        if len(Trace) == 2000: # Sum Every pair of bins
            assert self.TelID in [7,8,9] or self.EyeID == 5, f'Expected 2000 bins for TelID {self.TelID} or EyeID {self.EyeID}'
            Trace = Trace[::2] + Trace[1::2]
        self.PixelTrace = Trace




class PixelObservablesContainer:
    '''Class to hold the pixel observables for a single event
    '''
    def __init__(self):
        self._ID              = []
        self._TelID           = []
        self._EyeID           = []
        self._Status          = []
        self._Charge          = []
        self._Chi             = []
        self._PulseStart      = []
        self._PulseCentroid   = []
        self._PulseStop       = []
        self._PixelPhi        = []
        self._PixelTheta      = []
        self._PixelTimeOffset = []
        self._PixelTrace      = []

    def addPixel(self,Pixel:PixelContainer):
        self._ID             .append(torch.tensor(Pixel.ID))     # has to be [1,440] 
        self._TelID          .append(torch.tensor(Pixel.TelID))  # has to be [0,9]   # including heat
        self._EyeID          .append(torch.tensor(Pixel.EyeID))  # has to be [1,6]   
        self._Status         .append(torch.tensor(Pixel.Status))
        self._Charge         .append(torch.tensor(Pixel.Charge))
        self._Chi            .append(torch.tensor(Pixel.Chi))
        self._PulseStart     .append(torch.tensor(Pixel.PulseStart))
        self._PulseCentroid  .append(torch.tensor(Pixel.PulseCentroid))
        self._PulseStop      .append(torch.tensor(Pixel.PulseStop))
        self._PixelPhi       .append(torch.tensor(Pixel.PixelPhi))
        self._PixelTheta     .append(torch.tensor(Pixel.PixelTheta))
        self._PixelTimeOffset.append(torch.tensor(Pixel.PixelTimeOffset))
        self._PixelTrace     .append(Pixel.PixelTrace)

    def __len__(self):
        return len(self._ID)
    
    def GetPixelData(self):
        # shape = torch.stack((torch.tensor(self._ID),torch.tensor(self._TelID),torch.tensor(self._EyeID),torch.tensor(self._Status),torch.tensor(self._Charge),torch.tensor(self._Chi),torch.tensor(self._PulseStart),torch.tensor(self._PulseCentroid),torch.tensor(self._PulseStop),torch.tensor(self._PixelPhi),torch.tensor(self._PixelTheta))).shape
        # print(f'PixelData Shape is : {shape}, with N pixels : {len(self)}')
        return torch.stack((torch.tensor(self._ID),torch.tensor(self._TelID),torch.tensor(self._EyeID),torch.tensor(self._Status),torch.tensor(self._Charge),torch.tensor(self._Chi),torch.tensor(self._PulseStart),torch.tensor(self._PulseCentroid),torch.tensor(self._PulseStop),torch.tensor(self._PixelPhi),torch.tensor(self._PixelTheta),torch.tensor(self._PixelTimeOffset))).permute(1,0)

    def GetPixelTrace(self):
        return torch.stack(self._PixelTrace)
    
    def ShowPixels(self):
        '''Print the data on the first 10 (not too much) pixels 
        '''
        # have to loop through each of them to make sure they are 6 characters long
        print(f'            Pixel IDs   : [',end = '')
        for i in range(min(10,len(self))):print(str(self._ID           [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            Tel IDs     : [',end = '')
        for i in range(min(10,len(self))):print(str(self._TelID        [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            Eye IDs     : [',end = '')
        for i in range(min(10,len(self))):print(str(self._EyeID        [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            Status      : [',end = '')
        for i in range(min(10,len(self))):print(str(self._Status       [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            Charge      : [',end = '')
        for i in range(min(10,len(self))):print(str(self._Charge       [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            Chi         : [',end = '')
        for i in range(min(10,len(self))):print(str(self._Chi          [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PulseStart  : [',end = '')
        for i in range(min(10,len(self))):print(str(self._PulseStart   [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PulseCentr  : [',end = '')
        for i in range(min(10,len(self))):print(str(self._PulseCentroid[i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PulseStop   : [',end = '')
        for i in range(min(10,len(self))):print(str(self._PulseStop    [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PixelPhi    : [',end = '')
        for i in range(min(10,len(self))):print(str(self._PixelPhi     [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PixelTheta  : [',end = '')
        for i in range(min(10,len(self))):print(str(self._PixelTheta   [i].item())[:6].ljust(6),end = '   ')
        print(']')
        print(f'            PixelTimeOff: [',end = '')
        for i in range(min(10,len(self))):print(str(self._PixelTimeOffset[i].item())[:6].ljust(6),end = '   ')
        print(']')


    @property
    def ID(self):
        return torch.tensor(self._ID)
    @property
    def TelID(self):
        return torch.tensor(self._TelID)
    @property
    def EyeID(self):
        return torch.tensor(self._EyeID)
    @property
    def Status(self):
        return torch.tensor(self._Status)
    @property
    def Charge(self):
        return torch.tensor(self._Charge)
    @property
    def Chi(self):
        return torch.tensor(self._Chi)
    @property
    def PulseStart(self):
        return torch.tensor(self._PulseStart)
    @property
    def PulseCentroid(self):
        return torch.tensor(self._PulseCentroid)
    @property
    def PulseStop(self):
        return torch.tensor(self._PulseStop)
    @property
    def PixelPhi(self):
        return torch.tensor(self._PixelPhi)
    @property
    def PixelTheta(self):
        return torch.tensor(self._PixelTheta)
    @property
    def PixelTimeOffset(self):
        return torch.tensor(self._PixelTimeOffset)
    @property
    def PixelTrace(self):
        return torch.stack(self._PixelTrace)

    @property
    def HasPixelTrace(self):
        return self._PixelTrace[0] is not None




class EventContainer:
    '''Class to hold Data for a single event to be added to dataset'''
    
    def __init__(self):
        self._PixelObservables = PixelObservablesContainer()
        self._StationData      = torch.zeros(6)
        self._ShowerInfo       = torch.zeros(3)
        self._GenEventInfo     = torch.zeros(4)
        self._RecEventInfo     = torch.zeros(6)
        self._GenGeometry      = torch.zeros(6)
        self._RecGeometry      = torch.zeros(6)
        self._nOtherData       = 0  #track how many appends ive made to other data

    # Add Station Information
    def add_Station_Data(self,Data:torch.tensor):
        assert len(Data) == 6, f'Expected 5 values for Station Data, got {len(Data)}'
        self._StationData = Data
        self._nOtherData += 6
    def add_Station_TotalSignal(self,TotalSignal):
        self._StationData[0] = TotalSignal
        self._nOtherData += 1
    def add_Station_Time(self,Time):
        self._StationData[1] = Time
        self._nOtherData += 1
    def add_Station_Position(self,EyeCS,StationCS,BackwallAngle):
        '''XYZ is useless, so we define the position in the same way as pixels
           [Phi,Theta,Distance] In radians and meters
           Inputs are : EyeCS,StationCS,BackwallAngle
        '''
        EyeCS = torch.tensor(EyeCS)
        StationCS = torch.tensor(StationCS)
        # Calculate vector from eye to station
        Delta = StationCS-EyeCS
        # Calculate distance
        Distance = torch.sqrt(torch.sum(Delta**2))
        # Calculate Theta
        Theta = torch.acos(Delta[2]/Distance)
        # Calculate Phi
        Phi = torch.atan2(Delta[1],Delta[0])
        # Account for backwall angle
        Phi = Phi - BackwallAngle
        if Phi<0:
            Phi = Phi + 2*torch.pi
        # Add to Station Data
        self._StationData[2] = Phi
        self._StationData[3] = Theta
        self._StationData[4] = Distance
        self._nOtherData += 3
    def add_Station_Chii(self,Chii):
        '''And Like pixel we store Station Chi_i'''
        self._StationData[5] = Chii
        self._nOtherData += 1


    # Add Shower Info
    def add_Shower_Data(self,Data:torch.tensor):
        assert len(Data) == 3, f'Expected 3 values for Shower Data, got {len(Data)}'
        self._ShowerInfo = Data
        self._nOtherData += 3
    def add_Shower_ID(self,UniqueID):
        # Split the ID into 2 halfs
        self._ShowerInfo[0] = float(str(UniqueID)[:5])
        self._ShowerInfo[1] = float(str(UniqueID)[5:])
        self._nOtherData += 2
    def add_Shower_primary(self,primary):
        '''Must be string or int'''
        if (primary == 'photon') or (primary == 'gamma') or (primary == 22):
            primary = 22
        elif (primary == 'proton') or (primary == 2212):
            primary = 2212
        elif (primary == 'helium') or (primary == 1000002004):
            primary = 2004
        elif (primary == 'oxygen') or (primary == 1000008016):
            primary = 8016
        elif (primary == 'iron') or (primary == 1000026056):
            primary = 26056
        else:
            raise ValueError(f'Unknown primary {primary}')
        self._ShowerInfo[2] = primary
        self._nOtherData += 1
    # points to the Shower primary, in case i call this somewhere
    def add_GenEvent_primary(self,primary):
        self.add_Shower_primary(primary)
    def add_GenEvent_ID(self,UniqueID):
        self.add_Shower_ID(UniqueID)

    # Add Gen Event Info (Keep the ID and primary in the Gen Event, cause im too lazy to change this shit, send them to Shower info)
    def add_GenEvent_Data(self,Data:torch.tensor):
        assert len(Data) == 4, f'Expected 4 values for Gen Event Info, got {len(Data)}'
        self._GenEventInfo = Data
        self._nOtherData += 4
    def add_GenEvent_logE(self,logE):
        self._GenEventInfo[0] = logE
        self._nOtherData += 1
    def add_GenEvent_cosZenith(self,cosZenith):
        self._GenEventInfo[1] = cosZenith
        self._nOtherData += 1
    def add_GenEvent_Xmax(self,Xmax):
        self._GenEventInfo[2] = Xmax
        self._nOtherData += 1
    def add_GenEvent_dEdXmax(self,dEdXmax):
        self._GenEventInfo[3] = dEdXmax
        self._nOtherData += 1
    
    # Add Rec Event Info
    def add_RecEvent_Data(self,Data:torch.tensor):
        assert len(Data) == 6, f'Expected 6 values for Rec Event Info, got {len(Data)}'
        self._RecEventInfo = Data
        self._nOtherData += 6
    def add_RecEvent_logE(self,logE):
        self._RecEventInfo[0] = logE
        self._nOtherData += 1
    def add_RecEvent_cosZenith(self,cosZenith):
        self._RecEventInfo[1] = cosZenith
        self._nOtherData += 1
    def add_RecEvent_Xmax(self,Xmax):
        self._RecEventInfo[2] = Xmax
        self._nOtherData += 1
    def add_RecEvent_dEdXmax(self,dEdXmax):
        self._RecEventInfo[3] = dEdXmax
        self._nOtherData += 1
    def add_RecEvent_UspL(self,UspL):
        self._RecEventInfo[4] = UspL
        self._nOtherData += 1
    def add_RecEvent_UspR(self,UspR):
        self._RecEventInfo[5] = UspR
        self._nOtherData += 1

    # Add Gen Geometry
    def add_GenGeometry_Data(self,Data:torch.tensor):
        assert len(Data) == 6, f'Expected 6 values for Gen Geometry, got {len(Data)}'
        self._GenGeometry = Data
        self._nOtherData += 6
    def add_GenGeometry_SDPPhi(self,SDPPhi):
        self._GenGeometry[0] = SDPPhi
        self._nOtherData += 1
    def add_GenGeometry_SDPTheta(self,SDPTheta):
        self._GenGeometry[1] = SDPTheta
        self._nOtherData += 1
    def add_GenGeometry_Chi0(self,Chi0):
        self._GenGeometry[2] = Chi0
        self._nOtherData += 1
    def add_GenGeometry_Rp(self,Rp):
        self._GenGeometry[3] = Rp
        self._nOtherData += 1
    def add_GenGeometry_T0(self,T0):
        self._GenGeometry[4] = T0
        self._nOtherData += 1
    def add_GenGeometry_CoreEyeDistance(self,CoreEyeDist):
        self._GenGeometry[5] = CoreEyeDist
        self._nOtherData += 1

    # Add Rec Geometry
    def add_RecGeometry_Data(self,Data:torch.tensor):
        assert len(Data) == 6, f'Expected 6 values for Rec Geometry, got {len(Data)}'
        self._RecGeometry = Data
        self._nOtherData += 6
    def add_RecGeometry_SDPPhi(self,SDPPhi):
        self._RecGeometry[0] = SDPPhi
        self._nOtherData += 1
    def add_RecGeometry_SDPTheta(self,SDPTheta):
        self._RecGeometry[1] = SDPTheta
        self._nOtherData += 1
    def add_RecGeometry_Chi0(self,Chi0):
        self._RecGeometry[2] = Chi0
        self._nOtherData += 1
    def add_RecGeometry_Rp(self,Rp):
        self._RecGeometry[3] = Rp
        self._nOtherData += 1
    def add_RecGeometry_T0(self,T0):
        self._RecGeometry[4] = T0
        self._nOtherData += 1
    def add_RecGeometry_CoreEyeDistance(self,CoreEyeDist):
        self._RecGeometry[5] = CoreEyeDist
        self._nOtherData += 1
    # Errors in Rec Geometry
    # def add_RecGeometry_SDPPhiError(self,SDPPhiError):
    #     self._RecGeometry[5] = SDPPhiError
    #     self._nOtherData += 1
    # def add_RecGeometry_SDPThetaError(self,SDPThetaError):
    #     self._RecGeometry[6] = SDPThetaError
    #     self._nOtherData += 1
    # def add_RecGeometry_Chi0Error(self,Chi0Error):
    #     self._RecGeometry[7] = Chi0Error
    #     self._nOtherData += 1
    # def add_RecGeometry_RpError(self,RpError):
    #     self._RecGeometry[8] = RpError
    #     self._nOtherData += 1
    # def add_RecGeometry_T0Error(self,T0Error):
    #     self._RecGeometry[9] = T0Error
    #     self._nOtherData += 1
    
    # Add Pixels    
    def add_Pixel(self,Pixel:PixelContainer):
        self._PixelObservables.addPixel(Pixel)
    
    def add_Pixel_All(self,PixelObservatbles:PixelObservablesContainer):
        self._PixelObservables = PixelObservatbles

    @property
    def StationData(self):
        return self._StationData
    @property
    def ShowerInfo(self):
        return self._ShowerInfo
    @property
    def GenEventInfo(self):
        return self._GenEventInfo
    @property
    def RecEventInfo(self):
        return self._RecEventInfo
    @property
    def GenGeometry(self):
        return self._GenGeometry
    @property
    def RecGeometry(self):
        return self._RecGeometry
    @property
    def PixelObservables(self):
        return self._PixelObservables
    @property
    def nOtherData(self):
        return self._nOtherData

    def ShowEvent(self, ShowPixels = False):
        '''Prints the entire thing for checking whats happening
        '''
        print('Event Data ------------------------------------------------------------')
        print('Station Data')
        print(f'            TotalSignal : {str(self._StationData[0].item())[:6].ljust(6)} (VEM)')
        print(f'            Time        : {str(self._StationData[1].item())[:6].ljust(6)} (ns)')
        print(f'            Phi         : {str(self._StationData[2].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Theta       : {str(self._StationData[3].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Distance    : {str(self._StationData[4].item())[:6].ljust(6)} (m)')
        print('Shower Info')
        UniqueID = str(int(self._ShowerInfo[0].item()))+'{:04}'.format(int(self._ShowerInfo[1].item()))
        print(f'            Unique ID   : {UniqueID}')
        print(f'            Primary     : {self.PrimaryName}')
        print('Gen Event Info')
        print(f'            logE        : {str(self._GenEventInfo[0].item())[:6].ljust(6)} ([eV])')
        print(f'            cosZenith   : {str(self._GenEventInfo[1].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Xmax        : {str(self._GenEventInfo[2].item())[:6].ljust(6)} (g/cm^2)')
        print(f'            dEdXmax     : {str(self._GenEventInfo[3].item())[:6].ljust(6)} (PeV/g/cm^2)')
        print('Rec Event Info')
        print(f'            logE        : {str(self._RecEventInfo[0].item())[:6].ljust(6)} ([eV])')
        print(f'            cosZenith   : {str(self._RecEventInfo[1].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Xmax        : {str(self._RecEventInfo[2].item())[:6].ljust(6)} (g/cm^2)')
        print(f'            dEdXmax     : {str(self._RecEventInfo[3].item())[:6].ljust(6)} (PeV/g/cm^2)')
        print(f'            UspL        : {str(self._RecEventInfo[4].item())[:6].ljust(6)} (g/cm^2)')
        print(f'            UspR        : {str(self._RecEventInfo[5].item())[:6].ljust(6)} ()')
        print('Gen Geometry')
        print(f'            SDPPhi      : {str(self._GenGeometry[0].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            SDPTheta    : {str(self._GenGeometry[1].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Chi0        : {str(self._GenGeometry[2].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Rp          : {str(self._GenGeometry[3].item())[:6].ljust(6)} (m)')
        print(f'            T0          : {str(self._GenGeometry[4].item())[:6].ljust(6)} (ns)')
        print(f'            CoreEyeDist : {str(self._GenGeometry[5].item())[:6].ljust(6)} (m)')
        print('Rec Geometry')
        print(f'            SDPPhi      : {str(self._RecGeometry[0].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            SDPTheta    : {str(self._RecGeometry[1].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Chi0        : {str(self._RecGeometry[2].item()*180/torch.pi)[:6].ljust(6)} (deg)')
        print(f'            Rp          : {str(self._RecGeometry[3].item())[:6].ljust(6)} (m)')
        print(f'            T0          : {str(self._RecGeometry[4].item())[:6].ljust(6)} (ns)')
        print(f'            CoreEyeDist : {str(self._RecGeometry[5].item())[:6].ljust(6)} (m)')
        print('Pixel Observables')
        if ShowPixels:
            self.PixelObservables.ShowPixels()
        else:
            print(f'            Npixels     : {str(len(self.PixelObservables))[:6].ljust(6)}')

    def __str__(self):
        return f'Event {int(self._ShowerInfo[0].item())}{int(self._ShowerInfo[1].item())} with {len(self.PixelObservables)} pixels'
    
    
    
    #### Obtain Data ####
    # Meta Data
    @property
    def PrimaryName(self):
        if self._ShowerInfo[2] == 22:
            return 'photon'
        elif self._ShowerInfo[2] == 2212:
            return 'proton'
        elif self._ShowerInfo[2] == 2004:
            return 'helium'
        elif self._ShowerInfo[2] == 8016:
            return 'oxygen'
        elif self._ShowerInfo[2] == 26056:
            return 'iron'
    @property
    def PrimaryID(self):
        ID = self._ShowerInfo[2]
        if id in [2004,8016,26056]:
            ID = ID+1000000000
        return ID
    @property
    def EventID(self):
        return int(str(int(self._ShowerInfo[0].item()))+'{:04}'.format(int(self._ShowerInfo[1].item())))
    # Station Data
    @property
    def StationSignal(self):
        return self._StationData[0]
    @property
    def StationTime(self):
        return self._StationData[1]
    @property
    def StationPhi(self):
        return self._StationData[2]/torch.pi*180
    @property
    def StationTheta(self):
        return self._StationData[3]/torch.pi*180
    @property
    def StationDistance(self):
        return self._StationData[4]
    @property
    def StationChii(self):
        return self._StationData[5]
    
    
    # Gen Data
    @property
    def GenLogE(self):
        return self._GenEventInfo[0]
    @property
    def GenCosZenith(self):
        return self._GenEventInfo[1]
    @property
    def GenXmax(self):
        return self._GenEventInfo[2]
    @property
    def Gen_dEdXmax(self):
        return self._GenEventInfo[3]
    # Rec Data
    @property
    def RecLogE(self):
        return self._RecEventInfo[0]
    @property
    def RecCosZenith(self):
        return self._RecEventInfo[1]
    @property
    def RecXmax(self):
        return self._RecEventInfo[2]
    @property
    def Rec_dEdXmax(self):
        return self._RecEventInfo[3]
    @property
    def RecUspL(self):
        return self._RecEventInfo[4]
    @property
    def RecUspR(self):
        return self._RecEventInfo[5]
    # Gen Geometry
    @property
    def GenSDPPhi(self):
        return self._GenGeometry[0]
    @property
    def GenSDPTheta(self):
        return self._GenGeometry[1]
    @property
    def GenChi0(self):
        return self._GenGeometry[2]
    @property
    def GenRp(self):
        return self._GenGeometry[3]
    @property
    def GenT0(self):
        return self._GenGeometry[4]
    @property
    def GenCoreEyeDistance(self):
        return self._GenGeometry[5]
    # Rec Geometry
    @property
    def RecSDPPhi(self):
        return self._RecGeometry[0]
    @property
    def RecSDPTheta(self):
        return self._RecGeometry[1]
    @property
    def RecChi0(self):
        return self._RecGeometry[2]
    @property
    def RecRp(self):
        return self._RecGeometry[3]
    @property
    def RecT0(self):
        return self._RecGeometry[4]
    @property
    def RecCoreEyeDistance(self):
        return self._RecGeometry[5]
    



class DatasetContainer():
    '''Class to hold the dataset for the GNN
    '''
    def __init__(self,ExpectedSize,ExpectTraces = False):
        # Meta Values
        self._AveragePixelsPerEvent = 30
        self._ExpectedSize = ExpectedSize
        self._Nevents = 0
        self._Npixels = 0
        self._NpixelMeasurements = 12
        self._Expected_nOtherData = 31  # Updated to include station data
        self._IDs = None
        
        self.HasTraces = ExpectTraces # Check for building the Dataset

        # Actual Data
        self._pixelData         = torch.zeros(ExpectedSize*self._AveragePixelsPerEvent,self._NpixelMeasurements) # (PixelID,TelID,EyeID, Data...)
        if self.HasTraces:  
            self._pixelTraces   = torch.zeros(ExpectedSize*self._AveragePixelsPerEvent,1000) # (Trace)
        self._EventPixelPosition= torch.zeros(ExpectedSize,2,dtype=torch.int64) # (Start,Stop)
        self._otherData         = torch.zeros(ExpectedSize,self._Expected_nOtherData)                                
        
        # Other Data Has indices Station : [0,1,2,3,4,5]
        #                     ShowerInfo : [6,7,8]
        #                       GenEvent : [9,10,11,12]
        #                       RecEvent : [13,14,15,16,17,18]
        #                    GenGeoemtry : [19,20,21,22,23,24]
        #                    RecGeometry : [25,26,27,28,29,30]



    def add_Event(self, Event:EventContainer):
        '''Add an event to the dataset
        '''
        assert Event.nOtherData == self._Expected_nOtherData, f'Event has {Event.nOtherData} other data, expected {self._Expected_nOtherData}'
        if self.HasTraces: assert Event.PixelObservables.HasPixelTrace, 'Expecting traces, but none found'

        # Check if we need to resize
        if self._Nevents == self._ExpectedSize: # add another 100 empty events
            self._ExpectedSize += 100
            self._pixelData          = torch.cat((self._pixelData          ,torch.zeros(100*self._AveragePixelsPerEvent,self._NpixelMeasurements)))
            if self.HasTraces:
                self._pixelTraces    = torch.cat((self._pixelTraces        ,torch.zeros(100*self._AveragePixelsPerEvent,1000)))
            self._EventPixelPosition = torch.cat((self._EventPixelPosition ,torch.zeros(100,2)),dtype = torch.int64)
            self._otherData          = torch.cat((self._otherData          ,torch.zeros(100,self._Expected_nOtherData)))

        Npix = len(Event.PixelObservables)
        if Npix > 0:
            # Add the other data
            self._otherData[self._Nevents,:] = torch.cat((Event.StationData,Event.ShowerInfo,Event.GenEventInfo,Event.RecEventInfo,Event.GenGeometry,Event.RecGeometry))
            
            # Add the pixel data
            PixelData   = Event.PixelObservables.GetPixelData()
            SortIndex   = torch.argsort(PixelData[:, 7])
            PixelData   = PixelData[SortIndex]
            if self.HasTraces:
                PixelTraces = Event.PixelObservables.GetPixelTrace()
                PixelTraces = PixelTraces[SortIndex]
            
            self._pixelData[self._Npixels:(self._Npixels+Npix),:] = PixelData
            self._EventPixelPosition[self._Nevents,:] = torch.tensor([self._Npixels,self._Npixels+Npix])
            if self.HasTraces:
                self._pixelTraces[self._Npixels:(self._Npixels+Npix),:] = PixelTraces
            

            self._Nevents += 1
            self._Npixels += Npix

    def CleanEmpty(self):
        '''Remove the empty events
        '''
        # Check if the last EventPixel Position corresponds to the last pixel
        self._pixelData = self._pixelData[:self._Npixels]
        self._otherData = self._otherData[:self._Nevents]
        self._EventPixelPosition = self._EventPixelPosition[:self._Nevents]
        if self.HasTraces:
            self._pixelTraces = self._pixelTraces[:self._Npixels]

        assert self._EventPixelPosition[self._Nevents-1,1] == self._Npixels, f'Last EventPixelPosition is {self._EventPixelPosition[self._Nevents-1,1]}, expected {self._Npixels}'
        assert self._Npixels == self._pixelData.shape[0], f'Npixels is {self._Npixels}, expected {self._pixelData.shape[0]}'
        

    def __len__(self):
        return self._Nevents
    
    def Save(self,Path,Name):
        '''Save the dataset to a file {Path}/{Name}_dataValueName.pt
        '''
        self.CleanEmpty()
        # Check if the directory exists
        if not os.path.exists(Path):
            os.makedirs(Path)
        # Save the DataTensors individually
        torch.save(self._pixelData          ,Path+f'/{Name}_pixelData.pt')
        torch.save(self._otherData          ,Path+f'/{Name}_otherData.pt')
        torch.save(self._EventPixelPosition ,Path+f'/{Name}_EventPixelPosition.pt')
        if self.HasTraces: torch.save(self._pixelTraces        ,Path+f'/{Name}_pixelTraces.pt')
        # Save the meta data
        MetaData = [self._Nevents,self._Npixels]
        torch.save(MetaData,Path+f'/{Name}_MetaData.pt')
        
    def Load(self,Path,Name):
        '''Load the dataset from a file {Path}/{Name}_dataValueName.pt
        '''
        if Path.endswith('/'): # Remove the '/'
            Path = Path[:-1]
        # Load the DataTensors individually
        self._pixelData         = torch.load(Path+f'/{Name}_pixelData.pt')
        self._otherData         = torch.load(Path+f'/{Name}_otherData.pt')
        self._EventPixelPosition= torch.load(Path+f'/{Name}_EventPixelPosition.pt')
        # Load the meta data
        MetaData = torch.load(Path+f'/{Name}_MetaData.pt')
        self._Nevents = MetaData[0]
        self._Npixels = MetaData[1]
        self._Expected_nOtherData = self._otherData.shape[1]
    
    def LoadTraces(self,Path,Name):
        '''Load the traces from a file {Path}/{Name}_dataValueName.pt
        '''
        if Path.endswith('/'):
            Path = Path[:-1]
        self._pixelTraces = torch.load(Path+f'/{Name}_pixelTraces.pt')
        self.HasTraces = True

    def LoadAll(self,Path): #### MIGHT BE BUGGED? I GET ZERO EVENTS IN DATASET WHEN I USE THIS
        '''Loads All Runs from Path
        '''
        raise 'Not Implemented Yet with EventPixelPosition'
        Runs = ['Run010','Run030','Run080','Run090']

        self._ExpectedSize = 1e6
        self._Nevents = 0
        self._Npixels = 0
        if Path.endswith('/'): Path = Path[:-1]
        # Check if all Runs have Been Saved to path
        if os.path.exists(Path+f'/ALL_pixelData.pt'):
            self.Load(Path,'ALL')
        elif os.path.exists(Path+f'/RunALL_pixelData.pt'):
            self.Load(Path,'RunALL')
        else:
            for Run in Runs:
                RUN_pixelData = torch.load(Path+f'/{Run}_pixelData.pt')
                RUN_otherData = torch.load(Path+f'/{Run}_otherData.pt')
                RUN_metaData  = torch.load(Path+f'/{Run}_MetaData.pt')
                self._otherData = torch.cat((self._otherData,RUN_otherData)) # Need not be edited
                # Pixel data has Event Index in 0th 
                RUN_pixelData[:,0] = RUN_pixelData[:,0]+self._Nevents
                self._pixelData = torch.cat((self._pixelData,RUN_pixelData))
                self._Nevents += RUN_metaData[0]
                self._Npixels += RUN_metaData[1]

    def GetEventByID(self,ID):
        assert type(ID) == float or type(ID) == int, f'ID must be a float or int, not {type(ID)}'
        AllIDs = self.GetIDs()
        EventLocation = torch.where(AllIDs == ID)[0] # Event Location should just be the index of the event
        return self.GetEventsByIndex(EventLocation)
    
    def GetIDs(self,save = True):
        if self._IDs!=None:
            return self._IDs
        else:
            firstHalf  = self._otherData[:,6].clone().to(dtype = torch.int32)
            secondHalf = self._otherData[:,7].clone().to(dtype = torch.int32) # Should Be Exactly 4 characters long
            Full       = firstHalf*10000+secondHalf
            if save:   self._IDs  = Full[Full!=0]
            return Full[Full!=0]
        
    def Debug(self,Where=None):
        if Where == None:
            pass
        elif Where == 'ReadingData':
            print(f'Total Events in Set : {self._Nevents}')
        elif Where == 'Testing':
            print(self._otherData.requires_grad)
        else:
            print('Unknown Debug Location')
            print('Use : ReadingData, Testing')

    def GetEventsByIndex(self,Index):
        '''Returns the events at the given index
        '''
        EventLocation = Index # Event Location should just be the index of the event
        Event = EventContainer()
        Event.add_Station_Data(self._otherData[EventLocation,:6].squeeze())
        Event.add_Shower_Data(self._otherData[EventLocation,6:9].squeeze())
        Event.add_GenEvent_Data(self._otherData[EventLocation,9:13].squeeze())
        Event.add_RecEvent_Data(self._otherData[EventLocation,13:19].squeeze())
        Event.add_GenGeometry_Data(self._otherData[EventLocation,19:25].squeeze())
        Event.add_RecGeometry_Data(self._otherData[EventLocation,25:31].squeeze())

        # Get the pixel data
        PixelsStart = self._EventPixelPosition[EventLocation,0].item()
        PixelsStop  = self._EventPixelPosition[EventLocation,1].item()
        for iPix in range(PixelsStart,PixelsStop):
            # Initiate a Pixel container
            Pixel = PixelContainer()
            # Add the pixel data
            Pixel.add_ID             (self._pixelData[iPix,0].item())
            Pixel.add_TelID          (self._pixelData[iPix,1].item())
            Pixel.add_EyeID          (self._pixelData[iPix,2].item())
            Pixel.add_Status         (self._pixelData[iPix,3].item())
            Pixel.add_Charge         (self._pixelData[iPix,4].item())
            Pixel.add_Chi            (self._pixelData[iPix,5].item())
            Pixel.add_PulseStart     (self._pixelData[iPix,6].item())
            Pixel.add_PulseCentroid  (self._pixelData[iPix,7].item())
            Pixel.add_PulseStop      (self._pixelData[iPix,8].item())
            Pixel.add_PixelPhi       (self._pixelData[iPix,9].item())
            Pixel.add_PixelTheta     (self._pixelData[iPix,10].item())
            Pixel.add_PixelTimeOffset(self._pixelData[iPix,11].item())
            if self.HasTraces:
                Pixel.add_PixelTrace (self._pixelTraces[iPix,:])
            # Add the pixel to the event
            Event.add_Pixel(Pixel)
            
        return Event if Event.EventID != 0 else None
    
    def GetEventsIterable(self,HowMany = 1e99):
        '''Returns an iterable of events
        '''
        HowMany = min(HowMany,self._Nevents)
        for i in range(HowMany):
            Event = self.GetEventsByIndex(i)
            if Event == None or Event.EventID ==0 : yield None
            else: yield Event
            
    def __iter__(self):
        return self.GetEventsIterable()

    def ProduceGraphDataset(self): # This one is basically a roundabout way of loading the graph dataset, when no graphs have been made yet
        GraphsDataset = GraphDatasetContainer(ExpectedSize=self._Nevents)
        GraphsDataset.FromDatasetContainer(self._Nevents,self._Npixels,self._NpixelMeasurements,self._Expected_nOtherData,self._IDs,self._pixelData,self._otherData,self._EventPixelPosition,self._pixelTraces)
        GraphsDataset.CleanEmpty()
        return GraphsDataset

    def GetValues(self,ValueName):
        '''Returns all values for any given event vriable
        '''

        ValueNameIndex = {'StationTotalSignal' : 0 ,'StationTime'    : 1 ,'StationPhi'      : 2 ,'StationTheta' : 3 ,'StationDistance' : 4 ,'StationChi'         : 5,\
                            'ShowerUniqueID1'  : 6 ,'ShowerUniqueID2': 7 ,'ShowerPrimaryID' : 8 ,\
                            'GenLogE'          : 9 ,'GenCosZenith'   : 10,'GenXmax'         : 11,'Gen_dEdXmax'  : 12,\
                            'RecLogE'          : 13,'RecCosZenith'   : 14,'RecXmax'         : 15,'Rec_dEdXmax'  : 16,'RecUspL'         : 17,'RecUspR'            : 18,\
                            'GenSDPPhi'        : 19,'GenSDPTheta'    : 20,'GenChi0'         : 21,'GenRp'        : 22,'GenT0'           : 23,'GenCoreEyeDistance' : 24,\
                            'RecSDPPhi'        : 25,'RecSDPTheta'    : 26,'RecChi0'         : 27,'RecRp'        : 28,'RecT0'           : 29,'RecCoreEyeDistance' : 30}
        assert ValueName in ValueNameIndex, f'ValueName {ValueName} not found in {ValueNameIndex.keys()}'
        return self._otherData[:,ValueNameIndex[ValueName]]

    def GetLSTMProcessingDataset(self,GetTraces,GetAux,GetTruths):
        ''' Returns the LSTM Processing Dataset
        Given Inputs are functions that return the values for the LSTM Processing
        '''
        # Initialise the Dataset
        LSTMDataset = LSTMProcessingDatasetContainer()
        # Get values
        TraceInputs = GetTraces(self)
        AuxInputs   = GetAux(self)
        TruthInputs = GetTruths(self)
        # Add the values
        LSTMDataset.AddTraceInputs(TraceInputs)
        LSTMDataset.AddAuxInputs(AuxInputs)
        LSTMDataset.AddTruths(TruthInputs)
        return LSTMDataset




class GraphDatasetContainer(DatasetContainer):
    '''Basically the same thing as the Dataset container, but with ability to hold the graphs ready for processing.
    This is an Intermediate step before producing the actual processing dataset. 
    '''

    def __init__(self,ExpectedSize=100):
        super().__init__(ExpectedSize)
        # 50000 is the starting Size, will be resized when needed 50000 ~ some number of Very Large Events
        
        # Additional Graph-Related Data
        self._ActivePixels      = torch.zeros(50000,1).to(torch.bool) # Holds Active Pixels Mask, should be same shape as PixelData in dim = 0
        self._Edges             = torch.zeros(50000,3).to(torch.int) # Holds EventIndex, ActivePixelIndex1, ActivePixelIndex2
        self._NumberOfEdges     = torch.zeros(500,1).to(torch.int)   # Holds NumberOfEdges, should be the same shape as OtherData in dim = 0
        # MetaData Required for appending
        self._TotalEdges        = 0
        self._TotalGraphs       = 0
        
    def FromDatasetContainer(self,NEvents,NPixels,NpixelMeasurements,Expected_nOtherData,IDs,PixelData,OtherData,EventPixelPosition,NpixelTraces = None):
        '''Loads the data from a DatasetContainer
        '''
        self._Nevents             = NEvents if NEvents != None else OtherData.shape[0]
        self._Npixels             = NPixels if NPixels != None else PixelData.shape[0]
        self._NpixelMeasurements  = NpixelMeasurements
        self._Expected_nOtherData = Expected_nOtherData
        self._IDs                 = IDs
        self._pixelData           = PixelData
        self._otherData           = OtherData
        self._EventPixelPosition  = EventPixelPosition
        if NpixelTraces != None:
            self._pixelTraces     = NpixelTraces
            self.HasTraces        = True

        # Can Reshape ActivePixels to the pixelData size and NumberOfEdges to the Nevents size
        self._ActivePixels        = torch.zeros(NPixels,1).to(torch.bool)
        self._NumberOfEdges       = torch.zeros(NEvents,1).to(torch.int)
        self._TotalActivePixels   = 0 # Redundant, but needed for some functionst to not crash ## TODO: Remove this

    def CalculateGraphs(self,Timing = False):
        '''Calculates the graphs for all the events
        '''
        for iEvent in range(self._Nevents):
            EventID = self.CalculateGraph(iEvent,Timing=Timing)

    def CalculateGraph(self,iEvent,Timing = False):
        '''Calculates the graph for a given event
        '''
        if Timing:
            StartTime = time.time()
        # Get the event by index
        Event = self._GetEventsByIndex(iEvent)
        if Event == None: return
        
        ThisEventPixels = Event.PixelObservables
        PulseDuration   = ThisEventPixels.PulseStop -ThisEventPixels.PulseStart

        # Adjust the Duration based on the telescope
        for iPix in range(len(ThisEventPixels)):
            if ThisEventPixels.TelID[iPix] in [7.0,8.0,9.0] or ThisEventPixels.EyeID[iPix]==5:
                PulseDuration[iPix] *=50
            else:
                PulseDuration[iPix] *=100    

        ThisEventData = torch.cat((ThisEventPixels.PixelPhi.unsqueeze(1),ThisEventPixels.PixelTheta.unsqueeze(1),ThisEventPixels.PulseCentroid.unsqueeze(1),PulseDuration.unsqueeze(1)),dim=1)
        # print(ThisEventData[ThisEventData[:,2].argsort()])
        # Calculate the graph
        
        # UsedPixels,Edges = GetProcEvent(ThisEventData,minchunksize=2)
        UsedPixels,Edges = GetProcEvent(ThisEventData,ReturnPixelsMask=True)

        SizeOfActivePixelsMask          = torch.tensor([len(UsedPixels)])
        NumberOfEdgesInThisEvent        = torch.tensor([len(Edges)])
        
        # Check if we need to resize (Dont need to do the Active Pixels and NumberOfEdges, they are already resized)
        # if self._TotalActivePixels+SizeOfActivePixelsMask          > self._ActivePixels.shape[0]  : self._ActivePixels = torch.cat((self._ActivePixels,torch.zeros(50000,1)))
        # if self._TotalGraphs+1                                     > self._NumberOfEdges.shape[0] : self._NumberOfEdges = torch.cat((self._NumberOfEdges,torch.zeros(100,2)))
        if self._TotalEdges+NumberOfEdgesInThisEvent               > self._Edges.shape[0]         : self._Edges = torch.cat((self._Edges,torch.zeros(50000,3)))
            
        # Add the graph to the dataset
        
        LocationOfNEdges =  iEvent
        self._ActivePixels[self._EventPixelPosition[iEvent,0]:self._EventPixelPosition[iEvent,1]] = UsedPixels.unsqueeze(1)
        self._Edges[self._TotalEdges:(self._TotalEdges+NumberOfEdgesInThisEvent),:]               = torch.cat((torch.ones(NumberOfEdgesInThisEvent,1)*iEvent,Edges),dim=1)
        self._NumberOfEdges[LocationOfNEdges,:]                                                   = torch.tensor([NumberOfEdgesInThisEvent])

        # Update the meta data
        self._TotalEdges        += NumberOfEdgesInThisEvent
        self._TotalGraphs       += 1

        if not Timing and self._TotalGraphs%1000 == 0:
            print(f'Computed Event {iEvent} with ID {Event.EventID} with {len(UsedPixels)} active pixels and {len(Edges)} edges')
        elif Timing:
            print(f'Computed Event {str(iEvent).ljust(7)} with ID {Event.EventID} with {str(len(UsedPixels)).ljust(6)} active pixels and {str(len(Edges)).ljust(6)} edges in {str(time.time()-StartTime)[:6].ljust(6)} seconds')
        return Event.EventID
    
    def CleanEmpty(self):
        '''Remove the empty events
        '''
        # Check if the last EventPixel Position corresponds to the last pixel
        # Remove the empty events
        self._pixelData = self._pixelData[:self._Npixels]
        self._otherData = self._otherData[:self._Nevents]
        self._EventPixelPosition = self._EventPixelPosition[:self._Nevents]
        # self._ActivePixels = self._ActivePixels[:self._TotalActivePixels]
        self._Edges        = self._Edges[:self._TotalEdges]
        # self._NumberOfEdges= self._NumberOfEdges[:self._TotalGraphs]
        if self.HasTraces:
            self._pixelTraces = self._pixelTraces[:self._Npixels]

        assert self._EventPixelPosition[self._Nevents-1,1] == self._Npixels, f'Last EventPixelPosition is {self._EventPixelPosition[self._Nevents-1,1]}, expected {self._Npixels}'
        assert self._Npixels == self._pixelData.shape[0], f'Npixels is {self._Npixels}, expected {self._pixelData.shape[0]}'

    def Save(self,Path,Name):
        '''Save the dataset to a file {Path}/{Name}_dataValueName.pt
        '''
        self.CleanEmpty()
        # Check if the directory exists
        if not os.path.exists(Path):
            os.makedirs(Path)
        # Save the DataTensors individually
        torch.save(self._pixelData,Path+f'/{Name}_pixelData.pt')
        torch.save(self._otherData,Path+f'/{Name}_otherData.pt')
        torch.save(self._EventPixelPosition,Path+f'/{Name}_EventPixelPosition.pt')
        torch.save(self._ActivePixels,Path+f'/{Name}_ActivePixels.pt')
        torch.save(self._Edges,Path+f'/{Name}_Edges.pt')
        torch.save(self._NumberOfEdges,Path+f'/{Name}_NumberOfEdges.pt')
        if self.HasTraces:
            torch.save(self._pixelTraces,Path+f'/{Name}_pixelTraces.pt')
        
        # Save the meta data
        MetaData = [self._Nevents,self._Npixels,self._TotalActivePixels,self._TotalEdges,self._TotalGraphs]
        torch.save(MetaData,Path+f'/{Name}_MetaData.pt')

    def Load(self,Path,Name):
        '''Load the dataset from a file {Path}/{Name}_dataValueName.pt
        '''
        if Path.endswith('/'):
            Path = Path[:-1]
        if Name in ['Test','Run010','Run030','Run080','Run090']:
            # Load the DataTensors individually
            self._pixelData         = torch.load(Path+f'/{Name}_pixelData.pt')
            self._otherData         = torch.load(Path+f'/{Name}_otherData.pt')
            self._EventPixelPosition= torch.load(Path+f'/{Name}_EventPixelPosition.pt')
            self._ActivePixels      = torch.load(Path+f'/{Name}_ActivePixels.pt')
            self._Edges             = torch.load(Path+f'/{Name}_Edges.pt')
            self._NumberOfEdges     = torch.load(Path+f'/{Name}_NumberOfEdges.pt')
            # Load the meta data
            MetaData = torch.load(Path+f'/{Name}_MetaData.pt')
            self._Nevents           = MetaData[0].item() if type(MetaData[0]) == torch.Tensor else MetaData[0]
            self._Npixels           = MetaData[1].item() if type(MetaData[1]) == torch.Tensor else MetaData[1]
            if len(MetaData) > 2:  # TODO :  Update this shit for other cases
                self._TotalActivePixels = MetaData[2].item() if type(MetaData[2]) == torch.Tensor else MetaData[2]
                self._TotalEdges        = MetaData[3].item() if type(MetaData[3]) == torch.Tensor else MetaData[3]
                self._TotalGraphs       = MetaData[4].item() if type(MetaData[4]) == torch.Tensor else MetaData[4]
            else:
                self._TotalActivePixels = None
                self._TotalEdges        = None
                self._TotalGraphs       = None
            self._Expected_nOtherData = self._otherData.shape[1]
        elif Name in ['ALL','RunALL']:
            # Load Run010 first
            # Load the DataTensors individually
            self._pixelData         = torch.load(Path+f'/Run010_pixelData.pt')
            self._otherData         = torch.load(Path+f'/Run010_otherData.pt')
            self._EventPixelPosition= torch.load(Path+f'/Run010_EventPixelPosition.pt')
            self._ActivePixels      = torch.load(Path+f'/Run010_ActivePixels.pt')
            self._Edges             = torch.load(Path+f'/Run010_Edges.pt')
            self._NumberOfEdges     = torch.load(Path+f'/Run010_NumberOfEdges.pt')
            # Load the meta data
            MetaData = torch.load(Path+f'/Run010_MetaData.pt')
            self._Nevents           = MetaData[0].item() if type(MetaData[0]) == torch.Tensor else MetaData[0]
            self._Npixels           = MetaData[1].item() if type(MetaData[1]) == torch.Tensor else MetaData[1]
            self._TotalActivePixels = MetaData[2].item() if type(MetaData[2]) == torch.Tensor else MetaData[2]
            self._TotalEdges        = MetaData[3].item() if type(MetaData[3]) == torch.Tensor else MetaData[3]
            self._TotalGraphs       = MetaData[4].item() if type(MetaData[4]) == torch.Tensor else MetaData[4]
            self._Expected_nOtherData = self._otherData.shape[1]

            # Load the rest
            for Name in ['Run030','Run080','Run090']:
                # Load the DataTensors individually
                self._pixelData         = torch.cat((self._pixelData,torch.load(Path+f'/{Name}_pixelData.pt')))
                self._otherData         = torch.cat((self._otherData,torch.load(Path+f'/{Name}_otherData.pt')))
                New_EventPixelPosition  = torch.load(Path+f'/{Name}_EventPixelPosition.pt')
                New_EventPixelPosition  += self._EventPixelPosition[-1,1]
                self._EventPixelPosition= torch.cat((self._EventPixelPosition,New_EventPixelPosition))
                self._ActivePixels      = torch.cat((self._ActivePixels,torch.load(Path+f'/{Name}_ActivePixels.pt')))
                New_Edges               = torch.load(Path+f'/{Name}_Edges.pt')
                New_Edges[:,0]         += self._TotalGraphs
                self._Edges             = torch.cat((self._Edges,New_Edges))
                self._NumberOfEdges     = torch.cat((self._NumberOfEdges,torch.load(Path+f'/{Name}_NumberOfEdges.pt')))
                # Load the meta data
                MetaData = torch.load(Path+f'/{Name}_MetaData.pt')
                self._Nevents           += MetaData[0].item() if type(MetaData[0]) == torch.Tensor else MetaData[0]
                self._Npixels           += MetaData[1].item() if type(MetaData[1]) == torch.Tensor else MetaData[1]
                self._TotalActivePixels += MetaData[2].item() if type(MetaData[2]) == torch.Tensor else MetaData[2]
                self._TotalEdges        += MetaData[3].item() if type(MetaData[3]) == torch.Tensor else MetaData[3]
                self._TotalGraphs       += MetaData[4].item() if type(MetaData[4]) == torch.Tensor else MetaData[4]



        else: 
            raise f'Unknown Name {Name} Use : Test, Run010, Run030, Run080, Run090 or ALL, RunALL'
    def LoadTraces(self,Path,Name):
        '''Load the traces from a file {Path}/{Name}_dataValueName.pt
        '''
        if Path.endswith('/'):
            Path = Path[:-1]
        if Name in ['Test','Run010','Run030','Run080','Run090']:
            self._pixelTraces = torch.load(Path+f'/{Name}_pixelTraces.pt')
            self.HasTraces = True
        elif Name in ['ALL','RunALL']:
            # Load Run010 first
            self._pixelTraces = torch.load(Path+f'/Run010_pixelTraces.pt')
            # Load the rest
            for Name in ['Run030','Run080','Run090']:
                self._pixelTraces = torch.cat((self._pixelTraces,torch.load(Path+f'/{Name}_pixelTraces.pt')))
            self.HasTraces = True
        else:
            raise f'Unknown Name {Name} Use : Test, Run010, Run030, Run080, Run090 or ALL, RunALL'


    def GetEventsByIndex(self, Index):
        '''Returns the event, Active Pixels and Edges at the given index
        '''

        Event = super().GetEventsByIndex(Index)
        EventStart,EventStop = self._EventPixelPosition[Index]
        
        ActivePixels = self._ActivePixels[EventStart:EventStop].to(torch.bool)
        Edges        = self._Edges[self._Edges[:,0] == Index].to(torch.int)[:,1:]

        return Event,ActivePixels,Edges
    
    def _GetEventsByIndex(self, Index):
        return super().GetEventsByIndex(Index)
        
    
    def __getitem__(self,Index):
        '''Returns the event, Active Pixels and Edges at the given index
        '''
        return self.GetEventsByIndex(Index)

    def GetProcessingDataset(self, GetTruths, GetFeatures,GetStationFeatures, GetTruthsIsNorm, GetFeaturesIsNorm,GetStationFeaturesIsNorm,GetEdgeWeights = None):
        '''Returns a processing dataset
        Takes   GetTruths and GetFeatures which are functions that return 2D tensors of the truth and feature values, given the dataset
                GetTruthsIsNorm and GetFeaturesIsNorm are bools that indicate if the returned truth and feature tensors are normalised
                GetEdgeWeights, if not None, is a function that returns a 2D tensor of the edge weights, given the dataset
        '''
        assert GetTruthsIsNorm         , 'GetTruthsIsNorm   is not implemented yet, must be Normalised'
        assert GetFeaturesIsNorm       , 'GetFeaturesIsNorm is not implemented yet, must be Normalised'
        assert GetStationFeaturesIsNorm, 'GetStationFeaturesIsNorm is not implemented yet, must be Normalised'
        assert not self.HasTraces      , 'Traces are not implemented yet, must be implemented'  # TODO: Implement Traces and the rest of the shit above probably

        # Remove Inactive Pixels First
        BackUpPixelsData = self._pixelData.clone()
        self._pixelData  = self._pixelData[self._ActivePixels.flatten(),:]
        
        
        
        # Get the Event Indices for all events # Dont forget events which would have been removed completely
        
        # EventIndices,EventSizes = torch.unique(self._pixelData[:,0].to(torch.int),return_counts=True)
        # ActualEventSizes        = torch.zeros(TotalNumberOfEvents,dtype = torch.int)
        # ActualEventSizes[EventIndices] = EventSizes.to(torch.int)
        # EventStartIndices       = torch.cumsum(torch.cat([torch.tensor([0]),ActualEventSizes[:-1]]),dim=0)
        # EventFinishIndices      = torch.cumsum(                             ActualEventSizes       ,dim=0)

        # Now that i have Old EventStartIndices and EventFinishIndices, i can recalculate by summing the number of active pixels in each event
        
        ActivePixelsCumSum  = torch.cumsum(self._ActivePixels.flatten(),dim=0)
        EventFinishIndices  = ActivePixelsCumSum[self._EventPixelPosition[:,1]-1].flatten() # -1 because the stop index is used for range and thus 1 too big
        EventStartIndices   = torch.cat([torch.tensor([0]),EventFinishIndices[:-1]])

        # Do the same for the edges
        TotalNumberOfEvents     = len(self._otherData)
        EdgeIndices,EdgeSizes   = torch.unique(self._Edges[:,0].to(torch.int),return_counts=True)
        ActualEdgeSizes         = torch.zeros(TotalNumberOfEvents,dtype = torch.int)
        ActualEdgeSizes[EdgeIndices] = EdgeSizes.to(torch.int)
        EdgeFinishIndices       = torch.cumsum(                             ActualEdgeSizes       ,dim=0)
        EdgeStartIndices        = torch.cat   (( torch.tensor([0])          ,EdgeFinishIndices[:-1])    )

        # Get the Features and Truths
        AllFeatues              = GetFeatures(self)
        AllTruths               = GetTruths(self)
        AllStationFeatures      = GetStationFeatures(self)
        if GetEdgeWeights != None: AllEdgeWeights = GetEdgeWeights(self)

        # Construct the Processing Dataset
        ProcessingDataset = ProcessingDatasetContainer()
        ProcessingDataset.AddFeatures(AllFeatues,'Norm')
        ProcessingDataset.AddTruthVals(AllTruths)
        ProcessingDataset.AddStationFeatures(AllStationFeatures)
        ProcessingDataset.AddEdges(self._Edges[:,1:])
        if GetEdgeWeights != None: ProcessingDataset.AddEdgeWeights(AllEdgeWeights) 
        else: ProcessingDataset.AddEdgeWeights(None)
        ProcessingDataset.AddEventIndices(EventStartIndices,EventFinishIndices)
        ProcessingDataset.AddEdgeIndices(EdgeStartIndices,EdgeFinishIndices)

        # Restore Pixel Data
        self._pixelData = BackUpPixelsData
        return ProcessingDataset




class ProcessingDatasetContainer():

    def __init__(self):
        self.State = 'Build' # Either Build, Train, Validate, Test or Static

        # Normalization States for each chunk of the set [None,Norm,UnNorm] To be set when data is added
        self.NormStateTrain = 'None'
        self.NormStateVal   = 'None'
        self.NormStateTest  = 'None'

        self.NormStateFeatures = 'None'

        self.RandomIter = True

        # Data Containers
        self._StationFeatures = None
        self._Features        = None
        self._Edges           = None
        self._EdgeWeights     = None
        self._TruthVals       = None


        self._EventStartIndices = None
        self._EventFinishIndices= None
        self._EdgeStartIndices  = None
        self._EdgeFinishIndices = None

        # Sets Indices
        self._TrainIndices  = None
        self._ValIndices    = None
        self._TestIndices   = None

    @property
    def TotalLength(self):
        return len(self._TruthVals)
    # Data Handling Functions
    def AddFeatures(self,Features,State):
        '''Adds the features to the dataset
        ''' 
        # Some Checks
        assert self.State == 'Build', f'Cannot add features in state {self.State}'
        assert len(Features.shape) == 2, f'Need 3D tensor for features, not {len(Features.shape)}D' # (Nevents* Nnodes, Nfeatures)
        assert State == 'Norm' , f'Need Normalised Features, not {State}'
        # Add the features
        self._Features = Features
        self.NormStateFeatures = State

    def AddEdges(self,Edges):
        '''Adds the edges to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add edges in state {self.State}'
        assert len(Edges.shape) == 2, f'Need 3D tensor for edges, not {len(Edges.shape)}D' # (3, Edges) (3=EventIndex,EdgeStart,EdgeEnd)
        assert Edges.shape[1]   == 2, f'Need 2 edges per event, not {Edges.shape[1]}'
        # Add the edges
        self._Edges = Edges

    def AddEdgeWeights(self,Weights):
        '''Adds the edge weights
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add edge weights in state {self.State}'
        assert self._Features != None, f'Add Features before adding edge weights'
        assert self._Edges != None, f'Add Edges before adding edge weights'
        
        if Weights == None: # If no weights are given, set them to 1 for non-padding edges
            self._EdgeWeights = torch.ones(self._Edges.shape[0])
            # This results in the same thing if there are no padding edges
        else:
            assert self._Edges.shape == Weights.shape, f'Edge and Weight shapes do not match, {self._Edges.shape} vs {Weights.shape}'

            # Add the edge weights
            self._EdgeWeights = Weights

    def AddTruthVals(self,TruthVals):
        '''Adds the truth values to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add truth values in state {self.State}'
        assert len(TruthVals.shape) == 2, f'Need 2D tensor for truth values, not {len(TruthVals.shape)}D' # (Event,TruthVals)
    
        self._TruthVals = TruthVals

    def AddEventIndices(self,EventStartIndices,EventFinishIndices):
        '''Adds the event indices to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add event indices in state {self.State}'
        assert self._Features != None, f'Add Features before adding event indices'
        assert self._Edges != None, f'Add Edges before adding event indices'
        assert self._TruthVals != None, f'Add Truth Values before adding event indices'
        assert len(EventStartIndices.shape) == 1, f'Need 1D tensor for event start indices, not {len(EventStartIndices.shape)}D'
        assert len(EventFinishIndices.shape) == 1, f'Need 1D tensor for event finish indices, not {len(EventFinishIndices.shape)}D'
        assert EventStartIndices.shape[0] == EventFinishIndices.shape[0], f'Event start and finish indices do not match in length, {EventStartIndices.shape[0]} vs {EventFinishIndices.shape[0]}'
        assert EventStartIndices.shape[0] == self._TruthVals.shape[0], f'Event start indices and truth values do not match in length, {EventStartIndices.shape[0]} vs {self._TruthVals.shape[0]}'
        assert EventFinishIndices[-1]     == len(self._Features), f'Event finish indices and features do not match in length, {EventFinishIndices[-1]} vs {len(self._Features)}'
        self._EventStartIndices = EventStartIndices
        self._EventFinishIndices= EventFinishIndices

    def AddEdgeIndices(self, EdgeStartIndices, EdgeFinishIndices):
        '''Adds the EdgeStart and EdgeEnd indices to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add edge indices in state {self.State}'
        assert self._Features != None, f'Add Features before adding edge indices'
        assert self._Edges != None, f'Add Edges before adding edge indices'
        assert self._TruthVals != None, f'Add Truth Values before adding edge indices'
        assert len(EdgeStartIndices.shape) == 1, f'Need 1D tensor for EdgeStart indices, not {len(EdgeStartIndices.shape)}D'
        assert len(EdgeFinishIndices.shape) == 1, f'Need 1D tensor for EdgeEnd indices, not {len(EdgeFinishIndices.shape)}D'
        assert EdgeStartIndices.shape[0] == self._TruthVals.shape[0], f'EdgeStart indices length does not match the number of events, {EdgeStartIndices.shape[0]} vs {self._TruthVals.shape[0]}'
        assert EdgeFinishIndices[-1]     == len(self._Edges) , f'EdgeFinish indices and edges do not match in length, {EdgeFinishIndices[-1]} vs {len(self._Edges)}'

        self._EdgeStartIndices = EdgeStartIndices
        self._EdgeEndIndices = EdgeFinishIndices

    def AddStationFeatures(self,StationFeatures):
        '''Adds the Station Features to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add station data in state {self.State}'
        assert len(StationFeatures.shape) == 2, f'Need 2D tensor for station features, not {len(StationFeatures.shape)}D'

        self._StationFeatures = StationFeatures

    # Processing Functions
    def AssignIndices(self,distribution = [0.8,0.1,0.1],seed=None, Normalisation = 'Norm'):
        if seed == None: seed = 1234

        # Assert that all values have been set properly
        assert self.NormStateFeatures == 'Norm', f'Features not normalised'
        assert self._Features != None, f'Features not set'
        assert self._Edges != None, f'Edges not set'
        assert self._EdgeWeights != None, f'EdgeWeights not set'
        assert self._TruthVals != None, f'TruthVals not set'

        # Set the seed
        torch.manual_seed(seed)
        TotalLength = self._TruthVals.shape[0]
        # Get a tensor of shuffled indices
        Indices = torch.randperm(TotalLength)
        # Split the indices
        self._TrainIndices = Indices[:int(TotalLength*distribution[0])]
        self._ValIndices   = Indices[int(TotalLength*distribution[0]):int(TotalLength*(distribution[0]+distribution[1]))]
        self._TestIndices  = Indices[int(TotalLength*(distribution[0]+distribution[1])):]
        # Set the state
        self.State = 'Static'

        self.NormStateTrain = Normalisation
        self.NormStateVal   = Normalisation
        self.NormStateTest  = Normalisation

        # Change the dtypes
        self._Features        = self._Features       .to(torch.float32)
        self._Edges           = self._Edges          .to(torch.int32)
        self._EdgeWeights     = self._EdgeWeights    .to(torch.float32)
        self._TruthVals       = self._TruthVals      .to(torch.float32) 
        if self.HasStationFeatures:
            self._StationFeatures = self._StationFeatures.to(torch.float32)


    @property
    def ActiveIndices(self):
        if self.State == 'Static' or self.State == 'Build':
            return torch.arange(self.TotalLength)
        elif self.State == 'Train':
            return self._TrainIndices
        elif self.State == 'Val':
            return self._ValIndices
        elif self.State == 'Test':
            return self._TestIndices
        else:
            raise(f'Unknown State {self.State}')
        
    def __len__(self):
        return len(self.ActiveIndices)

    def __getitem__(self,idx):
        '''Returns One Event from the dataset by index, depending on the state
        '''
        idx = self.ActiveIndices[idx]

        Features    = self._Features   [self._EventStartIndices[idx]:self._EventFinishIndices[idx],:]
        Edges       = self._Edges      [self._EdgeStartIndices [idx]:self._EdgeEndIndices    [idx],:]
        EdgeWeights = self._EdgeWeights[self._EdgeStartIndices [idx]:self._EdgeEndIndices    [idx]]
        TruthVals   = self._TruthVals  [idx,:]
        Edges = Edges.T
        if self._StationFeatures != None:
            StationFeatures = self._StationFeatures[idx,:]
            return idx,Features,Edges,EdgeWeights,TruthVals,StationFeatures
        
        return idx,Features,Edges,EdgeWeights,TruthVals
    
    def __iter__(self):
        '''Returns an iterable of the dataset, depending on the state
        By default the order is randomised
        Iterable over the index, Features, Edges, EdgeWeights and TruthVals
        '''
        if self.RandomIter:
            NewOrder = torch.randperm(len(self))
            Indices = self.ActiveIndices[NewOrder]
        else:
            Indices = self.ActiveIndices
        
        for idx in Indices:
            Features    = self._Features   [self._EventStartIndices[idx]:self._EventFinishIndices[idx],:]
            Edges       = self._Edges      [self._EdgeStartIndices [idx]:self._EdgeEndIndices    [idx],:]
            EdgeWeights = self._EdgeWeights[self._EdgeStartIndices [idx]:self._EdgeEndIndices    [idx]]
            TruthVals   = self._TruthVals  [idx,:]
            Edges = Edges.T # Wrong Shape before
            
            if self._StationFeatures == None:
                yield idx,Features,Edges,EdgeWeights,TruthVals
            else:
                StationFeatures = self._StationFeatures[idx,:]
                yield idx,Features,Edges,EdgeWeights,TruthVals,StationFeatures

    def Save(self, path, name):
        '''Saves the dataset to a file
        '''
        # Check if the directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save all the values
        torch.save(self._Features, path + f'/{name}_Features.pt')
        torch.save(self._Edges, path + f'/{name}_Edges.pt')
        torch.save(self._EdgeWeights, path + f'/{name}_EdgeWeights.pt')
        torch.save(self._TruthVals, path + f'/{name}_TruthVals.pt')
        
        # Save the meta data
        MetaData = [self.NormStateFeatures, self.NormStateTrain, self.NormStateVal, self.NormStateTest]
        torch.save(MetaData, path + f'/{name}_MetaData.pt')
        
        # Save the indices
        Indices = [self._EventStartIndices, self._EventFinishIndices, self._EdgeStartIndices, self._EdgeFinishIndices]
        torch.save(Indices, path + f'/{name}_Indices.pt')

    def Load(self, path, name):
        '''Loads the dataset from a file
        '''
        # Load the DataTensors
        self._Features = torch.load(path + f'/{name}_Features.pt')
        self._Edges = torch.load(path + f'/{name}_Edges.pt')
        self._EdgeWeights = torch.load(path + f'/{name}_EdgeWeights.pt')
        self._TruthVals = torch.load(path + f'/{name}_TruthVals.pt')
        
        # Load the meta data
        MetaData = torch.load(path + f'/{name}_MetaData.pt')
        self.NormStateFeatures = MetaData[0]
        self.NormStateTrain = MetaData[1]
        self.NormStateVal = MetaData[2]
        self.NormStateTest = MetaData[3]
        
        # Load the indices if they exist
        if os.path.exists(path + f'/{name}_Indices.pt'):
            Indices = torch.load(path + f'/{name}_Indices.pt')
            self._EventStartIndices = Indices[0]
            self._EventFinishIndices = Indices[1]
            self._EdgeStartIndices = Indices[2]
            self._EdgeFinishIndices = Indices[3]
        
        # Set the state to Static
        self.State = 'Build'

    @property
    def HasStationFeatures(self):
        return self._StationFeatures != None




class LSTMProcessingDatasetContainer():
    '''A Processing Dataset for LSTM Networks
    '''

    def __init__(self):
        self.State = 'Build' # Either Build, Train, Validate, Test or Static

        self.RandomIter      = True
        self.TraceInputs     = None # The Trace Inputs [N,1000,C] where N is the number of events, 1000 is the number of time steps and C is the number of channels
        self.AuxInputs       = None # The Extra Inputs [N,F]      where N is the number of events and F is the number of features
        self.Truths          = None # The Truth Values [N,T]      where N is the number of events and T is the number of truth values
        
        # Sets Indices
        self._TrainIndices  = None
        self._ValIndices    = None
        self._TestIndices   = None

    @property
    def TotalLength(self):
        assert self.State != 'Build', f'Cannot get the total length in state {self.State}'
        return len(self.Truths)
    
    # Data Handling Functions
    def AddTraceInputs(self,TraceInputs):
        '''Adds the trace inputs to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add trace inputs in state {self.State}'
        assert len(TraceInputs.shape) == 3, f'Need 3D tensor for trace inputs, not {len(TraceInputs.shape)}D'
        # Add the trace inputs
        self.TraceInputs = TraceInputs

    def AddAuxInputs(self,AuxInputs):
        '''Adds the auxiliary inputs to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add auxiliary inputs in state {self.State}'
        assert len(AuxInputs.shape) == 2, f'Need 2D tensor for auxiliary inputs, not {len(AuxInputs.shape)}D'
        # Add the auxiliary inputs
        self.AuxInputs = AuxInputs

    def AddTruths(self,Truths):
        '''Adds the truth values to the dataset
        '''
        # Some Checks
        assert self.State == 'Build', f'Cannot add truth values in state {self.State}'
        assert len(Truths.shape) == 2, f'Need 2D tensor for truth values, not {len(Truths.shape)}D'
        # Add the truth values
        self.Truths = Truths

    # Processing Functions
    def AssignIndices(self,distribution = [0.8,0.1,0.1],seed=1234):
        # Assert that all values have been set properly
        assert self.TraceInputs != None, f'Trace Inputs not added'
        assert self.AuxInputs != None  , f'Auxiliary Inputs not added'
        assert self.Truths != None     , f'Truth Values not added'
        assert self.TraceInputs.shape[0] == self.AuxInputs.shape[0] == self.Truths.shape[0], f'Number of events in Trace Inputs, Auxiliary Inputs and Truth Values do not match'
        
        self.State = 'Static'

        torch.manual_seed(seed)

        Indices = torch.randperm(self.TotalLength)
        # Split the indices
        self._TrainIndices = Indices[:int(self.TotalLength*distribution[0])]
        self._ValIndices   = Indices[int(self.TotalLength*distribution[0]):int(self.TotalLength*(distribution[0]+distribution[1]))]
        self._TestIndices  = Indices[int(self.TotalLength*(distribution[0]+distribution[1])):]
      
    @property
    def ActiveIndices(self):
        if self.State == 'Static' or self.State == 'Build':
            return torch.arange(self.TotalLength)
        elif self.State == 'Train':
            return self._TrainIndices
        elif self.State == 'Val':
            return self._ValIndices
        elif self.State == 'Test':
            return self._TestIndices
        else:
            raise(f'Unknown State {self.State}')

    def __len__(self):
        return len(self.ActiveIndices)
    
    def __getitem__(self,idx):
        '''Returns One Event from the dataset by index, depending on the state
        '''
        idx = self.ActiveIndices[idx]

        TraceInputs = self.TraceInputs[idx,:,:]
        AuxInputs   = self.AuxInputs[idx,:]
        TruthVals   = self.Truths[idx,:]

        return idx,TraceInputs,AuxInputs,TruthVals

    def __iter__(self):
        '''Returns an iterable of the dataset, depending on the state
        By default the order is randomised
        Iterable over the index, TraceInputs, AuxInputs and TruthVals
        '''
        if self.RandomIter:
            NewOrder = torch.randperm(len(self))
            Indices = self.ActiveIndices[NewOrder]
        else:
            Indices = self.ActiveIndices
        
        for idx in Indices:
            TraceInputs = self.TraceInputs[idx,:,:]
            AuxInputs   = self.AuxInputs[idx,:]
            TruthVals   = self.Truths[idx,:]

            yield idx,TraceInputs,AuxInputs,TruthVals

    def Save(self, path, name):
        '''Saves the dataset to a file
        '''
        assert self.State != 'Build', f'Cannot save the dataset in state {self.State}'
        # Check if the directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save all the values
        torch.save(self.TraceInputs, path + f'/{name}_TraceInputs.pt')
        torch.save(self.AuxInputs, path + f'/{name}_AuxInputs.pt')
        torch.save(self.Truths, path + f'/{name}_Truths.pt')


    def Load(self, path, name):
        '''Loads the dataset from a file
        '''
        # Load the DataTensors
        
        if name in ['Run010','Run030','Run080','Run090','Test']:
            self.TraceInputs = torch.load(path + f'/{name}_TraceInputs.pt')
            self.AuxInputs = torch.load(path + f'/{name}_AuxInputs.pt')
            self.Truths = torch.load(path + f'/{name}_Truths.pt')
            
            self.AssignIndices()

        elif name in ['All','ALL','all']:
            # Load Run010 first
            self.TraceInputs = torch.load(path + f'/Run010_TraceInputs.pt')
            self.AuxInputs   = torch.load(path + f'/Run010_AuxInputs.pt')
            self.Truths      = torch.load(path + f'/Run010_Truths.pt')

            # Append other runs
            for run in ['Run030','Run080','Run090']:
                TraceInputs = torch.load(path + f'/{run}_TraceInputs.pt')
                AuxInputs   = torch.load(path + f'/{run}_AuxInputs.pt')
                Truths      = torch.load(path + f'/{run}_Truths.pt')
                self.TraceInputs = torch.cat((self.TraceInputs,TraceInputs))
                self.AuxInputs   = torch.cat((self.AuxInputs,AuxInputs))
                self.Truths      = torch.cat((self.Truths,Truths))
            
            self.AssignIndices()
        else:
            raise(f'Unknown Name {name}')
        
        
        