import numpy as np
import os
import torch

os.system('clear')

import warnings
warnings.filterwarnings("ignore")


from adst3 import RecEventProvider, GetDetectorGeometry
# from FD_Reconstruction_Definitions import Find_Pixel_Id, Get_Pixel_Pos_Dict_EBStyle # looks like i didnt use these
from Dataset2 import ProcessingDatasetContainer

######### SETUP #########



def clearout():
    os.system('clear')
clearout()

testfile = False
SavePath = '/remote/tychodata/ftairli/work/Projects/ProfileReconstruction/LastStep'

# Extra Functions

def Unnormalise_Xmax_Energy(Data):
    Data[:,0] = Data[:,0]*70+750
    Data[:,1] = Data[:,1]+19
    return Data


def CalculateStationPosition(EyeCS,StationCS,BackWallAngle):
    EyeCS     = np.array(EyeCS)
    StationCS = np.array(StationCS)
    Delta     = StationCS - EyeCS
    Distance  = np.sqrt(np.sum(Delta**2))
    Theta     = np.arccos(Delta[2]/Distance)
    Phi       = np.arctan2(Delta[1],Delta[0])
    Phi      -= BackWallAngle
    if Phi<0: Phi += 2*np.pi
    return Distance,Theta,Phi
def AdjustPrimaryName(Primary):
    if Primary == 1000026056: return 26056
    if Primary == 1000008016: return 8016
    if Primary == 1000002004: return 2004
    return Primary
               
def FixId(ID,Energy,Primary):
    primaries = {22:0,2212:1,2004:2,8016:3,26056:4}
    primary_number = primaries[Primary]
    if Energy > 18.0 and Energy <= 18.5:
        Energy_number = 1
    elif Energy > 18.5 and Energy <= 19.0:
        Energy_number = 2
    elif Energy > 19.0 and Energy <= 19.5:
        Energy_number = 3
    elif Energy > 19.5 and Energy <= 20.0:
        Energy_number = 4
    else:
        raise ValueError(f'Energy {Energy} not in range')
    # ID will be 9 digits long
    # RunNumber
    # EventGenerationNumber
    # EventGenerationNumber
    # EventGenerationNumber
    # EventGenerationNumber
    # Unused - Replace by Energy  Number
    # Unused - Replace by Primary Number
    # EventUsageNumber
    # EventUsageNumber
    
    # Return 2 halfs of the ID
    first_half = float(str(ID)[:5])
    second_half = 1000*Energy_number + 100*primary_number + ID%100
    return first_half,second_half


def ReadFiles(files,ExpectedSize,DatasetName):
    print('Initialising Data Arrays')

    MetaData  = []
    TruthData = []
    RecData   = []
    InputData = []
    EventIDs  = []




    print('Begining to read files')
    Nevents = 0                # Number of events read
    NeventsWithNoStations = 0  # Number of events with FD Station Missing
    NeventsWithNoEyes     = 0  # Number of events with FD Event Missing
    for filename in files:
        # detGeom = GetDetectorGeometry(filename) # in case its different per file, its same per event for sure
        print()
        for i,ev in enumerate(RecEventProvider(filename,2)):  # Read Mode 0=Full 2=No Traces 1=Only ShowerLevelObservables
            # Progress Bar
            Nevents += 1
            if len(files) == 1 : print(f'Event {Nevents}')
            else:
                if Nevents%1000 == 0:
                    print(f'Event {Nevents}, skipped {NeventsWithNoEyes} events',end='\r')

            
            
            # FdData
            try:
                FdEvent    = ev.GetHottestEye()
            except Exception as e:
                if 'No such eye' in str(e):
                    NeventsWithNoEyes += 1
                    # print(f'No Eye     in event {ev.GetSDEvent().GetEventId()}, Total FdEvents Missing = {NeventsWithNoEyes} ')
                    continue
                else:
                    raise e
            
            # Get Objects
            SDEvent     = ev.GetSDEvent()
            FdStations  = FdEvent.GetStationVector()
            # FdRecPixel  = FdEvent.GetFdRecPixel()
            GenShower   = ev.GetGenShower()
            RecShower   = FdEvent.GetFdRecShower()
            # RecGeometry = FdEvent.GetFdRecGeometry()
            # GenGeometry = FdEvent.GetGenGeometry()
            
            # ReferenceTime = 5*10**8 # For simulations its always the same, for real data, make sure the second isnt crossed over

            # Find Hottest Station
            HottestStation = None
            for station in FdStations:
                if not station.IsHybrid(): continue
                HottestStation = station
                break
            if HottestStation == None:
                NeventsWithNoStations += 1
                # print(f'No Station in event {ev.GetSDEvent().GetEventId()}, Total Stations Missing = {NeventsWithNoStations} ')
                continue





            # MetaData 
            #   PrimaryID
            #   EventID_1/2
            #   EventID_2/2
            #   Zenith Angle

            # TruthData
            #   LogE
            #   Xmax

            # InputData
            #   EnergyDeposit
            #   DepthOfDeposit

            LogE = np.log10(GenShower.GetEnergy())
            Primary = AdjustPrimaryName(GenShower.GetPrimary())
            EventID_1,EventID_2 = FixId(SDEvent.GetEventId(),LogE,Primary)
            EventID = int(EventID_1)*100000 + int(EventID_2)
            EventIDs.append(EventID)
            Zenith = GenShower.GetCosZenith()
            Xmax = GenShower.GetXmaxInterpolated()

            RecXmax = RecShower.GetXmax()
            RecEnergy = np.log10(RecShower.GetEnergy())

            EnergyDeposit  = np.array(RecShower.GetEnergyDeposit())
            DepthOfDeposit = np.array(RecShower.GetDepth())

            # Append the values
            MetaData.append([Primary,Zenith])
            TruthData.append([Xmax,LogE])
            RecData.append([RecXmax,RecEnergy])
            
            # Process the input values
            DepthsOfDeposit_normalised = DepthOfDeposit/5
            bin_edges = np.arange(0,401,1)
            counts,_ = np.histogram(DepthsOfDeposit_normalised,bins=bin_edges)
            total_energy_deposit,_ = np.histogram(DepthsOfDeposit_normalised,bins=bin_edges,weights=EnergyDeposit)
            average_energy_deposit = np.where(total_energy_deposit>0,total_energy_deposit/counts,0)
            InputData.append(torch.tensor(average_energy_deposit))

            # break
        print('-------------------------------File Done---------------------------------')
    print('Done Reading Files')
    print(f'Number of Events Read             = {Nevents}')
    print(f'Number of Events with No Eyes     = {NeventsWithNoEyes}')
    print(f'Number of Events with No Stations = {NeventsWithNoStations}')
    print('Saving Data')

    print('MetaData')
    MetaData = torch.tensor(MetaData)
    print('MetaData Shape = ',MetaData.shape)
    print('TruthData')
    TruthData = torch.tensor(TruthData)
    RecData = torch.tensor(RecData)
    print('TruthData Shape = ',TruthData.shape)
    print('RecData Shape = ',RecData.shape)
    # Normalise the TruthData
    TruthData[:,0] = TruthData[:,0]/70-750
    TruthData[:,1] = TruthData[:,1]-19
    RecData[:,0] = RecData[:,0]/70-750
    RecData[:,1] = RecData[:,1]-19
    
    print('InputData')
    InputData = torch.stack(InputData)
    # Normalise input data
    InputData = torch.log10(torch.clip(InputData,min=0)+1)

    print('InputData Shape = ',InputData.shape)
    print('Finished the Data Reading')

    ProcDS  = ProcessingDatasetContainer()
    ProcDS._Main = [InputData]
    ProcDS._Truth = TruthData
    ProcDS._Rec   = RecData
    ProcDS._MetaData = MetaData
    ProcDS._Aux = torch.zeros_like(TruthData)
    ProcDS._EventIds = EventIDs

    ProcDS.Unnormalise_Truth =  'FunctionDidntWork'
    ProcDS.Truth_Keys = ['Xmax','LogE']
    ProcDS.Truth_Units = ['g/cm^2','']

    ProcDS.Save(SavePath,Name=DatasetName)

    print(f'Saved the Data at {SavePath}/{DatasetName}.pt')



if __name__ == '__main__':
    if testfile:
        files = ['/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/Selected_Events_Extended.root']
        ReadFiles(files,ExpectedSize=30/100000,DatasetName = 'Test') # Expected size is events/1e6
    else:
        print('Going Through All of the Runs')
        files = []
        for RUN in ['Run010','Run030','Run080','Run090']:
            print(f'Run = {RUN}')
            dir = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
            energy = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
            mass   = ['helium','iron','oxygen','proton'] #,'photon']
            for e in energy:
                # Construct the filenames array
                for m in mass:
                    sub_e = e.replace('.','')
                    filename = f'{dir}/{e}/{m}/SIB23c_{sub_e}_{m}_Hybrid_CORSIKA76400_{RUN}.root'
                    files.append(filename)
        ReadFiles(files,1e6,'AllRuns')
            

print()
print()
print()
# os.system('python3.9 ProduceGraphs.py')