import torch
import Dataset

DatasetContainer = Dataset.DatasetContainer



Path_To_Data = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/DatasetFiles/RawData'

primaries = {22:0,2212:1,2004:2,8016:3,26056:4}


for runN,run in enumerate(['Run090']):#,'Run080','Run090']):
    Dataset = DatasetContainer(1)
    Dataset.Load(Path_To_Data,run)
    # print(Dataset._otherData[:,[5,6]].to(int))

    # Only need to operate on 6th 
    for i in range(Dataset._otherData.shape[0]):
        primary = Dataset._otherData[i,8].item()
        Energy  = Dataset._otherData[i,9].item()
        id      = Dataset._otherData[i,7]

        # Id will have 01EU (where EU is event Usage Number)
        # Keep EU, Replace 01 with the values for energy and primary

        primary_number = primaries[primary]
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
        
        # print(f'Primary {primary_number} , Energy {Energy_number}')
        new_id = 1000*Energy_number + 100*primary_number + id%100

        print(f'Event {i} ,Id {id}, Primary {primary} , Energy {Energy}, New Id {new_id}')
        Dataset._otherData[i,7] = new_id
    Dataset.Save(Path_To_Data,run)
    