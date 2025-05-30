from Dataset import DatasetContainer
import os

os.system('clear')

print('Loading Dataset')

DirPath = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Data/RawData'

for Run in ['Run010','Run030','Run080','Run090']:
    print(f'Run: {Run}')
    Dataset = DatasetContainer(ExpectedSize=250000)
    Dataset.Load(DirPath,Run)
    DatasetNew = DatasetContainer(ExpectedSize=250000)
    DatasetLength = len(Dataset)
    for i,Event in enumerate(Dataset):
        if Event != None:
            print(f'Processing Event {i}/{DatasetLength} with Id {Event.EventID}',end='\r')
            DatasetNew.add_Event(Event)

    DatasetNew.Save(DirPath,Run)
    print()
