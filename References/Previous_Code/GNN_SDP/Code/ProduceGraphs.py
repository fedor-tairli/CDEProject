from Dataset import DatasetContainer, GraphDatasetContainer
import os
import torch

# os.system('clear')

print('Loading Dataset')

Dataset = DatasetContainer(ExpectedSize=0)
filesDir = '/remote/tychodata/ftairli/work/Projects/GNN_SDP/Data/RawData'

if False: # Testing
    Dataset.Load(filesDir,'Test')
    SaveName = 'Test'
    # Produce The GraphDataset
    print('Producing GraphDataset')
    GraphDataset = Dataset.ProduceGraphDataset()
    del Dataset
    print('Calculating Graphs')
    GraphDataset.CalculateGraphs(Timing=False)

    print('Saving Graphs')

    GraphDataset.Save(filesDir,SaveName)

    print(f'Shape of _pixelData :{GraphDataset._pixelData.shape}')
    print(f'Shape of _ActivePixels :{GraphDataset._ActivePixels.shape}')
    print('Done')

else:
    for run in ['Run010','Run030','Run080','Run090']:
        Dataset.Load(filesDir,run)
        SaveName = run
        


        # Produce The GraphDataset
        print('Producing GraphDataset')
        GraphDataset = Dataset.ProduceGraphDataset()
        # del Dataset
        print('Calculating Graphs')
        GraphDataset.CalculateGraphs(Timing=False)

        print('Saving Graphs')

        GraphDataset.Save(filesDir,SaveName)

        print(f'Shape of _pixelData :{GraphDataset._pixelData.shape}')
        print(f'Shape of _ActivePixels :{GraphDataset._ActivePixels.shape}')
        print('Done')

