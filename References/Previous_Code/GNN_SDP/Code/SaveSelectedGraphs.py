import torch
import os
from Dataset import DatasetContainer, GraphDatasetContainer
import matplotlib.pyplot as plt


os.system('clear')

print('Loading Dataset')

GraphDataset = GraphDatasetContainer(ExpectedSize=1000)

GraphDataset.Load('/remote/tychodata/ftairli/work/Projects/GNN_SDP/Data/RawData','Test')

print(f'Dataset Length: {len(GraphDataset)}')


for i in range(len(GraphDataset)):
    Event,UsedPixels, Edges = GraphDataset[i]
    print(f'Event {i} with Id {Event.EventID} has {len(UsedPixels)} pixels and {len(Edges)} edges')
    Adjacency = torch.zeros((len(UsedPixels),len(UsedPixels)))
    for edge in Edges:
        Adjacency[edge[0],edge[1]] = 1
        Adjacency[edge[1],edge[0]] = 1
    Pixels = Event.PixelObservables.GetPixelData()
    Pixels = Pixels[Pixels[:,7].argsort()][UsedPixels]

    plt.figure(figsize = [20,20])
    plt.gca().invert_xaxis()
    
    for i in range(len(Adjacency)):
            for j in range(len(Adjacency)):
                if Adjacency[i,j] == 1:
                    plt.plot([Pixels[i,9],Pixels[j,9]],[90-Pixels[i,10],90-Pixels[j,10]],c='black',alpha=0.2,zorder = 1)
    plt.scatter(Pixels[:,9][torch.sum(Adjacency,dim=1)!=0],90-Pixels[:,10][torch.sum(Adjacency,dim=1)!=0],c=Pixels[:,7][torch.sum(Adjacency,dim=1)!=0],cmap='plasma',s = 100,zorder = 2)
    plt.scatter(Pixels[:,9][torch.sum(Adjacency,dim=1)==0],90-Pixels[:,10][torch.sum(Adjacency,dim=1)==0],color='k',s = 50,marker='x',zorder = 2)
    for i in range(len(Pixels)):
        plt.annotate(str(i),(Pixels[i,9],90-Pixels[i,10]))
    plt.grid()
    plt.xlabel('Phi')
    plt.ylabel('Elevation')
    
    x_center = (plt.xlim()[0] + plt.xlim()[1]) / 2  # Calculate the center of the x-axis
    threshold = 30
    if abs(x_center - plt.xlim()[0]) < threshold and abs(x_center - plt.xlim()[1]) < threshold:
        new_xlim = (x_center - threshold, x_center + threshold)
        plt.xlim(new_xlim)
    plt.savefig(f'../Results/Graph_{Event.EventID}.png')