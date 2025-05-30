from adst3 import RecEventProvider
import torch

FilesPath = '/remote/tychodata/ftairli/data/PCGF/REAL/run/2015/'
FileNames = [f'PCGF_run_{i:02d}_2015_HEAT.root' for i in range(1,13)]

Npixels = []
for FileName in FileNames:
    print(f'Processing {FileName}')
    i=0
    for ev in RecEventProvider(FilesPath+FileName):
        i+=1
        print(f'Event {i}',end='\r')
        N = ev.GetHottestEye().GetFdRecPixel().GetNumberOfTriggeredPixels()
        Npixels.append(N)
    print()
torch.save(torch.tensor(Npixels), 'Npixels.pt')