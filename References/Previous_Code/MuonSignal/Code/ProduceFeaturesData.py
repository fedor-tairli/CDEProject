import torch
import sys
import os
os.system('clear')
sys.path.append('/remote/tychodata/ftairli/work/Projects/MuonSignal/Models')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
Name = 'Model_4_Features'
model = torch.load('../Models/'+Name+'.pt')
print('LoadedModel: '+model.Name)

model = model.to(device)
model.eval()

for DataSetName in ['train','val','test']:
    print(f'Doing {DataSetName} data')
    Main = torch.load('../Data/NormData/Main_'+DataSetName+'.pt')
    Aux   = torch.load('../Data/NormData/Aux_'+DataSetName+'.pt')
    Main = Main.to(device)
    Aux = Aux.to(device)

    Main = Main.transpose(1,2)
    Aux = Aux.unsqueeze(2)
    
    
    print(Main.shape)
    print(Aux.shape)
    features = torch.zeros((len(Main),12))
    with torch.no_grad():
        for i in range(len(Main)):
            print(f'Current Progress {i}/{len(Main)}',end = '\r')
            _,features[i,:] = model(Main[i,:,:].unsqueeze(0),Aux[i,:,:].unsqueeze(0))
    torch.save(features,'../Data/NormData/Features_'+DataSetName+'.pt')
    print()




