import torch
import torch.nn as nn
import torch.nn.functional as F



def Loss(Pred,Truth,keys=['CherenkovFraction'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
    
    losses['Total'] = sum(losses.values())
    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses
    


def validate(model,Dataset,Loss,device,BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth,_  in Dataset:
            
            Preds .append( model(BatchMains,BatchAux).to('cpu'))
            Truths.append(       BatchTruth          .to('cpu'))
        Preds  = torch.cat(Preds ,dim=0)
        Truths = torch.cat(Truths,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss(Preds,Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)
    

def metric(model,Dataset,device,keys = ['ChrenkovFraction'],BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the 68% containment range of the angular deviation
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth, _ in Dataset:
            Preds .append( model(BatchMains,BatchAux).to('cpu'))
            Truths.append(       BatchTruth          .to('cpu'))
    Preds  = torch.cat(Preds ,dim=0).cpu()
    Truths = torch.cat(Truths,dim=0).cpu()
    Preds  = Dataset.Unnormalise_Truth(Preds )
    Truths = Dataset.Unnormalise_Truth(Truths)

    Units = Dataset.Truth_Units
    metrics = {}

    for i,key in enumerate(keys):
        if Units[i] == 'rad':
            AngDiv = torch.atan2(torch.sin(Preds[:,i]-Truths[:,i]),torch.cos(Preds[:,i]-Truths[:,i]))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)
        if Units[i] == 'deg':
            AngDiv = torch.atan2(torch.sin(torch.deg2rad(Preds[:,i]-Truths[:,i])),torch.cos(torch.deg2rad(Preds[:,i]-Truths[:,i])))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)*180/torch.pi
        else:
            metrics[key] = torch.quantile(torch.abs(Preds[:,i]-Truths[:,i]),0.68)
    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return metrics



# For no Activation function
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

class Model_ChrenkovFraction(nn.Module):
    Name = 'Model_ChrenkovFraction'
    Description = '''Simple Conv3d Model that will predict the Cherenkov Fraction
    No Aux Data inputs
    Try Regular pooling for now'''


    def __init__(self, in_main_channels=1, out_channels=1, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility and completeness
        self.kwargs = kwargs
        super(Model_ChrenkovFraction,self).__init__()
        
        assert len(in_main_channels) == 1, 'Only have one main'
        assert in_main_channels[0] == 1, 'Only have one main'
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,40,20,22) # Dont forget to unsquish to account for extra dimension of the channel

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 40x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      40x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      40x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Fully Connected Layers
        self.Dense1 = nn.Linear(N_kernels*2*2*2,N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes,1)
        assert out_channels == 1, 'Output channels should be 1 for now'

        # Dropout
        assert DropOut_rate == 0 , 'Dropout not implemented yet'

        # Activation Functions
        self.Conv_Activation = nn.LeakyReLU()
        self.FC_Activation   = nn.LeakyReLU()
        self.Fraction_Activation = nn.Sigmoid()
        self.Angle_Activation = nn.Tanh()
        self.No_Activation = Identity()



    def forward(self,Graph,Aux=None):

        # Unpack the Graph Data to Main
        device = self.Conv1.weight.device
        NEvents = len(Graph)

        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100 ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it.

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        # Main = Main.unfold(1,10,10)
        # Main = Main.sum(-1)
        # Main = Main[:,:80,:,:].unsqueeze(1).to(device)
       
       # Dont need to rebin this, because our dimensions are already small
       # Just take the first 40 bins
        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1

        # Process the Data

        X = self.Conv_Activation(self.Conv1(Main ))
        X = self.Conv_Activation(self.Conv2(  X  ))
        X = self.Conv_Activation(self.Conv3(  X  ))
        X = self.Pool1(X)
        X = self.Conv_Activation(self.Conv4(  X  ))
        X = self.Conv_Activation(self.Conv5(  X  ))
        X = self.Conv_Activation(self.Conv6(  X  ))
        X = self.Pool2(X)
        X = self.Conv_Activation(self.Conv7(  X  ))
        X = self.Conv_Activation(self.Conv8(  X  ))
        X = self.Conv_Activation(self.Conv9(  X  ))
        X = self.Pool3(X)

        X = X.view(X.size(0),-1)
        X = self.FC_Activation(self.Dense1(X))
        X = self.FC_Activation(self.Dense2(X))
        X = self.No_Activation(self.Dense3(X))

        X = self.Fraction_Activation(X)
        return X
