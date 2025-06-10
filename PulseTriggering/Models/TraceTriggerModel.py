# Define a simple trace trigger model here. 
# Idea : Hourglass shaped model which will bottleneck for some features.
#        It Will then expand again with deconvolution layers to reconstruct the input.
#        Features might be used to predict the trigger time. 

# Impor Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F



def Loss(Pred,Truth,Weights,keys=['TraceLikeness'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    
    # Calculate Loss
    #  Calculate the MSE Loss for the trace generator
    # Truth Contains the begining and end of the signal, and also status of the trace
    # We can use this to create weights for the loss to focus on the part of the singal that is important
    
    assert Truth.shape == Pred.shape, f'Truth shape {Truth.shape} does not match Pred shape {Pred.shape}'
    assert Weights.shape == Pred.shape, f'Weights shape {Weights.shape} does not match Pred shape {Pred.shape}'
    Truth = Truth.to(Pred.device)
    Weights = Weights.to(Pred.device)
    # MSE = F.mse_loss(Pred, Truth,weight = Weights)
    MSE = torch.mean(Weights * (Pred - Truth) ** 2)
    losses = {'Total': MSE , 'TraceLikeness': MSE}
    return losses


def metric(model,Dataset,device,keys=['TraceLikeness'],BatchSize = 256):
    '''
    Really do nothing becuase the only difference to loss is that we unnormalize the output so nothing can be done
    '''
    metrics = {'TraceLikeness': 0.0}
    return metrics

def validate(model,Dataset,device,keys=['TraceLikeness'],BatchSize = 1024):
    '''
    Runs Loss on the validation dataset and returns'''

    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds   = []
    Truths  = []
    Weights = []
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth,_  in Dataset:
            
            Preds  .append( model(BatchMains,BatchAux).to('cpu'))
            Truths .append(       BatchTruth          .to('cpu'))
            Weights.append(       BatchAux            .to('cpu'))
        Preds   = torch.cat(Preds ,dim=0)
        Truths  = torch.cat(Truths,dim=0)
        Weights = torch.cat(Weights,dim=0)
    
    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss(Preds,Truths,Weights,keys=Dataset.Truth_Keys,ReturnTensor=False)




class TraceTriggerModel(nn.Module):

    def __init__(self, input_channels=1, output_channels=1,**kwargs):
        super(TraceTriggerModel, self).__init__()
        self.Name = 'TraceTriggerModel'
        self.Description = '''
        A simple Model which will learn to reconstruct the trace,
        Has a encoder-decoder structure with a bottleneck in the middle.
        The bottleneck will be used to extract features and trigger fromt he trace.
        '''
        self.kwargs = kwargs
        # Define some parameters which ideal will later be set by user in initialization
        self.KernelSizes = [7,5,3]
        self.N_kernels   = [64, 32, 16]

        self.BottleneckSize = 8

        # Encoder
        self.Endocder = nn.Sequential(
            nn.Conv1d(input_channels   , self.N_kernels[0], kernel_size=self.KernelSizes[0], padding=(self.KernelSizes[0]-1)//2),
            nn.LeakyReLU(),
            nn.Conv1d(self.N_kernels[0], self.N_kernels[1], kernel_size=self.KernelSizes[1], padding=(self.KernelSizes[1]-1)//2),
            nn.LeakyReLU(),
            nn.Conv1d(self.N_kernels[1], self.N_kernels[2], kernel_size=self.KernelSizes[2], padding=(self.KernelSizes[2]-1)//2),
            nn.LeakyReLU(),
            nn.Conv1d(self.N_kernels[2], self.BottleneckSize, kernel_size=1),  
            nn.LeakyReLU(),
        )
            

        # Decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(self.BottleneckSize, self.N_kernels[2], kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.N_kernels[2], self.N_kernels[1], kernel_size=self.KernelSizes[2], padding=(self.KernelSizes[2]-1)//2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.N_kernels[1], self.N_kernels[0], kernel_size=self.KernelSizes[1], padding=(self.KernelSizes[1]-1)//2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(self.N_kernels[0], output_channels  , kernel_size=self.KernelSizes[0], padding=(self.KernelSizes[0]-1)//2),
        )
        

    def forward(self, x,aux=None):
        x = x[0] # Input will be a tuple, we take the first element
        x = x.to(self.Endocder[0].weight.device)
        # Pass through encoder
        x = self.Endocder(x)
        
        # Pass through decoder
        x = self.Decoder(x)
        
        return x

    def get_features(self, x,aux=None):
        x = x[0]
        x = x.to(self.Endocder[0].weight.device)
        # Get features from the bottleneck layer
        x = self.Endocder[:-1](x)  # Exclude the last layer to get features
        return x
