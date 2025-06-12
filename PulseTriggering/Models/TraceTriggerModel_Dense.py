# Define a simple trace trigger model here. 
# Idea : Hourglass shaped model which will bottleneck for some features.
#        It Will then expand again with deconvolution layers to reconstruct the input.
#        Features might be used to predict the trigger time. 

# Impor Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F



# def Loss(Pred,Truth,Weights,keys=['TraceLikeness'],ReturnTensor = True):

#     '''
#     Calculates MSE Loss for all the keys in the keys list
#     '''
    
#     # Calculate Loss
#     #  Calculate the MSE Loss for the trace generator
#     # Truth Contains the begining and end of the signal, and also status of the trace
#     # We can use this to create weights for the loss to focus on the part of the singal that is important
    
#     assert Truth.shape == Pred.shape, f'Truth shape {Truth.shape} does not match Pred shape {Pred.shape}'
#     assert Weights.shape == Pred.shape, f'Weights shape {Weights.shape} does not match Pred shape {Pred.shape}'
#     Truth = Truth.to(Pred.device)
#     Weights = Weights.to(Pred.device)
#     # MSE = F.mse_loss(Pred, Truth,weight = Weights)
#     MSE = torch.mean(Weights * (Pred - Truth) ** 2)
#     losses = {'Total': MSE , 'TraceLikeness': MSE}
#     return losses

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
    # MSE = torch.mean(Weights * (Pred - Truth) ** 2)
    # Added the penalty for predicting the peaks in the noize region
    MSE = torch.mean(Weights * (Pred - Truth) ** 2 + (1-Weights) * Pred**2 ) 
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




class TraceTriggerModelDense_NF(nn.Module):
    def __init__(self, input_channels=1, output_channels=1,BottleNeckSize = 3,**kwargs):
        super(TraceTriggerModelDense_NF, self).__init__()
        self.Name = f'TraceTriggerModelDense_{BottleNeckSize}F'
        self.Description = f'''
        A Simple trace trigger model with a bottleneck in the middle. 
        The idea is that this time it will produce N features per trace, rather than per bin as above. 
        The Feature will be used to predict signal
        This model has {BottleNeckSize} Features.
        '''
        self.BottleNeckSize = BottleNeckSize

        self.kwargs = kwargs
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(1000,1000), # (B,1000)
            nn.LeakyReLU(),
            nn.Linear(1000,500), # (B,500)
            nn.LeakyReLU(),
            nn.Linear(500,250),  # (B,250)
            nn.LeakyReLU(),
            nn.Linear(250,125),  # (B,125)
            nn.LeakyReLU(),
            nn.Linear(125, self.BottleNeckSize)  # (B,3*N)
        )
        
        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(self.BottleNeckSize, 125),  # (B,125)
            nn.LeakyReLU(),
            nn.Linear(125, 250),  # (B,250)
            nn.LeakyReLU(),
            nn.Linear(250, 500),  # (B,500)
            nn.LeakyReLU(),
            nn.Linear(500, 1000), # (B,1000)
            nn.LeakyReLU(),
            nn.Linear(1000, 1000), # (B,1000)
        )

    def forward(self, x, aux=None):
        x = x[0]
        x = x.to(self.Encoder[0].weight.device)
        # Pass through encoder
        x = self.Encoder(x)
        # Pass through bottleneck
        x = self.Decoder(x)
        return x

    def get_features(self, x, aux=None):
        x = x[0]
        x = x.squeeze(1)
        x = x.to(self.Encoder[0].weight.device)
        x = self.Encoder(x)
        return x
    

def Weight_Loss(Pred,Truth,Weights,keys=['TraceLikeness'],ReturnTensor = True):

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
    MSE = torch.mean(Weights * (Pred - Weights) ** 2)
    losses = {'Total': MSE , 'TraceLikeness': MSE}
    return losses



