import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

def print_grad(name):
    def hook(grad):
        print(f"{name}: {grad}")
    return hook


class Tracker():
     
    def __init__(self,scheduler=None):
        self.EpochLoss         = []
        self.EpochLossPhi      = []
        self.EpochLossTheta    = []
        self.EpochValLoss      = []
        self.EpochValLossPhi   = []
        self.EpochValLossTheta = []
        
        self.EpochLearningRate = []
        
        try:
            self.MinLearningRate = scheduler.min_lrs[0]
        except:
            self.MinLearningRate   = 1e-6

        self.Abort_Call_Reason = None
        self.Abort_Call = False
        

        # Weights, Gradients, if needed
        self.BatchLoss    = []
        self.BatchWeights = []
        self.BatchGrads   = []

        self.ModelStates  = []
        self.ValLossIncreasePatience = 10
        if scheduler is not None:
            if get_scheduler_type(scheduler) == 'ReduceLROnPlateau':
                if scheduler.patience > self.ValLossIncreasePatience:
                    self.ValLossIncreasePatience = scheduler.patience+3


                    
    def PurgeModelStates(self):
        # Leaves Min Validation Loss State Around
        MinValLoss = np.argmin(self.EpochValLoss)
        self.ModelStates = [self.ModelStates[MinValLoss]]

    def EpochStart(self,Info):
        self.EpochLearningRate.append(Info['EpochLearningRate'])

    def InBatch(self,Info):
        self.BatchLoss.append(Info['BatchLoss'])
        if Info['BatchWeights'] is not None:
            self.BatchWeights.append(Info['BatchWeights'])
            self.BatchGrads.append(Info['BatchGrads'])
        
        # Check if any loss is a NaN
        if Info['BatchLoss'] != Info['BatchLoss']:
            self.Abort_Call_Reason = 'NaN Batch Loss'
            self.Abort_Call = True

    def EpochEnd(self,Info):
        self.EpochLoss.append(Info['EpochLoss'])
        self.EpochLossPhi.append(Info['EpochLossPhi'])
        self.EpochLossTheta.append(Info['EpochLossTheta'])

        self.EpochValLoss.append(Info['EpochValLoss'])
        self.EpochValLossPhi.append(Info['EpochValLossPhi'])
        self.EpochValLossTheta.append(Info['EpochValLossTheta'])

        self.ModelStates.append(copy.deepcopy(Info['ModelState']))
        
        print(f'Epoch Loss: {Info["EpochLoss"]:.4f} | Epoch Val Loss: {Info["EpochValLoss"]:.4f}')

        # Abort Checks
        # Check if any loss is a NaN
        if Info['EpochLoss'] != Info['EpochLoss']:
            self.Abort_Call_Reason = 'NaN Epoch Loss'
            self.Abort_Call = True
        if Info['EpochValLoss'] != Info['EpochValLoss']:
            self.Abort_Call_Reason = 'NaN Val Loss'
            self.Abort_Call = True
        
        # Check if val loss is increasing
        
        if len(self.EpochValLoss) > self.ValLossIncreasePatience: # this redundant?
            if len(self.EpochValLoss) - np.argmin(self.EpochValLoss) > self.ValLossIncreasePatience:
                self.Abort_Call_Reason = 'Val Loss Increasing'
                self.Abort_Call = True

        if self.EpochLearningRate[-1] <= self.MinLearningRate:
            self.Abort_Call_Reason = 'Learning Rate too low'
            self.Abort_Call = True
        
    
    def AbortHuh(self):
        return self.Abort_Call
    

def plot_Histograms(pred):
    fig,ax = plt.subplots(1,2)
    ax[0].hist(pred[:,0],bins = 100)
    ax[1].hist(pred[:,1],bins = 100)
    plt.show()
    plt.pause(1)
    plt.close()
    return

def get_scheduler_type(scheduler):
    if isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
        return 'ExponentialLR'
    elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return 'ReduceLROnPlateau'
    # Add more schedulers as elif conditions here as needed.
    else:
        return 'Unknown Scheduler Type'


def IterateInBatches(Dataset,BatchSize = 16):
    '''
    Iterates over the Dataset in Batches
    '''
    counter = 0
    for EvIndex,EvFeatures,EvEdges,EvEdgesWeights,EvTruth in Dataset:
        if EvEdges.shape[1] == 0: continue
        if counter == 0:
            Indices      = torch.tensor([EvIndex])
            Batching     = torch.ones(len(EvFeatures),dtype=torch.int64)*counter
            Features     = EvFeatures
            Edges        = EvEdges
            EdgesWeights = EvEdgesWeights
            Truth        = EvTruth.unsqueeze(0)
        else:
            EvEdges      = EvEdges+ len(Features) # Need to adjust the edges to refer to the appropriate nodes in the batch
            Batching     = torch.cat((Batching,torch.ones(len(EvFeatures),dtype=torch.int64)*counter))
            Indices      = torch.cat((Indices,torch.tensor([EvIndex])))
            Features     = torch.cat((Features,EvFeatures),dim=0)
            Edges        = torch.cat((Edges,EvEdges),dim=1)
            EdgesWeights = torch.cat((EdgesWeights,EvEdgesWeights),dim=0)
            Truth        = torch.cat((Truth,EvTruth.unsqueeze(0)),dim=0)
        counter += 1
        if counter == BatchSize:
            yield Indices,Batching,Features,Edges,EdgesWeights,Truth
            counter = 0


def PlotOnEpoch(Dataset,Model,Epoch,plotSavePath,device):
    ''' To Plot the Predictions against the Truth on the Validation Set'''
    print('Plotting')
    Dataset.State = 'Val'
    BatchSize = len(Dataset)//32
    PredSDPList  = []
    TruthSDPList = []
    with torch.no_grad():
        for BatchEventIndex,BatchGraphBatching,BatchFeatures,BatchEdges,BatchEdgesWeights,BatchTruth in IterateInBatches(Dataset,BatchSize):
            BatchGraphBatching = BatchGraphBatching.to(device)
            BatchFeatures      = BatchFeatures     .to(device)
            BatchEdges         = BatchEdges        .to(device)
            BatchEdgesWeights  = BatchEdgesWeights .to(device)
            BatchTruth         = BatchTruth        .to(device)

            predictions = Model(BatchFeatures,BatchEdges,BatchEdgesWeights,BatchGraphBatching)
            PredSDPList.append(predictions.cpu().detach())
            TruthSDPList.append(BatchTruth.cpu().detach())
    
    ModelPredSDP = torch.cat(PredSDPList,dim=0)
    ModelTruthSDP = torch.cat(TruthSDPList,dim=0)

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    # Scatter plot for the first subplot
    ax[0,0].scatter(ModelTruthSDP[:, 0], ModelPredSDP[:, 0], s=1)
    ax[0,0].plot([min(ModelTruthSDP[:, 0]), max(ModelTruthSDP[:, 0])], [min(ModelTruthSDP[:, 0]), max(ModelTruthSDP[:, 0])], 'r')
    ax[0,0].set_xlabel('Truth Cos(Phi)')
    ax[0,0].set_ylabel('Predicted Cos(Phi)')
    ax[0,0].set_title('Cos(Phi)')


    # Scatter plot for the second subplot
    ax[0,1].scatter(ModelTruthSDP[:, 1], ModelPredSDP[:, 1], s=1)
    ax[0,1].plot([min(ModelTruthSDP[:, 1]), max(ModelTruthSDP[:, 1])], [min(ModelTruthSDP[:, 1]), max(ModelTruthSDP[:, 1])], 'r')
    ax[0,1].set_xlabel('Truth Sin(Phi)')
    ax[0,1].set_ylabel('Predicted Sin(Phi)')
    ax[0,1].set_title('Sin(Phi)')

    # Scatter plot for the third subplot
    ax[1,0].scatter(ModelTruthSDP[:, 2], ModelPredSDP[:, 2], s=1)
    ax[1,0].plot([min(ModelTruthSDP[:, 2]), max(ModelTruthSDP[:, 2])], [min(ModelTruthSDP[:, 2]), max(ModelTruthSDP[:, 2])], 'r')
    ax[1,0].set_xlabel('Truth Cos(Theta)')
    ax[1,0].set_ylabel('Predicted Cos(Theta)')
    ax[1,0].set_title('Cos(Theta)')

    # Scatter plot for the fourth subplot
    ax[1,1].scatter(ModelTruthSDP[:, 3], ModelPredSDP[:, 3], s=1)
    ax[1,1].plot([min(ModelTruthSDP[:, 3]), max(ModelTruthSDP[:, 3])], [min(ModelTruthSDP[:, 3]), max(ModelTruthSDP[:, 3])], 'r')
    ax[1,1].set_xlabel('Truth Sin(Theta)')
    ax[1,1].set_ylabel('Predicted Sin(Theta)')
    ax[1,1].set_title('Sin(Theta)')

    if Epoch == 0:
        # Clear the savepath Dir
        if not plotSavePath.endswith('/'): plotSavePath += '/'
        if not os.path.exists(plotSavePath):
            os.system(f'mkdir {plotSavePath}')
        os.system(f'rm -rf {plotSavePath}*')
    # print(f'Saving to {plotSavePath}Epoch_{Epoch}.png')
    plt.savefig(f'{plotSavePath}Epoch_{Epoch}.png')
        


        
        


def Train(model,Dataset,optimiser,scheduler,Loss,Validation,Tracker,\
          Epochs=50,BatchSize = 16,Accumulation_steps =1 ,device = 'cuda',batchBreak = 1e99,normStateIn = 'Net',normStateOut = 'Val',plotOnEpochCompletionPath = None,SaveBatchInfo = False):
    '''
    Usage:
        Call it and it will loop over training, and return the model and Training Track
        Pass in Model, Dataset ,Optimiser, Scheduler : Self Explanatory, Dataset- ProcessingDataset Defined in Dataset.py
        Loss       : The Loss Function
        Validation : Function that will validate the model and return validation loss
        Tracker    : Custom Class which will be used to track the training, on EpochEnd may return an abort signal
        Epochs     : Number of Epochs to train for
        BatchSize  : Size of the Batch
        Accumulation_steps : How many steps to accumulate gradients before stepping
        device     : 'cuda' or 'cpu'
        batchBreak : How many effective Batches to process before breaking
        normState  : 'Net' or 'Val' to keep track of validation state
        plotOnEpochCompletionPath : Takes a path to save the plots of the predictions against the truth
        SaveBatchInfo  : Save the weights and gradients of the model every batch or Dont Save 
    '''
    assert SaveBatchInfo == False, 'SaveBatchInfo is not implemented'
    # Initialise Tracker
    Tracker = Tracker(scheduler)
    Min_Val_Loss = 1e99
    tolerance = 0.05
    
    # Training Loop
    torch.autograd.set_detect_anomaly(True)
    for i in range(Epochs):
        print(f'Epoch {i+1}/{Epochs}') # Progress Bar
        Info = {'EpochLearningRate':[param_group['lr'] for param_group in optimiser.param_groups][0]} # Information required for the Tracker at this stage
        Tracker.EpochStart(Info)
        
        model.train()
        batchN           = 0 # Batch Number, Effectively - by summing gradients
        epoch_loss       = 0
        epoch_loss_Phi   = 0
        epoch_loss_Theta = 0

        # Ensure Dataset is in Train State before processing
        Dataset.State = 'Train'
        for BatchEventIndex,BatchGraphBatching,BatchFeatures,BatchEdges,BatchEdgesWeights,BatchTruth in IterateInBatches(Dataset,BatchSize): # Dataset Event index is not used
            
            # Check if the Event is okay i guess
            if batchN > batchBreak:
                break
            
            
            BatchGraphBatching = BatchGraphBatching.to(device) # Tensor specifies graphs in the batch
            BatchFeatures      = BatchFeatures     .to(device) # Tensor containing the Node Features
            BatchEdges         = BatchEdges        .to(device) # Tensor Specifying the Edges
            BatchEdgesWeights  = BatchEdgesWeights .to(device) # Tensor Speciyfing the Edge Weights
            BatchTruth         = BatchTruth        .to(device) # Tensor containing the Truth values


            if Accumulation_steps == 1 or batchN%Accumulation_steps == 0:
                optimiser.zero_grad()

            predictions = model(BatchFeatures,BatchEdges,BatchEdgesWeights,BatchGraphBatching)   
            loss,PhiLoss,ThetaLoss = Loss(predictions,BatchTruth)
            
            loss.backward()


            
            epoch_loss += loss.item()
            epoch_loss_Phi += PhiLoss.item()
            epoch_loss_Theta += ThetaLoss.item()


            optimiser.step()
                            
            print(f'Batch {str(batchN).ljust(6)}/{len(Dataset)//BatchSize} - Loss: {str(loss.item())[:6].ljust(6)} - Phi: {str(PhiLoss.item())[:6].ljust(6)} - Theta: {str(ThetaLoss.item())[:6].ljust(6)}',end = '\r')
            if batchN % 100 == 0: print()
   
            batchN += 1

            # if SaveBatchInfo:
            #     named_params = {name: p.cpu() for name, p in model.named_parameters() if p.requires_grad}
            #     named_gradients = {name: p.grad.cpu() for name, p in model.named_parameters() if p.requires_grad}
            #     Info = {'BatchLoss': loss.item(),'BatchWeights':named_params,'BatchGrads':named_gradients} # Information required for the Tracker at this stage
            #     Tracker.InBatch(Info)
            if Tracker.AbortHuh():
                print(f'Batch that broke: {batchN}')
                break
        print() # Progress Bar

        # Epoch End will devide by number of batches
        epoch_loss = epoch_loss/batchN   
        epoch_loss_Phi = epoch_loss_Phi/batchN
        epoch_loss_Theta = epoch_loss_Theta/batchN

        # Validation and Early stopping
        val_loss,val_loss_Phi,val_loss_Theta = Validation(model,Dataset,Loss,device,normStateOut)
        if val_loss < Min_Val_Loss*(1-tolerance):
            Min_Val_Loss = val_loss
        if get_scheduler_type(scheduler) == 'ReduceLROnPlateau':
            scheduler.step(Min_Val_Loss)
        if get_scheduler_type(scheduler) == 'ExponentialLR':
            scheduler.step()
        
        Info = {'EpochLoss': epoch_loss, 'EpochValLoss':val_loss,'ModelState':model.state_dict(),\
                'EpochLossPhi':epoch_loss_Phi, 'EpochLossTheta': epoch_loss_Theta,\
                'EpochValLossPhi':val_loss_Phi, 'EpochValLossTheta': val_loss_Theta
                } # Information required for the Tracker at this stage
        Tracker.EpochEnd(Info)
        

        if Tracker.AbortHuh():
            print(f'Aborting Training: {Tracker.Abort_Call_Reason}')
            break
        if plotOnEpochCompletionPath!=None:
            PlotOnEpoch(Dataset,model,i,plotOnEpochCompletionPath,device)


    return model,Tracker
