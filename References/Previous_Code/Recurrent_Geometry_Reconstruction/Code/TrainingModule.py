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
        self.EpochLossChi0     = []
        self.EpochLossRp       = []
        self.EpochValLoss      = []
        self.EpochValLossChi0  = []
        self.EpochValLossRp    = []
        
        self.EpochLearningRate = []
        
        try:
            self.MinLearningRate = scheduler.min_lrs[0]
        except:
            self.MinLearningRate   = 1e-8

        self.Abort_Call_Reason = None
        self.Abort_Call = False
        

        # Weights, Gradients, if needed
        self.BatchLoss    = []
        self.BatchWeights = []
        self.BatchGrads   = []

        self.ModelStates  = []
        self.ValLossIncreasePatience = 15
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
        self.EpochLossChi0.append(Info['EpochLossChi0'])
        self.EpochLossRp.append(Info['EpochLossRp'])

        self.EpochValLoss.append(Info['EpochValLoss'])
        self.EpochValLossChi0.append(Info['EpochValLossChi0'])
        self.EpochValLossRp.append(Info['EpochValLossRp'])

        self.ModelStates.append(copy.deepcopy(Info['ModelState']))
        
        if Info['EpochLoss']> 0.0001 and Info['EpochValLoss'] > 0.0001:
            print(f'Epoch Loss: {Info["EpochLoss"]:.4f} | Epoch Val Loss: {Info["EpochValLoss"]:.4f} | {Info["metric"]}')
        else:
            print(f'Epoch Loss: {Info["EpochLoss"]:.4e} | Epoch Val Loss: {Info["EpochValLoss"]:.4e} | {Info["metric"]}')


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
    Dataset Must be a ProcessingDataset
    '''
    counter = 0
    for EvIndex,EvTraces,EvAux,EvTruth in Dataset:
        if counter == 0:
            Indices      = torch.tensor([EvIndex])
            Traces       = EvTraces.unsqueeze(0)
            Aux          = EvAux.unsqueeze(0)
            Truth        = EvTruth.unsqueeze(0)
            
        else:
            Indices      = torch.cat((Indices,torch.tensor([EvIndex])))
            Traces       = torch.cat((Traces,EvTraces.unsqueeze(0)))
            Aux          = torch.cat((Aux,EvAux.unsqueeze(0)))
            Truth        = torch.cat((Truth,EvTruth.unsqueeze(0)))

        counter += 1
        if counter == BatchSize:
            yield Indices,Traces,Aux,Truth
            counter = 0


def PlotOnEpoch(Dataset,Model,Epoch,plotSavePath,device):
    ''' To Plot the Predictions against the Truth on the Validation Set'''
    print('Plotting')
    Dataset.State = 'Val'
    BatchSize = len(Dataset)//256
    PredGeomList  = []
    TruthGeomList = []
    with torch.no_grad():
        for BatchEventIndex,BatchTraces,BatchAux,BatchTruth in IterateInBatches(Dataset,BatchSize):
            BatchTraces        = BatchTraces.to(device)
            BatchAux           = BatchAux   .to(device)
            BatchTruth         = BatchTruth .to(device)

            predictions = Model(BatchTraces,BatchAux)
            PredGeomList.append(predictions.cpu().detach())
            TruthGeomList.append(BatchTruth.cpu().detach())
    
    ModelPredGeom  = torch.cat(PredGeomList,dim=0)
    ModelTruthGeom = torch.cat(TruthGeomList,dim=0)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Scatter plot for the first subplot
    ax[0].scatter(ModelTruthGeom[:, 0].numpy(), ModelPredGeom[:, 0].numpy(), s=1)
    ax[0].plot([min(ModelTruthGeom[:, 0]), max(ModelTruthGeom[:, 0])], [min(ModelTruthGeom[:, 0]), max(ModelTruthGeom[:, 0])], 'r')
    ax[0].set_xlabel('Truth Chi0')
    ax[0].set_ylabel('Predicted Chi0')
    ax[0].set_title('Chi0')


    # Scatter plot for the second subplot
    ax[1].scatter(ModelTruthGeom[:, 1].numpy(), ModelPredGeom[:, 1].numpy(), s=1)
    ax[1].plot([min(ModelTruthGeom[:, 1]), max(ModelTruthGeom[:, 1])], [min(ModelTruthGeom[:, 1]), max(ModelTruthGeom[:, 1])], 'r')
    ax[1].set_xlabel('Truth Rp')
    ax[1].set_ylabel('Predicted Rp')
    ax[1].set_title('Rp')

    
    if Epoch == 0:
        # Clear the savepath Dir
        if not plotSavePath.endswith('/'): plotSavePath += '/'
        if not os.path.exists(plotSavePath):
            os.system(f'mkdir {plotSavePath}')
        # os.system(f'rm -rf {plotSavePath}*')
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
        batchN           = 0
        epoch_loss       = 0
        epoch_loss_Chi0  = 0
        epoch_loss_Rp    = 0

        # Ensure Dataset is in Train State before processing
        Dataset.State = 'Train'
        for BatchEventIndex,BatchTraces,BatchAux,BatchTruth in IterateInBatches(Dataset,BatchSize): # Dataset Event index is not used
            
            # Check if the Event is okay i guess
            if batchN > batchBreak:
                break
            
            
            BatchTraces    = BatchTraces.to(device) # Tensor holds the combined signal and chi trace
            BatchAux       = BatchAux   .to(device) # Tensor containing the extra event data - station data
            BatchTruth     = BatchTruth .to(device) # Tensor containing the Truth values


            if Accumulation_steps == 1 or batchN%Accumulation_steps == 0:
                optimiser.zero_grad()
            
            predictions = model(BatchTraces,BatchAux)   
            loss,Chi0Loss,RpLoss = Loss(predictions,BatchTruth)

            
            loss.backward()
            

            
            epoch_loss      += loss.item()
            epoch_loss_Chi0 += Chi0Loss.item()
            epoch_loss_Rp   += RpLoss.item()


            optimiser.step()
                            
            print(f'Batch {str(batchN).ljust(6)}/{len(Dataset)//BatchSize} - Loss: {str(loss.item())[:6].ljust(6)} - Chi0: {str(Chi0Loss.item())[:6].ljust(6)} - Rp: {str(RpLoss.item())[:6].ljust(6)}',end = '\r')
            # if batchN % 100 == 0: print()
   
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
        epoch_loss      = epoch_loss/batchN        if batchN !=0 else epoch_loss
        epoch_loss_Chi0 = epoch_loss_Chi0/batchN   if batchN !=0 else epoch_loss_Chi0
        epoch_loss_Rp   = epoch_loss_Rp/batchN     if batchN !=0 else epoch_loss_Rp
        # Validation and Early stopping # metric is a str to be printed
        val_loss,val_loss_Chi0,val_loss_Rp,metric = Validation(model,Dataset,Loss,device,normStateOut)

        if val_loss < Min_Val_Loss*(1-tolerance):
            Min_Val_Loss = val_loss
        if get_scheduler_type(scheduler) == 'ReduceLROnPlateau':
            scheduler.step(Min_Val_Loss)
        if get_scheduler_type(scheduler) == 'ExponentialLR':
            scheduler.step()
        
        Info = {'EpochLoss': epoch_loss, 'EpochValLoss':val_loss,'ModelState':model.state_dict(),\
                'EpochLossChi0':epoch_loss_Chi0, 'EpochLossRp': epoch_loss_Rp,\
                'EpochValLossChi0':val_loss_Chi0, 'EpochValLossRp': val_loss_Rp,\
                'metric':metric
                } # Information required for the Tracker at this stage
        Tracker.EpochEnd(Info)
        

        if Tracker.AbortHuh():
            print(f'Aborting Training: {Tracker.Abort_Call_Reason}')
            break
        if plotOnEpochCompletionPath!=None:
            PlotOnEpoch(Dataset,model,i,plotOnEpochCompletionPath,device)


    return model,Tracker
