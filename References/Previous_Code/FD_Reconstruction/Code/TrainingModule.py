import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

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
        # Other metrics to be added

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

def Train(model,Dataloader,DataLoader_val,optimiser,scheduler,Loss,Validation,Tracker,Epochs=50,accum_steps=1,device = 'cuda',batchBreak = 1e99,normStateIn = 'Net',normStateOut = 'Val',plotHistograms = False,SaveBatchInfo = False):
    '''
    Usage:
        Call it and it will loop over training, and return the model and Training Track
        Pass in Model, Dataloader,Optimiser, Scheduler : Self Explanatory
        Loss       : The Loss Function
        Validation : Function that will validate the model and return validation loss
        Tracker    : Custom Class which will be used to track the training, on EpochEnd may return an abort signal
        Epochs     : Number of Epochs to train for
        accum_steps: Number of batches to accumulate before backprop
        device     : 'cuda' or 'cpu'
        batchBreak : Number of batches to train for, for debugging purposes
        normState  : 'Net' or 'Val' to keep track of validation state
        plotHistograms : Plot the histograms of the predictions every 100 batches or Dont Plot
        SaveBatchInfo  : Save the weights and gradients of the model every batch or Dont Save 
    '''
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
        batchN = 0
        batchT = len(Dataloader)
        epoch_loss = 0
        epoch_loss_Phi = 0
        epoch_loss_Theta = 0

        

        for BatchMain,BatchTruth in Dataloader:
            if batchN > batchBreak:
                break
            batchN += 1

            BatchMain = BatchMain.to(device)
            BatchTruth = BatchTruth.to(device)
            
            if (batchN % accum_steps == 1)  or accum_steps == 1:
                optimiser.zero_grad()

            predictions,normStateOut = model(BatchMain,normStateIn = normStateIn,normStateOut = normStateOut)           
            loss,PhiLoss,ThetaLoss = Loss(predictions,BatchTruth,normState = normStateOut)
            
            if batchN %100 ==0: 
                print(f'Batch {batchN}/{batchT}, Current Loss = {loss.item()}',end='\r') # Progress Bar
                if plotHistograms: plot_Histograms(predictions.detach().cpu().numpy())

            loss.backward()

            if batchN % accum_steps == 0:
                optimiser.step()

                
            epoch_loss += loss.item()
            epoch_loss_Phi += PhiLoss.item()
            epoch_loss_Theta += ThetaLoss.item()


            if SaveBatchInfo:
                named_params = {name: p.cpu() for name, p in model.named_parameters() if p.requires_grad}
                named_gradients = {name: p.grad.cpu() for name, p in model.named_parameters() if p.requires_grad}
                Info = {'BatchLoss': loss.item(),'BatchWeights':named_params,'BatchGrads':named_gradients} # Information required for the Tracker at this stage
                Tracker.InBatch(Info)
            torch.cuda.empty_cache()
            if Tracker.AbortHuh():
                break
        print() # Progress Bar


        epoch_loss = epoch_loss/len(Dataloader)
        epoch_loss_Phi = epoch_loss_Phi/len(Dataloader)
        epoch_loss_Theta = epoch_loss_Theta/len(Dataloader)

        # Validation and Early stopping
        val_loss,val_loss_Phi,val_loss_Theta = Validation(model,DataLoader_val,Loss,device,normStateOut)
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


    return model,Tracker
