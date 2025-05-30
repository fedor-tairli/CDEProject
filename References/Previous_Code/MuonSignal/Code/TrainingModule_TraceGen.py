import torch
import copy
import numpy as np

class Tracker():
     
    def __init__(self,scheduler=None):
        self.EpochLoss    = []
        self.EpochValLoss = []
        self.BatchLoss    = []
        
        self.Abort_Call_Reason = None
        self.Abort_Call = False
        # Other metrics to be added

        # Weights, Gradients, if needed
        self.BatchWeights = []
        self.BatchGrads   = []

        self.ModelStates  = []
        self.ValLossIncreasePatience = 3
        if scheduler is not None:
            if scheduler.patience > self.ValLossIncreasePatience:
                self.ValLossIncreasePatience = scheduler.patience+3

    def EpochStart(self,Info):
        # self.temp = Info['Temp']
        pass

    def InBatch(self,Info):
        self.BatchLoss.append(Info['BatchLoss'])
        self.BatchWeights.append(Info['BatchWeights'])
        self.BatchGrads.append(Info['BatchGrads'])
        # Check if any loss is a NaN
        if Info['BatchLoss'] != Info['BatchLoss']:
            self.Abort_Call_Reason = 'NaN Batch Loss'
            self.Abort_Call = True

    def EpochEnd(self,Info):
        self.EpochLoss.append(Info['EpochLoss'])
        self.EpochValLoss.append(Info['EpochValLoss'])
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

        
    
    def AbortHuh(self):
        return self.Abort_Call






def Train(model,Dataloader,DataLoader_val,optimiser,scheduler,Loss,Validation,Tracker,Epochs=50,accum_steps=1,device = 'cuda',batchBreak = 1e99):
    '''
    Usage:
        Call it and it will loop over training, and return the model and Training Track
        Pass in Model, Dataloader,Optimiser, Scheduler : Self Explanatory
        Loss       : The Loss Function
        Validation : Function that will validate the model and return validation loss
        Tracker    : Custom Class which will be used to track the training, on EpochEnd may return an abort signal
        Printer    : Function that will print things during the training (not implemented yet)
        Epochs     : Number of Epochs to train for
        accum_steps: Number of batches to accumulate before backprop
        device     : 'cuda' or 'cpu'
        batchBreak : Number of batches to train for, for debugging purposes
    '''
    # Initialise Tracker
    Tracker = Tracker(scheduler)
    Min_Val_Loss = 1e99
    tolerance = 0.05

    # Training Loop
    for i in range(Epochs):
        print(f'Epoch {i+1}/{Epochs}') # Progress Bar
        Info = {'Temp':'Not Empty'} # Information required for the Tracker at this stage
        Tracker.EpochStart(Info)
        
        model.train()
        batchN = 0
        batchT = len(Dataloader)
        epoch_loss = 0


        

        for Features, Trace in Dataloader:
            if batchN > batchBreak:
                break
            batchN += 1
            print(f'Batch {batchN}/{batchT}',end='\r') # Progress Bar

            Features = Features.to(device)
            Trace = Trace.to(device)

            if batchN % accum_steps == 1:
                optimiser.zero_grad()

            predictions = model(Features)

            loss = Loss(predictions,Trace)
            
            (loss/accum_steps).backward()

            if batchN % accum_steps == 0:
                optimiser.step()
                
            epoch_loss += loss.item()

            named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            named_gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}

            Info = {'BatchLoss': loss.item(),'BatchWeights':named_params,'BatchGrads':named_gradients} # Information required for the Tracker at this stage
            Tracker.InBatch(Info)
            torch.cuda.empty_cache()
        print() # Progress Bar


        epoch_loss = epoch_loss/len(Dataloader)
        Val_Loss = Validation(model,DataLoader_val,Loss,device,Unnorm = False)
        if Val_Loss < Min_Val_Loss*(1-tolerance):
            Min_Val_Loss = Val_Loss
            
        scheduler.step(Min_Val_Loss)
        Info = {'EpochLoss': epoch_loss, 'EpochValLoss':Val_Loss,'ModelState':model.state_dict()} # Information required for the Tracker at this stage
        Tracker.EpochEnd(Info)
        

        if Tracker.AbortHuh():
            break


    return model,Tracker


