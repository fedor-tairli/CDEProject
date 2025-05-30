import torch
import copy
import numpy as np

class Tracker():
     
    def __init__(self,scheduler=None):
        self.EpochLoss_T    = []
        self.EpochLoss_E    = []
        self.EpochLoss_C    = []
        self.EpochLoss_A    = []
        self.EpochLoss_X    = []

        self.EpochValLoss_T = []
        self.EpochValLoss_E = []
        self.EpochValLoss_C = []
        self.EpochValLoss_A = []
        self.EpochValLoss_X = []

        self.BatchLoss_T    = []
        self.BatchLoss_E    = []
        self.BatchLoss_C    = []
        self.BatchLoss_A    = []
        self.BatchLoss_X    = []

        
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
        self.BatchLoss_T.append(Info['BatchLoss_T'])
        self.BatchLoss_E.append(Info['BatchLoss_E'])
        self.BatchLoss_C.append(Info['BatchLoss_C'])
        self.BatchLoss_A.append(Info['BatchLoss_A'])
        self.BatchLoss_X.append(Info['BatchLoss_X'])

        # self.BatchWeights.append(Info['BatchWeights'])
        # self.BatchGrads.append(Info['BatchGrads'])
        # Check if any loss is a NaN
        if Info['BatchLoss_T'] != Info['BatchLoss_T']:
            self.Abort_Call_Reason = 'NaN Batch Loss'
            self.Abort_Call = True

    def EpochEnd(self,Info):
        self.EpochLoss_T.append(Info['EpochLoss_T'])
        self.EpochLoss_E.append(Info['EpochLoss_E'])
        self.EpochLoss_C.append(Info['EpochLoss_C'])
        self.EpochLoss_A.append(Info['EpochLoss_A'])
        self.EpochLoss_X.append(Info['EpochLoss_X'])

        self.EpochValLoss_T.append(Info['EpochValLoss_T'])
        self.EpochValLoss_E.append(Info['EpochValLoss_E'])
        self.EpochValLoss_C.append(Info['EpochValLoss_C'])
        self.EpochValLoss_A.append(Info['EpochValLoss_A'])
        self.EpochValLoss_X.append(Info['EpochValLoss_X'])

               
        self.ModelStates.append(copy.deepcopy(Info['ModelState']))
        print(f'Epoch Loss: {Info["EpochLoss_T"]:.4f} | Epoch Val Loss: {Info["EpochValLoss_T"]:.4f}')
        print(f'E_Loss    : {Info["EpochLoss_E"]:.4f} | C_Loss    : {Info["EpochLoss_C"]:.4f} | A_Loss    : {Info["EpochLoss_A"]:.4f} | X_Loss    : {Info["EpochLoss_X"]:.4f}')

        # Abort Checks
        # Check if any loss is a NaN
        if Info['EpochLoss_T'] != Info['EpochLoss_T']:
            self.Abort_Call_Reason = 'NaN Epoch Loss'
            self.Abort_Call = True
        if Info['EpochValLoss_T'] != Info['EpochValLoss_T']:
            self.Abort_Call_Reason = 'NaN Val Loss'
            self.Abort_Call = True
        # Check if val loss is increasing
        
        if len(self.EpochValLoss_T) > self.ValLossIncreasePatience: # this redundant?
            if len(self.EpochValLoss_T) - np.argmin(self.EpochValLoss_T) > self.ValLossIncreasePatience:
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


    # Training Loop
    for i in range(Epochs):
        print(f'Epoch {i+1}/{Epochs}') # Progress Bar
        Info = {'Temp':'Not Empty'} # Information required for the Tracker at this stage
        Tracker.EpochStart(Info)
        
        model.train()
        batchN = 0
        batchT = len(Dataloader)
        epoch_loss_T = 0
        epoch_loss_E = 0
        epoch_loss_C = 0
        epoch_loss_A = 0
        epoch_loss_X = 0


        

        for D_Main, D_Aux,logE,Core,Axis,Xmax in Dataloader:
            if batchN > batchBreak:
                break
            batchN += 1
            print(f'Batch {batchN}/{batchT}',end='\r') # Progress Bar

            D_Main = D_Main.to(device)
            D_Aux  = D_Aux.to(device)
            logE   = logE.to(device)
            Core   = Core.to(device)
            Axis   = Axis.to(device)
            Xmax   = Xmax.to(device)

            
            if batchN % accum_steps == 1:
                optimiser.zero_grad()

            predictions = model(D_Main,D_Aux,)

            # predictions = model.UnnormaliseY(predictions)
            # Truth       = model.UnnormaliseY(Truth)

            loss_T,loss_E,loss_C,loss_A,loss_X = Loss(predictions,(logE,Core,Axis,Xmax))
            
            (loss_T/accum_steps).backward()

            if batchN % accum_steps == 0:
                optimiser.step()
                
            epoch_loss_T += loss_T.item()
            epoch_loss_E += loss_E.item()
            epoch_loss_C += loss_C.item()
            epoch_loss_A += loss_A.item()
            epoch_loss_X += loss_X.item()

            named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
            named_gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}

            Info = {'BatchLoss_T': loss_T.item(),'BatchWeights':named_params,'BatchGrads':named_gradients,\
                    'BatchLoss_E': loss_E.item(),'BatchLoss_C': loss_C.item(),'BatchLoss_A': loss_A.item(),'BatchLoss_X': loss_X.item()} # Information required for the Tracker at this stage
            Tracker.InBatch(Info)
            torch.cuda.empty_cache()
        print() # Progress Bar


        epoch_loss_T = epoch_loss_T/len(Dataloader)
        epoch_loss_E = epoch_loss_E/len(Dataloader)
        epoch_loss_C = epoch_loss_C/len(Dataloader)
        epoch_loss_A = epoch_loss_A/len(Dataloader)
        epoch_loss_X = epoch_loss_X/len(Dataloader)

        ValLoss_T,ValLoss_E,ValLoss_C,ValLoss_A,ValLoss_X = Validation(model,DataLoader_val,Loss,model_Coefficients = model.LossCoefficients,device = device)

        scheduler.step(ValLoss_T)
        Info = {'EpochLoss_T': epoch_loss_T, 'EpochValLoss_T':ValLoss_T,'ModelState':model.state_dict(),\
                'EpochLoss_E': epoch_loss_E, 'EpochValLoss_E':ValLoss_E,\
                'EpochLoss_C': epoch_loss_C, 'EpochValLoss_C':ValLoss_C,\
                'EpochLoss_A': epoch_loss_A, 'EpochValLoss_A':ValLoss_A,\
                'EpochLoss_X': epoch_loss_X, 'EpochValLoss_X':ValLoss_X} # Information required for the Tracker at this stage
        Tracker.EpochEnd(Info)
        Abort = Tracker.AbortHuh()

        if Abort:
            break


    return model,Tracker


