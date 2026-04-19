import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from time import time

class Tracker():
     
    def __init__(self,scheduler,Training_Parameters = {},Model_Parameters = {},**kwargs):
        self.StartTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.EpochLoss         = {'Total':[]} # Average loss over epoch
        self.EpochValLoss      = {'Total':[]} # Loss on Validation Set
        self.EpochMetric       = {} # Metric on Validation Set
        self.EpochLearningRate = []
        try:
            self.MinLearningRate = scheduler.min_lrs[0]
        except:
            self.MinLearningRate   = 1e-12

        self.Abort_Call_Reason = None
        self.Abort_Call = False
        
        self.ModelStates  = []
        self.ValLossIncreasePatience = Training_Parameters['ValLossIncreasePatience'] if 'ValLossIncreasePatience' in Training_Parameters else 15
        if scheduler is not None:
            if get_scheduler_type(scheduler) == 'ReduceLROnPlateau':
                if scheduler.patience > self.ValLossIncreasePatience:
                    self.ValLossIncreasePatience = scheduler.patience+3
        # Store For Logging
        self.Training_Parameters = Training_Parameters
        self.Model_Parameters = Model_Parameters

    def EpochStart(self,Info):
        self.EpochLearningRate.append(Info['EpochLearningRate'])

    def InBatch(self,Info):
        pass
        # self.BatchLoss.append(Info['BatchLoss'])
        # if Info['BatchWeights'] is not None:
        #     self.BatchWeights.append(Info['BatchWeights'])
        #     self.BatchGrads.append(Info['BatchGrads'])
        
        # # Check if any loss is a NaN
        # if Info['BatchLoss'] != Info['BatchLoss']:
        #     self.Abort_Call_Reason = 'NaN Batch Loss'
        #     self.Abort_Call = True

    def EpochEnd(self,Info):
        # Abort Checks
        # Check if any loss is a NaN
        if Info['EpochLoss'] != Info['EpochLoss']:
            self.Abort_Call_Reason = 'NaN Epoch Loss'
            self.Abort_Call = True
        if Info['EpochValLoss'] != Info['EpochValLoss']:
            self.Abort_Call_Reason = 'NaN Val Loss'
            self.Abort_Call = True
        

        if self.EpochLearningRate[-1] <= self.MinLearningRate:
            self.Abort_Call_Reason = 'Learning Rate too low'
            self.Abort_Call = True
        
        # if np.argmin(self.EpockValLoss['Total']) > Info['EpochValLoss']['Total']:
        #     self.ModelStates.append(copy.deepcopy(Info['ModelState']))
        #     if len(self.ModelStates) > 2:
        #         self.ModelStates.pop(0)
            
        # print('Saving Model State')
        # print(Info['ModelState'])
        # self.ModelStates.append(copy.deepcopy(Info['ModelState']))
        # Printout 
        if Info['EpochLoss']['Total']> 0.0001 and Info['EpochValLoss']['Total'] > 0.0001:
            print(f'Epoch Loss: {Info["EpochLoss"]["Total"]:.4f} | Epoch Val Loss: {Info["EpochValLoss"]["Total"]:.4f}')
            
        else:
            print(f'Epoch Loss: {Info["EpochLoss"]["Total"]:.4e} | Epoch Val Loss: {Info["EpochValLoss"]["Total"]:.4e}')

        for key,val in Info['EpochValLoss'].items():
            if val > 0.0001:
                print(f'{key} Val Loss: {val:.4f} |',end='')
            else:
                print(f'{key} Val Loss: {val:.4e} |',end='')
        print()
        
        for key,val,unit in zip(Info['EpochMetric'].keys(),Info['EpochMetric'].values(),Info['MetricUnits']):
            if unit == 'rad':
                val = val*180/np.pi
                print(f'{key} : {val:.4f} deg |',end='')
            elif unit == 'deg':
                print(f'{key} : {val:.4f} deg |',end='')
            else:
                print(f'{key} : {val:.4f} {unit} |',end='')
        print()

        # Append the losses to the tracker
        for key in Info['EpochLoss'].keys():
            if key not in self.EpochLoss: self.EpochLoss[key] = []
            self.EpochLoss[key].append(Info['EpochLoss'][key])
        
        for key in Info['EpochValLoss'].keys():
            if key not in self.EpochValLoss: self.EpochValLoss[key] = []
            self.EpochValLoss[key].append(Info['EpochValLoss'][key])
        
        for key in Info['EpochMetric'].keys():
                if key not in self.EpochMetric: self.EpochMetric[key] = []
                self.EpochMetric[key].append(Info['EpochMetric'][key])

        # Check if val loss is increasing
        if len(self.EpochValLoss['Total']) - np.argmin(self.EpochValLoss['Total']) > self.ValLossIncreasePatience:
            self.Abort_Call_Reason = 'Val Loss Increasing'
            self.Abort_Call = True

    def AbortHuh(self):
        return self.Abort_Call
    
    def MakeLog(self,PathToLogFolder,ModelName):
        '''
        Makes a log of the training and parameters for the model
        '''

        if PathToLogFolder[-1] != '/':
            PathToLogFolder += '/'
        if not os.path.exists(PathToLogFolder): raise ValueError(f'Path {PathToLogFolder} does not exist')

        # Produce the log
        LogName = PathToLogFolder+ModelName+datetime.now().strftime("%Y-%m-%d_%A_%H-%M")+'_Log.txt'
        if len(self.EpochLoss['Total'])>0:
            with open(LogName,'w') as f:
                f.write(f'Log file for training of {ModelName} \n')
                f.write(f'Start Time: {self.StartTime} \n')
                f.write(f'End   Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n')
                f.write('\n')
                f.write(f'Training Parameters: \n')
                for key in self.Training_Parameters.keys():
                    f.write(f'    {key} : {self.Training_Parameters[key]} \n')
                f.write('\n')
                f.write(f'Model Parameters: \n')
                for key in self.Model_Parameters.keys():
                    f.write(f'    {key} : {self.Model_Parameters[key]} \n')
                f.write('\n')
                f.write(f'Training Log: \n')
                if self.Abort_Call:
                    f.write  (f'    Training Exit Reason: {self.Abort_Call_Reason} \n')
                else: f.write(f'    Training Exit Reason: Reached Final Epoch \n')
                f.write('\n')
                f.write(f'    Final Losses: \n')
                for key in self.EpochLoss.keys():
                    f.write(f'        {key} : {self.EpochLoss[key][-1]} \n')
                f.write(f'    Final Validation Losses: \n')
                for key in self.EpochValLoss.keys():
                    f.write(f'        {key} : {self.EpochValLoss[key][-1]} \n')
                f.write(f'    Final Metrics: \n')
                for key in self.EpochMetric.keys():
                    if key == 'Units':
                        continue
                    if len(self.EpochMetric[key]) == 0:
                        continue
                    f.write(f'        {key} : {self.EpochMetric[key][-1]} \n')
                f.write('\n')
                
                
        else:
            print('No epochs were completed, no log will be made')








def get_scheduler_type(scheduler):
    if isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
        return 'ExponentialLR'
    elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return 'ReduceLROnPlateau'
    # Add more schedulers as elif conditions here as needed.
    else:
        return 'Unknown Scheduler Type'

def PlotOnEpoch(Dataset,Model,Epoch,plotSavePath,device):
    ''' To Plot the Predictions against the Truth on the Validation Set'''
    print('Plotting')
    Dataset.State = 'Val'
    TrainBatchSize = Dataset.BatchSize
    Dataset.BatchSize = 256
    PredList  = []
    TruthList = []
    was_train = Model.training # For safety, though i dont expect to be called during training
    Model.eval()
    with torch.no_grad():
        for _,BatchMains,BatchAux,BatchTruth,_ in Dataset:
            PredList .append(Model(BatchMains,BatchAux).cpu().detach())
            TruthList.append(               BatchTruth .cpu().detach())
        ModelPred  = torch.cat(PredList ,dim=0)
        ModelTruth = torch.cat(TruthList,dim=0)

        NVars = len(Dataset.Truth_Keys)
        fig, axs = plt.subplots(1, NVars, figsize=(10*NVars, 10))
        if NVars == 1: axs = [axs]
        for i in range(NVars):
            axs[i].scatter(ModelTruth[:, i].numpy(), ModelPred[:, i].numpy(), s=1)
            axs[i].plot([min(ModelTruth[:, i]), max(ModelTruth[:, i])], [min(ModelTruth[:, i]), max(ModelTruth[:, i])], 'r')
            axs[i].set_xlabel(f'Truth {Dataset.Truth_Keys[i]}')
            axs[i].set_ylabel(f'Predicted {Dataset.Truth_Keys[i]}')
            axs[i].set_title(Dataset.Truth_Keys[i])

    plt.savefig(f'{plotSavePath}Epoch_{Epoch}.png')
    Dataset.BatchSize = TrainBatchSize
    if was_train:
        Model.train()


def merge_parameters(PriorityParameters, SecondaryParameters):
    return {**SecondaryParameters, **PriorityParameters} # This overwrites the Secondary if there is same Priority key


import inspect
def resolve_calculator(calculator, Training_Parameters):
    if inspect.isclass(calculator):
        return calculator(Training_Parameters)   # initialize
    return calculator                        # already a function / callable


def Train(model,Dataset,optimiser,scheduler,Loss,Validation,Metric,Tracker,\
          Training_Parameters = {}, Model_Parameters = {},**kwargs):
    
    '''
    Usage:
        Call it and it will loop over training, and return the model and Training Track
        Inputs
          Pass in Model, Dataset ,Optimiser, Scheduler : Self Explanatory, Dataset is ProcessingDataset Defined in Dataset2.py
          Loss       : The Loss Function
          Validation : Function that will validate the model and return validation loss
          Metric     : Function that will return the metric
          Tracker    : Custom Class which will be used to track the training, (on EpochEnd can be used for an abort signal)
        Assumed Inputs
          device     : 'cuda' or 'cpu'
          plotOnEpochCompletionPath : Takes a path to save the plots of the predictions against the truth
          SaveBatchInfo  : Save the weights and gradients of the model every batch or Dont Save 
          AutodetectAnomaly : autograd function setting
        # Dictionary Inputs
          Training_Parameters : Dictionary Containing the Training Parameters
            LR       : Learning Rate
            epochs   : Number of Epochs
            BatchSize: Batch Size
            accumulation_steps : Accumulation Steps
            epoch_done : Number of epochs already done (if model is loaded from a checkpoint)
            batchBreak : Number of batches to break after
            ValLossIncreasePatience : Number of epochs to wait before aborting if the validation loss is increasing
            Optimiser : 'Adam' or 'SGD'
            Debug_Mode : If True, will run the training in debug mode
          Model_Parameters : Dictionary Containing the Model Parameters (Model Dependent, this function only stores them in a log file)
    '''

    # Kwargs may be passed separately and for cont. will be appened to both Training and Model Parameters

    Training_Parameters = merge_parameters(Training_Parameters, kwargs)    
    Model_Parameters    = merge_parameters(Model_Parameters   , kwargs)

    # Set Training parameters
    plotOnEpochCompletionPath = Training_Parameters.get('plotOnEpochCompletionPath', None  )
    SaveBatchInfo             = Training_Parameters.get('SaveBatchInfo'            , False )
    AutodetectAnomaly         = Training_Parameters.get('AutodetectAnomaly'        , True  )
    device                    = Training_Parameters.get('device'                   , 'cuda')
    LogPath                   = Training_Parameters.get('LogPath'                  , None  )
    Debug_Mode                = Training_Parameters.get('Debug_Mode'               , False )
    Epochs                    = Training_Parameters.get('epochs'                   , 100   )
    Accumulation_steps        = Training_Parameters.get('accumulation_steps'       , 1     )
    batchBreak                = Training_Parameters.get('batchBreak'               , 1e99  )
    tolerance                 = Training_Parameters.get('validation_tolerance'     , 0.05  ) # Tolerance for validation loss improvement for scheduler step
    if Debug_Mode:
        print(f'    Train Function is setting Batch Break and Epoch counter to 1 for Debug Mode, ignoring Training Parameters')
        batchBreak = 1
        Epochs     = 1
        Dataset.BatchSize = 1


    # Initialise calculator functions
    
    Loss       = resolve_calculator(Loss      ,Training_Parameters)
    Validation = resolve_calculator(Validation,Training_Parameters)
    Metric     = resolve_calculator(Metric    ,Training_Parameters)


    assert SaveBatchInfo == False, 'SaveBatchInfo is not implemented'
    # Initialise Tracker
    if callable(Tracker):
        Tracker = Tracker(scheduler,Training_Parameters = Training_Parameters, Model_Parameters = Model_Parameters)
            
    Min_Val_Loss = 1e99
    
    CUDA_Memory_Fails_Counter = 0

    # Training Loop for Ungraphed Data
    torch.autograd.set_detect_anomaly(AutodetectAnomaly)
    batch_time = None
    for i in range(Epochs):
        print(f'Epoch {i+1}/{Epochs}') # Progress Bar
        Info = {'EpochLearningRate':[param_group['lr'] for param_group in optimiser.param_groups][0]} # Information required for the Tracker at this stage
        if Debug_Mode:
            print(f'    Epoch info for Tracker: {Info}')
        else:
            Tracker.EpochStart(Info)
        
        model.train()
        batchN           = 0
        # Initialise epoch losses
        epoch_loss       = {}
        epoch_loss['Total'] = 0
        
        # Ensure Dataset is in Train State before processing
        Dataset.State = 'Train'
        
        for _, BatchMains, BatchAux, BatchTruth, _ in Dataset: # Dataset Event index is not used
            try:
                # ---- Checks and Batch SetUp ----
                if batchN > batchBreak: break
                if Accumulation_steps == 1 or batchN%Accumulation_steps == 0: optimiser.zero_grad()
                
                # ---- Forward Pass ----
                predictions = model(BatchMains,BatchAux)

                # ---- Backward Pass ----

                Loss_Extra_Info = {'Epoch': i, 'Batch': batchN, 'BatchSize': Dataset.BatchSize, 'keys': Dataset.Truth_Keys } # Extra information that can be used in the loss function if needed
                Loss_Extra_Info = merge_parameters(Loss_Extra_Info, Training_Parameters) 

                losses = Loss(predictions,BatchTruth, Loss_Extra_Info) # Loss will be a dictionary if multiple losses are used (At least one of the losses must be labeled 'Total')
                losses['Total'].backward()
                optimiser.step()
                
                


                # ---- Records and Timing ----
                for key in losses.keys():
                    if key not in epoch_loss: epoch_loss[key] = 0
                    epoch_loss[key] += losses[key].item()
                
                
                if batchN % 100 == 0:
                    if batch_time is None:
                        batch_time = time()
                    else:
                        batch_time = time() - batch_time
                        print(f'Batch {str(batchN).ljust(6)}/{len(Dataset)//Dataset.BatchSize} - Loss: {str(losses["Total"].item())[:6].ljust(6)} - 100 Batch Time: {batch_time:.2f}s',end = '\r')
                        batch_time = time()
                

                    
                if Debug_Mode: print(f' \n   Batch Losses for Tracker: {losses}')
                batchN += 1
                
                if Tracker.AbortHuh():
                    print(f'Batch that broke: {batchN}')
                    break
                
                torch.cuda.empty_cache()

            # ---- Exception Handling ----
            except KeyboardInterrupt:
                print('Keyboard Interrupt, stopping training')
                Tracker.Abort_Call_Reason = 'Keyboard Interrupt'
                Tracker.Abort_Call = True
                break
            except Exception as e:

                if 'CUDA out of memory' in str(e):
                    CUDA_Memory_Fails_Counter += 1
                    print('CUDA out of memory, skipping batch')
                    torch.cuda.empty_cache()
                    if CUDA_Memory_Fails_Counter > 10:
                        print('Too many CUDA out of memory errors, aborting training')
                        Tracker.Abort_Call_Reason = 'Too many CUDA out of memory errors'
                        Tracker.Abort_Call = True
                        break
                if 'NaN' in str(e) or 'nan' in str(e):
                    print(f'NaN Loss detected in batch {batchN}, stopping training')
                    print(f' Losses: {losses}')
                    raise e
                else:
                    print(f'Error in batch {batchN}, Unknown, stopping training')
                    # print(str(e))
                    raise e
                    Tracker.Abort_Call_Reason = f'Error in batch {batchN}, {e}'
                    Tracker.Abort_Call = True
                    break
        
        # ---- Epoch End Checks ----
        print() # Progress Bar
        if Tracker.AbortHuh():
            print(f'Aborting Training from epoch break: {Tracker.Abort_Call_Reason}')
            break
        
        # Epoch End will divide by number of batches
        for key in epoch_loss.keys():
            epoch_loss[key] = epoch_loss[key]/batchN
        try:
            # Validation and Early stopping # metric is a str to be printed
            if Debug_Mode: print(f'\n    Calculating Validation Loss for Tracker')
            # Can Use the same Info as in the last Loss calculation
            val_losses  = Validation(model,Dataset,Loss,Loss_Extra_Info)
            val_loss    = val_losses['Total'] # Needed for scheduler step below
            val_metrics = Metric(model,Dataset,Loss_Extra_Info)
            
            if type(val_metrics) == dict and 'Units' in val_metrics:
                val_metric_units = val_metrics['Units']
                val_metrics.pop('Units')            
            else:
                val_metric_units = Dataset.Truth_Units

        except KeyboardInterrupt:
            print('Keyboard Interrupt, stopping training')
            Tracker.Abort_Call_Reason = 'Keyboard Interrupt during Validation'
            Tracker.Abort_Call = True
        except Exception as e:
            print(f'Error in Validation, Unknown, stopping training')
            # print(str(e))
            Tracker.Abort_Call_Reason = f'Unknown Error in Validation, {e}'
            Tracker.Abort_Call = True
            raise e
            
        if True: # Scheduler Step is done here
            if val_loss < Min_Val_Loss*(1-tolerance):
                Min_Val_Loss = val_loss
            if get_scheduler_type(scheduler) == 'ReduceLROnPlateau':
                scheduler.step(Min_Val_Loss)
            elif get_scheduler_type(scheduler) == 'ExponentialLR':
                scheduler.step()


        Info = {'EpochLoss': epoch_loss, 'EpochValLoss':val_losses,'ModelState':model.state_dict(),\
                'EpochMetric':val_metrics, 'MetricUnits':val_metric_units,
                } # Information required for the Tracker at this stage
        
        if Debug_Mode:
            print(f'    Epoch info for Tracker: \n {Info}')
        else:
            Tracker.EpochEnd(Info)

        

        if Tracker.AbortHuh():
            print(f'Aborting Training: {Tracker.Abort_Call_Reason}')
            break
        if plotOnEpochCompletionPath!=None:
            PlotOnEpoch(Dataset,model,i,plotOnEpochCompletionPath,device)
        print('-------------------------------------')
        
    if LogPath is not None:
        Tracker.MakeLog(LogPath,model.Name)
        print(f'Made Log at {LogPath}')
    return model,Tracker
