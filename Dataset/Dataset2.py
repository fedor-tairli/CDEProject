##########################################################
#                                                         #
#          This File Defines the Dataset Object           #
#                 A more general version                  #
#                  Try to use snakecase                   #
#                                                         #
###########################################################

import os 
import time

import torch
from typing import Union



class EventContainer:
    '''Class to hold the event data
    2.0 Version
    Should work More generally for newer things
    Expect to use the older dataset for the older things. May not be backwards compatible.
    '''
    # Basically a mini copy of the dataset container to hold the event data
    # Try to use snakecase
    # Class Data Values

    def __init__(self,Event_level_keys,Pixel_level_keys,HasTraces = False,Trace_length = 1000):
        # Meta Data
        self.Number_of_pixels = 0 # Also Serves as the active pixel counter
        self.Event_level_keys = Event_level_keys
        self.Pixel_level_keys = Pixel_level_keys
        self.HasTraces = HasTraces

        # Data Arrays
        self._event_data = torch.zeros((1,len(self.Event_level_keys)))
        self._pixel_data = torch.zeros((1,len(self.Pixel_level_keys)))
        self._trace_data = torch.zeros((1,Trace_length)) # Default trace length is 1000

    def __len__(self):
        return self.Number_of_pixels
    
    def add_from_dataset(self,event_data,pixel_data,trace_data):
        '''Add data from the dataset'''
        self._event_data = event_data
        self._pixel_data = pixel_data
        self._trace_data = trace_data

    def add_event_value(self,key,value):
        '''Add a value to the event'''
        self._event_data[0,self.Event_level_keys[key]] = value
    
    def add_pixel_value(self,key,value):
        '''Add a value to the pixel'''
        self._pixel_data[self.Number_of_pixels,self.Pixel_level_keys[key]] = value
    
    def add_trace_values(self,trace):
        '''Add a trace to the pixel'''
        self._trace_data[self.Number_of_pixels,:] = trace
    
    def next_pixel(self):
        '''Move to the next pixel'''
        self.Number_of_pixels += 1
        if self.Number_of_pixels >= self._pixel_data.shape[0]:
            # Expand the arrays by 10 pixels
            self._pixel_data = torch.cat((self._pixel_data, torch.zeros((10,len(self.Pixel_level_keys))) ),0)
            self._trace_data = torch.cat((self._trace_data, torch.zeros((10,self._trace_data.shape[1] )) ),0)


    def get_value(self,key):
        '''Get a value from the event'''
        return self._event_data[0,self.Event_level_keys[key]].clone()
    
    def get_pixel_values(self,key):
        '''Get a value from the pixel'''
        return self._pixel_data[:,self.Pixel_level_keys[key]].clone()

    def get_trace_values(self):
        '''Get a trace from the pixel'''
        return self._trace_data.clone()

    def clean_empty(self):
        '''Crop the data to the correct size'''
        self._pixel_data = self._pixel_data[:self.Number_of_pixels,:]
        self._trace_data = self._trace_data[:self.Number_of_pixels,:]




class ProcessingDatasetContainer:
    '''Class to hold the dataset's values ready for processing via machine
    Dataset2.0 Version
    Single Object for all model types
    Will Try to make this a subclass of the processing dataset
    '''
    # Need to use
    def __init__(self,DirPath = None):
        if DirPath is not None:
            if DirPath.endswith('/'): DirPath = DirPath[:-1]
            DS = torch.load(DirPath+'/CurrentProcessingDataset.pt')
            self.__dict__ = DS.__dict__
            self.AssignIndices()

        else:
            self.Name = 'Unassigned'        # Name of the dataset
            self.State = 'Build'            # State of the dataset, Build/Static or Train, Val, Test  
            self.GraphData = False          # Flag to be used if the data is for a graph and return the graph data
            self.RandomIter      = True     # Flag to check if the data is to be iterated randomly
            self.BatchSize       = 1        # Batch Size for iteration in batches, default is unbatched
            # Data Arrays
            self._Main  = []                # Main Data, Will be an N by ... tensor (may be 2D may be 3D tensor)
                                            # If the model is layered there may be multiple main data to be returned as tuple
                                            # Usually will contain pixel data
            self._Graph = []                # Not necessarily graph data, used for sparse data. 
                                            # cont. A triple nested list of whatever extra data is required. Replaces the Main Data for Graphs
            self._Aux   = None              # Auxillary Data, Will be an N by A tensor,Usually will contain the event global values
            self._Truth = None              # Truth Data            , Will be an N by T Tensor
            self._Rec   = None              # Reconstruction Results, same as truth            , Used to compare the results
            self._EventIds = None           # will be the tensor of event ids
            self._MetaData = None           # Will be the event metadata -  values like primary, energy, etc. NOT inp/outp


            self.Unnormalise_Truth  = None  # Will be the function which unnormalises the Truth/Rec/Pred data
            self.Truth_Keys         = ()    # Will be the names of the Truth Values
            self.Truth_Units        = ()    # Will be the units of the Truth Values once unnormalised

            # Indeces Definitions                
            self._TrainIndeces = None       # Indeces for the training data
            self._ValIndeces   = None       # Indeces for the validation data
            self._TestIndeces  = None       # Indeces for the testing data

    # Functions to add the data will be defined in a separate file which will depend on the model type
    # This will allow for the addition of the data in a more general way
    

    def set_Name(self,Name):
        '''Set the name of the dataset'''
        self.Name = Name

    def __len__(self):
        return len(self.ActiveIndeces)
    @property
    def ActiveIndeces(self):
        '''Returns the active indeces'''
        if self.State in ['Static','Build']:
            return torch.arange(len(self._Truth))
        elif self.State == 'Train':
            return self._TrainIndeces
        elif self.State == 'Val':
            return self._ValIndeces
        elif self.State == 'Test':
            return self._TestIndeces
        else:
            raise ValueError(f'Unknown State: {self.State}')

    def AssignIndices(self,distribution = [0.7,0.2,0.1],seed = 1234):
        self.State = 'Static'
        if seed is not None: torch.manual_seed(seed)

        Indices = torch.randperm(len(self))
        # Split the indeces
        self._TrainIndeces = Indices[:int(distribution[0]*len(self))]
        self._ValIndeces   = Indices[int(distribution[0]*len(self)):int((distribution[0]+distribution[1])*len(self))]
        self._TestIndeces  = Indices[int((distribution[0]+distribution[1])*len(self)):]

    def SetState(self,State):
        '''Set the state of the dataset'''
        assert State in ['Build','Static','Train','Val','Test'], "Unknown State"
        self.State = State
        
    def __iter__(self): # Iterates over the active indeces
        
        # Change the order of the indeces if random iteration is selected
        if self.RandomIter:
            NewOrder = torch.randperm(len(self))
            Indices = self.ActiveIndeces[NewOrder]
        else:
            Indices = self.ActiveIndeces
        
        if self.GraphData:
            Index = 0 
            while Index < len(Indices):
                # If the data is for a graph
                # Create a tuple of empty tensors to hold the main data
                Aux      =       torch.zeros((self.BatchSize,*  self._Aux.shape  [1:]))
                Truth    =       torch.zeros((self.BatchSize,*  self._Truth.shape[1:]))
                Rec      =       torch.zeros((self.BatchSize,*  self._Rec.shape  [1:]))
                EventIds =       torch.zeros( self.BatchSize)
                Graphs   = []

                # Fill the tensors
                for i in range(self.BatchSize):
                    if Index >= len(Indices):
                        Aux      = Aux[:i]
                        Truth    = Truth[:i]
                        Rec      = Rec[:i]
                        EventIds = EventIds[:i]
                        break
                    # Fill the tensors
                    Graphs               .append( self._Graph   [Indices[Index]])
                    EventIds[i]                 = self._EventIds[Indices[Index]]
                    Aux[i]                      = self._Aux     [Indices[Index]]
                    Truth[i]                    = self._Truth   [Indices[Index]]
                    Rec[i]                      = self._Rec     [Indices[Index]]
                    Index += 1
                yield EventIds, Graphs, Aux, Truth, Rec

        else:
            Index = 0
            while Index < len(Indices):
                    
                # If the data is not for a graph
                # Create a tuple of empty tensors to hold the main data
                Mains    = tuple(torch.zeros((self.BatchSize,*       data.shape  [1:])) for data in self._Main)
                Aux      =       torch.zeros((self.BatchSize,*  self._Aux.shape  [1:]))
                Truth    =       torch.zeros((self.BatchSize,*  self._Truth.shape[1:]))
                Rec      =       torch.zeros((self.BatchSize,*  self._Rec.shape  [1:]))
                EventIds =       torch.zeros( self.BatchSize)
                # Fill the tensors
                for i in range(self.BatchSize):
                    if Index >= len(Indices): # If the indeces are finished Reduce the last batch to the correct size
                        Mains    = tuple(data[:i] for data in Mains)
                        Aux      = Aux[:i]
                        Truth    = Truth[:i]
                        Rec      = Rec[:i]
                        EventIds = EventIds[:i]
                        break
                    # Fill the tensors
                    EventIds[i]                 = self._EventIds[Indices[Index]]
                    for j in range(len(Mains)):
                        Mains[j][i]             = self._Main[j] [Indices[Index]]
                    Aux[i]                      = self._Aux     [Indices[Index]]
                    Truth[i]                    = self._Truth   [Indices[Index]]
                    Rec[i]                      = self._Rec     [Indices[Index]]
                    Index += 1
                yield EventIds, Mains, Aux, Truth, Rec

    def Save(self,DirPath,Name=None):
        '''Save the dataset to a file to make the loading quicker, so that I dont have to reprocess the data every time'''
        if DirPath.endswith('/'): DirPath = DirPath[:-1]
        if not os.path.exists(DirPath): os.makedirs(DirPath)
        # Save the dataset
        if Name is None: torch.save(self,DirPath+'/CurrentProcessingDataset.pt')
        else:            torch.save(self,DirPath+f'/{Name}.pt')

    






class DatasetContainer:
    '''Class to hold the dataset
    2.0 Version
    Should work More generally for newer things
    Expect to use the older dataset for the older things. May not be backwards compatible.
    '''
    
    # Class Data Values
    Average_pixels_per_event = 30 # Overestimation of the average number of pixels per event for reading purposes
    Trace_length             = 100 # Default Trace length is 1000 will only store Triggered Signal Region
    def __init__(self,ExpectTraces:bool = False):
        
        # assert type(ExpectTraces) == bool, "ExpectTraces should be a boolean"
        self.Name = 'Unassigned' # Name of the dataset
        # Meta Data
        self.Number_of_events = 0 # These get increased during data read
        self.Number_of_pixels = 0 # These get increased during data read
        self.Event_level_keys = {} # Name of value : its position in the array
        self.Pixel_level_keys = {} # Name of value : its position in the array
        self.HasTraces        = ExpectTraces # Does have traces or not?
        self.All_keys_are_added = False # Just a flag to check if all keys are added

        # Functional Information
        self.Normalisation_functions_event   = {} # Normalisation functions for given keys
        self.Normalisation_functions_pixel   = {} # (not all will have the function, eg. ID values)
        self.Unnormalisation_functions_event = {}
        self.Unnormalisation_functions_pixel = {}
        

        # Data Arrays 
        self._event_data = torch.tensor([[]]) # 2D tensor for each event, 0thD is the event number, 1stD is the value according to the event level keys
        self._pixel_data = torch.tensor([[]]) # 2D tensor for each pixel, 0thD is the pixel number, 1stD is the value according to the pixel level keys
        self._trace_data = torch.tensor([[]]) # 2D tensor for each pixel, 0thD is the pixel number, 1stD is the trace data
        self._event_pixel_position = torch.tensor([[],[]]) # N by 2 tensor for pixel positions in the event, 0 is start of the event, 1 is end of the event


        # Extra Properties
        self.IDlist = None


    def __len__(self):
        return self.Number_of_events
    
    def add_event_value(self,key):
        '''Add a new event level key'''
        # Check if the key is already added
        if key in self.Event_level_keys:
            print("Key already added")
        else:
            self.Event_level_keys[key] = len(self.Event_level_keys)

    def add_pixel_value(self,key):
        '''Add a new pixel level key'''
        # Check if the key is already added
        if key in self.Pixel_level_keys:
            print("Key already added")
        else:
            self.Pixel_level_keys[key] = len(self.Pixel_level_keys)
    # Adds the functions to normalise and unnormalise the data
    def add_normalisation_event(self,key,func):
        '''Add a normalisation function for a key'''
        assert callable(func), "Function is not callable"
        assert key in self.Event_level_keys, "Key not found"
        self.Normalisation_functions_event[key] = func
    
    def add_unnormalisation_event(self,key,func):
        '''Add a unnormalisation function for a key'''
        assert callable(func), "Function is not callable"
        assert key in self.Event_level_keys, "Key not found"
        self.Unnormalisation_functions_event[key] = func

    def add_normalisation_pixel(self,key,func):
        '''Add a normalisation function for a key'''
        assert callable(func), "Function is not callable"
        assert key in self.Pixel_level_keys, "Key not found"
        self.Normalisation_functions_pixel[key] = func

    def add_unnormalisation_pixel(self,key,func):
        '''Add a unnormalisation function for a key'''
        assert callable(func), "Function is not callable"
        assert key in self.Pixel_level_keys, "Key not found"
        self.Unnormalisation_functions_pixel[key] = func

    def get_blank_event(self):
        '''Return a blank event to be filled up and added to the dataset'''
        return EventContainer(self.Event_level_keys,self.Pixel_level_keys,self.HasTraces,self.Trace_length)
    
    def add_name(self,name):
        '''Add a name to the dataset'''
        self.Name = name

    # To avoid long data reading times, dont use torch.cat()
    # Instead initialise the tensors with zeros and fill them (at the end cut to size)
    def preallocate_data(self,Expected_events):
        '''Preallocate the data arrays'''
        self._event_data           = torch.zeros((Expected_events                              ,len(self.Event_level_keys)))
        self._event_pixel_position = torch.zeros((Expected_events                              ,2                         ))
        self._pixel_data           = torch.zeros((Expected_events*self.Average_pixels_per_event,len(self.Pixel_level_keys)))
        self._trace_data           = torch.zeros((Expected_events*self.Average_pixels_per_event,self.Trace_length         ))

    def add_event(self,event:EventContainer):
        '''Add an event to the dataset'''
        # Check if there is enough space in the dataset to add the event
        if (self.Number_of_events >= self._event_data.shape[0]) or (self.Number_of_pixels >= self._pixel_data.shape[0]):
            # Expand the arrays by 1000 events
            self._event_data = torch.cat((self._event_data, torch.zeros((1000,len(self.Event_level_keys))) ),0)
            self._pixel_data = torch.cat((self._pixel_data, torch.zeros((1000*self.Average_pixels_per_event,len(self.Pixel_level_keys))) ),0)
            self._trace_data = torch.cat((self._trace_data, torch.zeros((1000*self.Average_pixels_per_event,self.Trace_length)) ),0)
            self._event_pixel_position = torch.cat((self._event_pixel_position, torch.zeros((1000,2)) ),0)

        # Add the event data
        self._event_data[self.Number_of_events,:] = event._event_data
        self._pixel_data[self.Number_of_pixels:self.Number_of_pixels+len(event),:] = event._pixel_data[:len(event),:] # to avoid using clean_empty to save time (i think)
        self._trace_data[self.Number_of_pixels:self.Number_of_pixels+len(event),:] = event._trace_data[:len(event),:] # same as above
        self._event_pixel_position[self.Number_of_events,0] = self.Number_of_pixels
        self._event_pixel_position[self.Number_of_events,1] = self.Number_of_pixels + len(event)
        # Increase the counters
        self.Number_of_events += 1
        self.Number_of_pixels += len(event)

    def get_values(self,key):
        '''Get the values from the dataset'''
        return self._event_data[:,self.Event_level_keys[key]].clone()
    
    def get_pixel_values(self,key):
        '''Get the values from the dataset'''
        return self._pixel_data[:,self.Pixel_level_keys[key]].clone()
    
    def get_trace_values(self):
        '''Get the values from the dataset'''
        return self._trace_data.clone()
    
    def get_event_by_index(self,index):
        '''Get the event from the dataset'''
        event = EventContainer(self.Event_level_keys,self.Pixel_level_keys,self.HasTraces,self.Trace_length)
        event.add_from_dataset(self._event_data[index,:].unsqueeze(0),\
                               self._pixel_data[int(self._event_pixel_position[index,0]):int(self._event_pixel_position[index,1]),:],\
                               self._trace_data[int(self._event_pixel_position[index,0]):int(self._event_pixel_position[index,1]),:])
        return event
    
    def get_event_by_id(self,ID):
        raise NotImplementedError("Not Implemented Yet")

    def clean_empty(self):
        '''Crop the data to the correct size'''
        self._event_data           = self._event_data          [:self.Number_of_events,:]
        self._event_pixel_position = self._event_pixel_position[:self.Number_of_events,:]
        self._pixel_data           = self._pixel_data          [:self.Number_of_pixels,:]
        self._trace_data           = self._trace_data          [:self.Number_of_pixels,:]

    # Save and Load                                                
    def Save(self,DirPath):
        '''Save the dataset to a file'''
        assert self.Name != 'Unassigned', "Name not assigned"
        # Check if the directory exists
        if DirPath.endswith('/'):
            DirPath = DirPath[:-1]
        if not os.path.exists(DirPath):
            os.makedirs(DirPath)
        # Save the dataset Separately
        self.clean_empty()
        # Save Meta Data
        MetaData = {'Name':self.Name,\
                    'Number_of_events':self.Number_of_events,\
                    'Number_of_pixels':self.Number_of_pixels,\
                    'Event_level_keys':self.Event_level_keys,\
                    'Pixel_level_keys':self.Pixel_level_keys,\
                    'HasTraces':self.HasTraces,\
                    'All_keys_are_added':self.All_keys_are_added,\
                    'Normalisation_functions_event':self.Normalisation_functions_event,\
                    'Normalisation_functions_pixel':self.Normalisation_functions_pixel,\
                    'Unnormalisation_functions_event':self.Unnormalisation_functions_event,\
                    'Unnormalisation_functions_pixel':self.Unnormalisation_functions_pixel}
        torch.save(MetaData,DirPath+f'/{self.Name}_MetaData.pt')

        # Save Data
        torch.save(self._event_data,DirPath+f'/{self.Name}_EventData.pt')
        torch.save(self._pixel_data,DirPath+f'/{self.Name}_PixelData.pt')
        torch.save(self._trace_data,DirPath+f'/{self.Name}_TraceData.pt')
        torch.save(self._event_pixel_position,DirPath+f'/{self.Name}_EventPixelPosition.pt')

    def Load(self, DirPath, Names: Union[list, str], LoadTraces=False): # Do multiloading straight away
        '''Load the dataset from a file'''
        if DirPath.endswith('/'):
            DirPath = DirPath[:-1]
        self.HasTraces = LoadTraces
        if type(Names) == str:
            Names = [Names]
        for Name in Names:
            print(f'Loading {Name}')
            # Load Meta Data
            MetaData = torch.load(DirPath+f'/{Name}_MetaData.pt')
            if Name == 'Unassigned': self.Name = MetaData['Name'] 
            else                   : self.Name+='_'+MetaData['Name']
            # These Should be the same for all datasets
            self.Event_level_keys = MetaData['Event_level_keys']
            self.Pixel_level_keys = MetaData['Pixel_level_keys']
            self.Normalisation_functions_event = MetaData['Normalisation_functions_event']
            self.Normalisation_functions_pixel = MetaData['Normalisation_functions_pixel']
            self.Unnormalisation_functions_event = MetaData['Unnormalisation_functions_event']
            self.Unnormalisation_functions_pixel = MetaData['Unnormalisation_functions_pixel']

            # Load Data
            self._event_data = torch.load(DirPath+f'/{Name}_EventData.pt') if (self._event_data.numel() == 0) else torch.cat((self._event_data,torch.load(DirPath+f'/{Name}_EventData.pt')),0)
            self._pixel_data = torch.load(DirPath+f'/{Name}_PixelData.pt') if (self._pixel_data.numel() == 0) else torch.cat((self._pixel_data,torch.load(DirPath+f'/{Name}_PixelData.pt')),0)

            if LoadTraces :
                self._trace_data = torch.load(DirPath+f'/{Name}_TraceData.pt') if (self._trace_data.numel() == 0) else torch.cat((self._trace_data,torch.load(DirPath+f'/{Name}_TraceData.pt')),0)
            current_pixel_position = torch.load(DirPath+f'/{Name}_EventPixelPosition.pt')
            current_pixel_position += self.Number_of_pixels
            self._event_pixel_position = current_pixel_position if (self._event_pixel_position.numel() ==0 ) else torch.cat((self._event_pixel_position,current_pixel_position),0)
        
            # Adjust counters
            self.Number_of_events += MetaData['Number_of_events']
            self.Number_of_pixels += MetaData['Number_of_pixels']
        self.All_keys_are_added = True
        
    # Some special Functions
    @property
    def IDs(self):
        '''Return the IDs of the events'''
        if self.IDlist != None: return self.IDlist
        else:
            First  = self._event_data[:,self.Event_level_keys['EventID_1/2']].to(torch.int64)
            Second = self._event_data[:,self.Event_level_keys['EventID_2/2']].to(torch.int64)
            # Chekc that Second is less than 10000, where it is equal to 10000, set to zero
            Second[Second == 10000] = 0
            self.IDlist = (First*10000 + Second).tolist()
            return self.IDlist
        
    def __iter__(self):
        '''Iterate over the events'''
        for i in range(self.Number_of_events):
            yield self.get_event_by_index(i)


    def LoadFromCSV_Preprocessed(self,DatasetName:str,NumberOfEvents:int,NumberOfPixels:int,\
                          EventLevelKeys:dict,PixelLevelKeys:dict,EventPixelPosition:torch.tensor,\
                          EventLevelData:torch.tensor,PixelLevelData:torch.tensor,Traces:torch.tensor=None,**kwargs):
        
        '''Load the dataset from CSV, kinda*
        Kinda, because this is meant to accept a preporcessed data in form of the above tensors (not all are tensors)
        Simply Run a couple of checks and assign the values to the dataset.
        '''

        assert DatasetName != None, "Dataset Name not assigned"
        assert len(EventLevelKeys) == EventLevelData.shape[1], "Event Level Keys and Data do not match"
        assert len(PixelLevelKeys) == PixelLevelData.shape[1], "Pixel Level Keys and Data do not match"
        assert NumberOfEvents == EventLevelData.shape[0]     , "Number of Events and Data do not match"
        assert NumberOfEvents == EventPixelPosition.shape[0], "Number of Events and Pixel Position do not match"
        assert NumberOfPixels == PixelLevelData.shape[0]     , "Number of Pixels and Data do not match"
        if not Traces is None: assert NumberOfPixels == Traces.shape[0], "Number of Pixels and Traces do not match"

        # Assign the values to the dataset
        self.Name = DatasetName
        self.Number_of_events = NumberOfEvents
        self.Number_of_pixels = NumberOfPixels
        self.Event_level_keys = EventLevelKeys
        self.Pixel_level_keys = PixelLevelKeys
        if not Traces is None: self.HasTraces = True
        self.All_keys_are_added = True

        # Unnormalisation functions to be assigned separately

        # Assign the data to the dataset
        self._event_data           = EventLevelData
        self._pixel_data           = PixelLevelData
        if self.HasTraces: self._trace_data = Traces
        self._event_pixel_position = EventPixelPosition

        # Leave IDs List As None For now