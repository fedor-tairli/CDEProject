# Define location of some data files

GoldenRec2019 = '/remote/kennya/data/ADST/Golden/Golden_icrc2019/'
DenseRings    = '/remote/teslaa/tesla/bmanning/data/DenseRings/QGSJETII-04/'
Common        = '/remote/tychodata/ftairli/work/Projects/Common/'

EGFile        = '/remote/kennya/data/ADST/Golden/Golden_icrc2019/2017/01/GoldenRec_2017_01_01_wc.root'
EGMCFile      = '/remote/tychodata/ftairli/work/Projects/Common/EGMCFile.root'
# Define relative paths which will allow me to not fuck up the paths. 
# Assume the code is located in the code directories for the Project files. 

import os
import sys
import inspect
import pickle
import pandas as pd

# Add script directory to PyPath


def get_caller_dir():
    caller_frame = inspect.currentframe().f_back
    caller_file = inspect.getframeinfo(caller_frame).filename
    caller_path = os.path.abspath(caller_file)
    caller_dir = os.path.dirname(caller_path)

    return caller_dir

def get_dir(WantDir,mydir):
    last_dir = os.path.basename(mydir)
    if last_dir != 'Code':
        raise ValueError('Script is not in the "Code" directory of the project')
    else:
        parent_dir = os.path.dirname(mydir)
        newdir = os.path.join(parent_dir, WantDir)
        return newdir

def get_project_name(directory_path):
    directory_path = os.path.normpath(directory_path)
    path_components = directory_path.split(os.sep)
    try:
        projects_index = path_components.index('Projects')
    except ValueError:
        raise ValueError("The 'Projects' directory was not found in the given path.")
    if projects_index + 1 >= len(path_components):
        raise ValueError("The 'Projects' directory does not have a project subdirectory in the given path.")
    project_name = path_components[projects_index + 1]

    return project_name    


class ProjectPaths:

    '''
    This here class is going to be used to share the paths between different scripts in the project
    Every path should be assigned to the 
    '''
    def __init__(self, mypath):
        # Starter Variables
        self.project_name = self.get_project_name(mypath)
        self.project_path = self.get_project_path(mypath)
        self.code_path = os.path.join(self.project_path, 'Code')
        self.data_path = os.path.join(self.project_path, 'Data')
        self.models_path = os.path.join(self.project_path, 'Models')
        self.results_path = os.path.join(self.project_path, 'Results')

        self.save()



        # # Other Paths that need predefining

        # self.ADSTs_dir = None
        # self.RawData   = None





    # Functions Follow

    # Gets the name of the project
    def get_project_name(self,mypath):
        path_components = mypath.split(os.sep)
        try:
            projects_index = path_components.index('Projects')
        except ValueError:
            raise ValueError("The 'Projects' directory was not found in the given path.")
        
        if projects_index + 1 >= len(path_components):
            raise ValueError("The 'Projects' directory does not have a project subdirectory in the given path.")
        
        return path_components[projects_index + 1]


    # Gets the path to the project's main folder
    def get_project_path(self,mypath):
        path_components = mypath.split(os.sep)
        try:
            projects_index = path_components.index('Projects')
        except ValueError:
            raise ValueError("The 'Projects' directory was not found in the given path.")
        
        if projects_index + 1 >= len(path_components):
            raise ValueError("The 'Projects' directory does not have a project subdirectory in the given path.")
        
        # print(path_components)
        return os.path.join(os.sep,*path_components[:projects_index + 2])

    # Auto Saving
    def save(self):
        filename = os.path.join(self.project_path, f"{self.project_name}_project.pkl")
        # print(self.project_path)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Custom attribute setting
    def __setattr__(self, name, value):
        # Check if it already exists
        for attr_name, attr_value in self.__dict__.items():
            if attr_name != name and value == attr_value:
                raise ValueError(f"The value '{value}' already exists in the attribute '{attr_name}'.")

        # If it's a directory, ensure it ends with a path separator
        if os.path.isdir(value) and not value.endswith(os.sep):
            value += os.sep

        super().__setattr__(name, value)
        if name not in ['project_path', 'project_name']:
            self.save()

    def LoadRawData(self,merge=True):
        if hasattr(self,'RawData'): 
            with open(self.RawData,'rb') as RawData:
                return(RawData)
        else:
            with open(self.EventData,'rb') as Event, open(self.StationsData,'rb') as Stations, open(self.EyesData,'rb') as Eyes:
                if merge:
                    Data = pd.merge(Event,Stations, on='EventId')
                    Data = pd.merge(Data,Eyes,on='EventId')
                    return(Data)
                else:
                    return(Event,Stations,Eyes)
                
    def PrintVariables(self,values = False):
        for attr_name, attr_value in self.__dict__.items():
            if values:
                print(attr_name, attr_value)
            else:
                print(attr_name)

def load_ProjectPaths(mypath):
    '''
    Usage: load_ProjectPaths (get_caller_dir())
    Loads Projects Paths    
    
    '''
    mypath = os.path.abspath(mypath)
    path_components = mypath.split(os.sep)

    try:
        projects_index = path_components.index('Projects')
    except ValueError:
        raise ValueError("The 'Projects' directory was not found in the given path.")
    
    if projects_index + 1 >= len(path_components):
        raise ValueError("The 'Projects' directory does not have a project subdirectory in the given path.")
    
    project_name = path_components[projects_index + 1]
    project_path = os.path.join(mypath, '..')
    project_file = os.path.join(project_path, f"{project_name}_project.pkl")

    if os.path.exists(project_file):
        with open(project_file, 'rb') as f:
            project = pickle.load(f)
    else:
        project = ProjectPaths(mypath)
    return project



def get_root_files_recursive(directory):
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            _, ext = os.path.splitext(entry.path)
            if ext.lower() == '.root':
                files.append(entry.path)
        elif entry.is_dir():
            files.extend(get_root_files_recursive(entry.path))
    return files
