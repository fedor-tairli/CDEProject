from adst3 import RecEventProvider,__LocatedLibraries
import paths
import sys
import os
# os.system('clear')



# Paths = paths.ProjectPaths(paths.get_caller_dir())
# for attr_name, attr_value in Paths.__dict__.items():
#     print(attr_name, attr_value)

# print(os.path.join('/remote/tychodata/ftairli/work/Projects/Base/','Project_Paths.pkl'))


for path in __LocatedLibraries():
    print(path)