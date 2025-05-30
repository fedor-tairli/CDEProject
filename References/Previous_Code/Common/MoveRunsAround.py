import os
import shutil

# Define the directories for the two disks
disk1 = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/'
disk2 = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/'

source_dir = 'SIB23c/'
# dest_dir   = 'temp_SIB23c/'

# Define the energy, mass, and run options

runs            = ['Run010','Run030','Run080','Run090']
energies        = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
energies_inname = ['180_185','185_190','190_195','195_200']
masses          = ['proton','helium','oxygen','iron']


# I need to go over every run, energy and mass. 
# One of the two disks will have a broken softlink, the other will have the file. check which one has the file and make a link to it in the other disk
for run in runs:
    for energy,energy_inname in zip(energies,energies_inname):
        for mass in masses:
            # Construct the possible filenames 
            disk1_filename = f'{disk1}{source_dir}{energy}/{mass}/SIB23c_{energy_inname}_{mass}_Hybrid_CORSIKA76400_{run}.root'
            disk2_filename = f'{disk2}{source_dir}{energy}/{mass}/SIB23c_{energy_inname}_{mass}_Hybrid_CORSIKA76400_{run}.root'

            Disk_with_file = None
            # Check if the file exists in disk1
            if os.path.lexists(disk1_filename) and not os.path.islink(disk1_filename):
                Disk_with_file = 1
                # Check if the file exists in disk2
                if os.path.lexists(disk2_filename):
                    if os.path.islink(disk2_filename):
                        os.remove(disk2_filename)
                        os.symlink(disk1_filename,disk2_filename)
                    else:
                        raise ValueError(f'Both {disk1_filename} and {disk2_filename} are files')
                else:
                    os.symlink(disk1_filename,disk2_filename)
            elif os.path.lexists(disk2_filename) and not os.path.islink(disk2_filename):
                Disk_with_file = 2
                if os.path.lexists(disk1_filename):
                    if os.path.islink(disk1_filename):
                        os.remove(disk1_filename)
                        os.symlink(disk2_filename,disk1_filename)
                    else:
                        raise ValueError(f'Both {disk1_filename} and {disk2_filename} are files')
                else:
                    os.symlink(disk2_filename,disk1_filename)
            else:
                raise FileNotFoundError(f'Neither {disk1_filename} nor {disk2_filename} exist')
            print(f'Link for {run} {energy} {mass} created, with file in disk{Disk_with_file}')               


# def touch(path):
#     with open(path, 'a'):
#         os.utime(path, None)


# def move_files(Run,dest_disk,link_disk,Test=True):
#     for energy,energy_inname in zip(energies,energies_inname):
#         for mass in masses:
#             print()
#             # Construct the filenames and paths
#             filename = f'SIB23c_{energy_inname}_{mass}_Hybrid_CORSIKA76400_{Run}.root'
#             src_link = f'{disk1}{source_dir}{energy}/{mass}/{filename}'
#             # Find file
#             if not os.path.exists(src_link):
#                 raise FileNotFoundError(f'{src_link} does not exist')
#             if os.path.islink(src_link): src_link = os.readlink(src_link)
#             print(f'Found : {src_link}')

#             # Destinations for the files
#             Dest      = dest_disk + dest_dir + f'{energy}/{mass}/'
#             link_Dest = link_disk + dest_dir + f'{energy}/{mass}/'

#             print(f'Dest  : {Dest}{filename}')


#             if Test:
#                 # Prepare destination for copying
#                 os.makedirs(Dest,exist_ok=True)
#                 touch(Dest + filename)

#                 # Link in the other disk
#                 os.makedirs(link_Dest,exist_ok=True)
#                 os.symlink(Dest + filename,link_Dest + filename)


#             else:
#                 print(f'Moving')
#                 shutil.move(src_link,f'{Dest}{filename}')



# Test = False

# # Move Run010 to disk1
# Run      = 'Run010'
# dest_disk = disk1
# link_disk = disk2

# move_files(Run,dest_disk,link_disk,Test)

# # Move Run030 to disk2
# Run      = 'Run030'
# dest_disk = disk2
# link_disk = disk1
# move_files(Run,dest_disk,link_disk,Test)

# # Move Run080 to disk1
# Run      = 'Run080'
# dest_disk = disk1
# link_disk = disk2
# move_files(Run,dest_disk,link_disk,Test)

# # Move Run090 to disk2
# Run      = 'Run090'
# dest_disk = disk2
# link_disk = disk1
# move_files(Run,dest_disk,link_disk,Test)



