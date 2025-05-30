import os

os.system('clear')

prim_directory   = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/'
seco_directory   = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c/'

Energies = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
Energies_file = ['180_185','185_190','190_195','195_200']

Masses   = ['proton','helium','oxygen','iron']
Runs     = ['Run010','Run030','Run080','Run090']

Relocate = [True,True,False,False]
irods_directory = '/pauger/Simulations/libraries/MCTask/Offline_v3r99p2a/IdealMC_CORSIKA/Hybrid_CORSIKA76400/SIB23c/'
if __name__ == '__main__':
    # print('Checking Paths')
    # for Energy,Energy_File,Relocate in zip(Energies,Energies_file,Relocate):
    #     for Mass in Masses:
    #         for Run in Runs:
                
    #             filename = f'SIB23c_{Energy_File}_{Mass}_Hybrid_CORSIKA76400_{Run}.root'
    #             filepath = f'{Energy}/{Mass}/'
                
    #             print(f'Download :   {prim_directory}{filepath}{filename}')

    #             if Relocate:
    #                 print(f'Relocate :   {seco_directory}{filepath}{filename}')
    #         print()

    
    # print('Making Directories')
    # for Energy,Energy_File,Relocate in zip(Energies,Energies_file,Relocate):
    #     for Mass in Masses:
    #         for Run in Runs:
                
    #             filename = f'SIB23c_{Energy_File}_{Mass}_Hybrid_CORSIKA76400_{Run}.root'
    #             filepath = f'{Energy}/{Mass}/'
                
    #             os.makedirs(f'{prim_directory}{filepath}',exist_ok=True)
    #             os.makedirs(f'{seco_directory}{filepath}',exist_ok=True)
    
    
    # print('Making Dummy Files')
    # for Energy,Energy_File,Relocate in zip(Energies,Energies_file,Relocate):
    #     for Mass in Masses:
    #         for Run in Runs:
                
    #             filename = f'SIB23c_{Energy_File}_{Mass}_Hybrid_CORSIKA76400_{Run}.root'
    #             filepath = f'{Energy}/{Mass}/'
                
    #             os.system(f'touch {prim_directory}{filepath}{filename}')
    #             os.system(f'touch {seco_directory}{filepath}{filename}')

    # print('Downloading Files')
    # for Energy,Energy_File,Relocate in zip(Energies,Energies_file,Relocate):
    #     for Mass in Masses:
    #         for Run in Runs:
                
    #             finished = False
    #             filename = f'SIB23c_{Energy_File}_{Mass}_Hybrid_CORSIKA76400_{Run}.root'
    #             filepath = f'{Energy}/{Mass}/'
    #             # Create Dummy files first, becasue REASONS
    #             os.system(f'touch {prim_directory}{filepath}{filename}')
    #             os.system(f'touch {seco_directory}{filepath}{filename}')
    #             # Check if file exists and empty
    #             size = os.path.getsize(f'{prim_directory}{filepath}{filename}')
    #             secoSize = os.path.getsize(f'{seco_directory}{filepath}{filename}')
    #             if (size == 0) and (secoSize == 0):
    #                 print('Downloading : ',filename)
    #                 os.system(f'iget -fP {irods_directory}{filepath}{filename} {prim_directory}{filepath}{filename}')
    #             if Relocate and (secoSize == 0):
    #                 print('Relocating  : ',filename)
    #                 os.system(f'mv {prim_directory}{filepath}{filename} {seco_directory}{filepath}{filename}')
    #             print(f'Finished with {filename}')

    print('Softlinking Files')
    for Energy,Energy_File,Relocate in zip(Energies,Energies_file,Relocate):
        for Mass in Masses:
            for Run in Runs:
                filename = f'SIB23c_{Energy_File}_{Mass}_Hybrid_CORSIKA76400_{Run}.root'
                filepath = f'{Energy}/{Mass}/'
                if Relocate:
                    if os.path.exists(f'{prim_directory}{filepath}{filename}') and os.path.getsize(f'{prim_directory}{filepath}{filename}') == 0:
                        os.system(f'rm -f {prim_directory}{filepath}{filename}')
                    
                        os.system(f'ln -s {seco_directory}{filepath}{filename} {prim_directory}{filepath}{filename}')
                    print(f'Linked : {filename}')
                else:
                    print(f'Not Linking : {filename}')

