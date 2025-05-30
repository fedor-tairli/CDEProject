# Script to move data from tycho to sommers

import os

Tyc_Dir = '/remote/tychodata/ftairli/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'
Som_Dir = '/remote/andromeda/data/Simulations__libraries__MCTask__Offline_v3r99p2a__IdealMC_CORSIKA__Hybrid_CORSIKA76400/SIB23c'

EnergyDirs  = ['18.0_18.5','18.5_19.0','19.0_19.5','19.5_20.0']
EnergyNames = ['180_185'  ,'185_190'  ,'190_195'  ,'195_200'  ]

MassDirs   = ['proton','helium','oxygen','iron']
MassNames  = ['proton','helium','oxygen','iron']

RunNames   = ['Run080']

for EnergyDir,EnergyName in zip(EnergyDirs,EnergyNames):
    for MassDir, MassName in zip(MassDirs,MassNames):
        for RunName in RunNames:
            print(f'Moving {EnergyName} {MassName} {RunName}')
            Full_File_Tyc_Path = Tyc_Dir + '/' + EnergyDir + '/' + MassDir + '/' + f'SIB23c_{EnergyName}_{MassName}_Hybrid_CORSIKA76400_{RunName}.root'
            Full_File_Som_Path = Som_Dir + '/' + EnergyDir + '/' + MassDir + '/' + f'SIB23c_{EnergyName}_{MassName}_Hybrid_CORSIKA76400_{RunName}.root'
            
            # Move the file
            os.system(f'cp {Full_File_Tyc_Path} {Full_File_Som_Path}')
            # If sucess delete old file and soft link to the old place
            if os.path.exists(Full_File_Som_Path):
                os.system(f'rm {Full_File_Tyc_Path}')
                os.system(f'ln -s {Full_File_Som_Path} {Full_File_Tyc_Path}')
                print('Moved:',Full_File_Tyc_Path)
            else:
                print('Failed:',Full_File_Tyc_Path)
            print()
            