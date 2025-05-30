# This script downloads the datafiles from Database to local machine


import os

DoLow = True
###   Setup a percistent SSH Session with :   ssh -M -S /tmp/ssh-control-socket -f -N auger@ipnp40.troja.mff.cuni.cz
###   enter password
###   Can Manually close connection with  :   ssh -S /tmp/ssh-control-socket -O exit auger@ipnp40.troja.mff.cuni.cz

HostName = 'auger@ipnp40.troja.mff.cuni.cz'
rsync_cmd = f'rsync -avz -e "ssh -S /tmp/ssh-control-socket" --progress '
# Open the ssh connection
print('Opening SSH Connection')
os.system(f'ssh -M -S /tmp/ssh-control-socket -f -N {HostName}')
print('Checking Connection')
os.system(f'ssh -S /tmp/ssh-control-socket -O check {HostName}')

if DoLow:
    TopDir_dst = '/remote/tychodata/ftairli/data/CDE/MC/low/'
    TopDir_src = '~/HECO_simulations/low/'
    # If low, 45 batch dirs
    BatchDirs = [f'b{str(i).zfill(2)}' for i in range(1,46)]
    
else:
    TopDir = '/remote/tychodata/ftairli/data/CDE/MC/high/'
    TopDir = '~/HECO_simulations/high/'
    # If high, 6 batch dirs
    BatchDirs = [f'b{str(i).zfill(2)}' for i in range(1,7)]

for BatchDir in BatchDirs:
    print(f'Working on {BatchDir}')
    # Check if the work has already been done
    # Make the directory
    os.system(f'mkdir -p {TopDir_dst + BatchDir}/PCGF/')
    # Download the files with rsync
    print(f'{rsync_cmd}{HostName}:{TopDir_src + BatchDir}/PCGF/ {TopDir_dst + BatchDir}/PCGF')
    os.system(f'{rsync_cmd}{HostName}:{TopDir_src + BatchDir}/PCGF/ {TopDir_dst + BatchDir}/PCGF')
    print()

print('Done')
# Exit the ssh connection
os.system(f'ssh -S /tmp/ssh-control-socket -O exit {HostName}')