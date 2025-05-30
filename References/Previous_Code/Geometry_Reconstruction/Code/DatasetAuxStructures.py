###########################################################
#                                                         #
#        Here we define event and aux functions           #
#              Used with specific analysis                #
#               Internal Structure is different           #
#               Function Names are uniforms               #
#                                                         #
###########################################################


import torch
import os
# import networkx as nx
from scipy.sparse.csgraph import connected_components


def IsNeighboursLooseTime(Pixels,selfloop=False,TimeTollerance = 0.3):
    return 'Deprecated'
    # 0 - Phi | 1 - Theta | 2 - Time
    PhiDiffs       = torch.abs(Pixels[:,0].unsqueeze(0) - Pixels[:,0].unsqueeze(1))
    AverageTheta   = (Pixels[:,1].unsqueeze(0) + Pixels[:,1].unsqueeze(1))/2
    Adjustment     = torch.cos((AverageTheta-30)*torch.pi/60)**2
    PhiDiffs      -= Adjustment*(AverageTheta<60)
    ThetaDiffs     = torch.abs(Pixels[:,1].unsqueeze(0) - Pixels[:,1].unsqueeze(1))
    PhiAdjacency   = PhiDiffs<2
    ThetaAdjacency = ThetaDiffs<1.2*torch.sqrt(torch.tensor(3.0))
    # Find Already Existing Neighbours
    Adjacency      = PhiAdjacency*ThetaAdjacency
    Adjacency[torch.eye(Adjacency.shape[0]).bool()] = selfloop
    TimeDiffs      = torch.abs(Pixels[:,2].unsqueeze(0) - Pixels[:,2].unsqueeze(1))
    SortedTimeDiffs = torch.sort(TimeDiffs.flatten()[Adjacency.flatten()],descending=True).values[0::2]
    # for diff in SortedTimeDiffs:
    #     if 
    #     print('->',diff.item())
    # print(SortedTimeDiffs)

    for i,time in enumerate(SortedTimeDiffs):
        print(time.item(),SortedTimeDiffs[i+1].item(),time.item()-SortedTimeDiffs[i+1].item())
        if time > 7000: continue # Hard Cutoff Upper limit
        if time < 300 : continue # Hard Cutoff Lower limit

        if time-SortedTimeDiffs[i+1]<TimeTollerance*time: # if there exists a Dt such that there is another within 10% of it, thats our cutoff
            print(f'--> {time.item()}')
            TimeCutoff = time
            break
        else:
            print(f'--- {time.item()}')
    TimeAdjacency = TimeDiffs<=TimeCutoff
    Adjacency      = Adjacency*TimeAdjacency
    # Connect the Unconnected Chunks
    # for i in range(0,TimeDiffs.shape[0]):
    #     if torch.sum(Adjacency[i,:]) == 0: continue # I has no neighbours
    #     if Adjacency[i,i+1] == 0: # Check if there is any more chunks after I (I will be the last element of the chunk with this if)
    #         for j in range(i+1,TimeDiffs.shape[0]):
    #             if torch.sum(Adjacency[j,:]) != 0:  # J will be the first element of the next chunk
    #                 Adjacency[i,j] = True
    #                 break # go to next I
    return Adjacency

def IsNeighboursThoroughTime(Pixels,selfloop=False,TimeTollerance = 1):
    return 'Deprecated'
    # 0 - Phi | 1 - Theta | 2 - Time
    # Uses Theta to allow for larger gap of Phi at high elevations, adding about 1 deg at 
    PhiDiffs       = torch.abs(Pixels[:,0].unsqueeze(0) - Pixels[:,0].unsqueeze(1))
    AverageTheta   = (Pixels[:,1].unsqueeze(0) + Pixels[:,1].unsqueeze(1))/2
    Adjustment     = torch.cos((AverageTheta-30)*torch.pi/60)**2
    PhiDiffs      -= Adjustment*(AverageTheta<60)
    ThetaDiffs     = torch.abs(Pixels[:,1].unsqueeze(0) - Pixels[:,1].unsqueeze(1))
    PhiAdjacency   = PhiDiffs<2
    ThetaAdjacency = ThetaDiffs<1.2*torch.sqrt(torch.tensor(3.0))
    # Distance Adjacency Only
    Adjacency      = PhiAdjacency*ThetaAdjacency
    Adjacency[torch.eye(Adjacency.shape[0]).bool()] = selfloop
    TimeDiffs      = torch.abs(Pixels[:,2].unsqueeze(0) - Pixels[:,2].unsqueeze(1))
    
    for i in range(0,TimeDiffs.shape[0]):
        for j in range(0,TimeDiffs.shape[0]):
            if not Adjacency[i,j]: continue
            if TimeDiffs[i,j] < 300: continue # Hard Cutoff Lower limit
            if TimeDiffs[i,j] > 7000:
                Adjacency[i,j] = False # Hard Cutoff
                continue
            
            print('')
            print(f'Testing {i} and {j} | ',end = '')
            # print('')
            # print(f' Testing {i} and {j} with time diff {str(TimeDiffs[i,j].item())[:5].ljust(5)}',end = '')
            # Find Average time of j's neighbours
            # if   i==0 :               jNeighbours = TimeDiffs[:,j][torch.cat((torch.tensor([0]),Adjacency[i+1:,j]))]
            # elif i==len(TimeDiffs)-1: jNeighbours = TimeDiffs[:,j][torch.cat((Adjacency[:i,j],torch.tensor([0])))]
            # else:                     jNeighbours = TimeDiffs[:,j][torch.cat((Adjacency[:i,j],torch.tensor([0]),Adjacency[i+1:,j]))]
            
            jNeighbours = TimeDiffs[:,j][torch.cat((Adjacency[:i,j],torch.tensor([0]),Adjacency[i+1:,j])).type(torch.bool)]
            jNeighbours = torch.unique(jNeighbours) # Contains all neighbours of J that are not I 
            if len(jNeighbours) < 1: continue# Check if there are any neighbours (not I)
            if TimeDiffs[i,j] > (TimeTollerance+1)*torch.amax(jNeighbours): Adjacency[i,j] = False # Should Delete Most of the Accidents
            print(f' Pass = {Adjacency[i,j]} timeDiff = {TimeDiffs[i,j]}',end = '')
            
            # print(f' Pass = {Adjacency[i,j]} {jNeighbours.tolist()}',end = '')
            Adjacency = Adjacency* Adjacency.T
    # Connect the Unconnected Chunks
    # for i in range(0,TimeDiffs.shape[0]):
    #     if torch.sum(Adjacency[i,:]) == 0: continue # I has no neighbours
    #     if Adjacency[i,i+1] == 0: # Check if there is any more chunks after I (I will be the last element of the chunk with this if)
    #         for j in range(i+1,TimeDiffs.shape[0]):
    #             if torch.sum(Adjacency[j,:]) != 0:  # J will be the first element of the next chunk
    #                 Adjacency[i,j] = True
    #                 Adjacency[j,i] = True
    #                 break # go to next I
    return Adjacency

# If changing defaults here dont forget to change them in the GetProcEvent function
def IsNeighboursDurationTime(Pixels,selfloop=False,minchunksize = 2,DurationScale = 1.5):
    '''
    Returns the Adjacency matrix depending on Theta,Phi and Time, Using Pulse Duration
    Inputs are Pixels, a torch tensor of shape [N,4] 
               selfloop = False, whether to allow selfloops in the graph
               minchunksize = 2, minimum number of pixels in a chunk
               DurationScale = 1.5, allowed gap between pulses in units of pulse duration 1 gives 1xAveragePulseDuration
    '''
    # 0 - Phi | 1 - Theta | 2 - Time | 3 - PulseDuration
    PulseDuration = Pixels[:,3] /100 # Convert to bins (100ns units)
    # Uses Theta to allow for larger gap of Phi at high elevations (SPHERES)
    # adding about 1 deg at 60 deg elevation 
    # Reducing to 0 at 30 deg elevation
    PhiDiffs       = torch.abs(Pixels[:,0].unsqueeze(0) - Pixels[:,0].unsqueeze(1))
    AverageTheta   = (Pixels[:,1].unsqueeze(0) + Pixels[:,1].unsqueeze(1))/2 # Replace with Unaveraged theta? may be faster
    Adjustment     = torch.cos((AverageTheta-30)*torch.pi/60)**2
    PhiDiffs      -= Adjustment*(AverageTheta<60)
    ThetaDiffs     = torch.abs(Pixels[:,1].unsqueeze(0) - Pixels[:,1].unsqueeze(1))
    PhiAdjacency   = PhiDiffs<2
    ThetaAdjacency = ThetaDiffs<1.2*torch.sqrt(torch.tensor(3.0))
    # Distance Adjacency Only
    Adjacency      = PhiAdjacency*ThetaAdjacency
    Adjacency[torch.eye(Adjacency.shape[0]).bool()] = selfloop
    TimeDiffs      = torch.abs(Pixels[:,2].unsqueeze(0) - Pixels[:,2].unsqueeze(1))
    PulseDurationGap = PulseDuration.unsqueeze(0)/2 + PulseDuration.unsqueeze(1)/2
    PulseDurationGap *= DurationScale
    for i in range(0,TimeDiffs.shape[0]):
        for j in range(0,TimeDiffs.shape[0]):
            if not Adjacency[i,j]: continue
            # print(f'Testing {i} and {j} | TimeDiff = {str(TimeDiffs[i,j].item())[:6].ljust(6)} | PulseDurationGap = {str(PulseDurationGap[i,j].item())[:6].ljust(6)} ',end = '')
            if TimeDiffs[i,j] < PulseDurationGap[i,j]: 
                # print(f' Pass = {Adjacency[i,j]}') # If within each other's pulse duration, continue
                continue
            else: # If not, delete the edge
                Adjacency[i,j] = False 
                Adjacency[j,i] = False
                # print(f' Pass = {Adjacency[i,j]}')
    # Connect the Unconnected Chunks
    n_components, labels = connected_components(Adjacency.cpu().numpy())
    labels = torch.tensor(labels)
    chunks = [torch.where(labels==i)[0] for i in range(n_components)]
    # Delete all edges for chunks of size minchunksize if minchunksize > 1
    if minchunksize > 1:
        for chunk in chunks:
            if len(chunk) <= minchunksize:
                Adjacency[chunk,:] = False
                Adjacency[:,chunk] = False

    chunks = [chunk for chunk in chunks if len(chunk)>=minchunksize]
    for i in range(len(chunks)-1):
        Adjacency[chunks[i][-1],chunks[i+1][0]] = True
        Adjacency[chunks[i+1][0],chunks[i][-1]] = True

    return Adjacency

def GetProcEvent(Pixels,selfloop=None,minchunksize=None,DurationScale=None,ReturnPixelsMask = False):
    '''
    Will return a used pixels index and edges
    pixels are (N) array where N is the number of active pixels
    edges are (M,2) array where M is the number of edges

    Remember that PixelsIndeces are adresses for their original locations
    Edges are from 0 to N-1, their new locations. And are semi-independent of the N pixels
    '''
    Args = {}
    if selfloop      is not None: Args['selfloop']      = selfloop
    if minchunksize  is not None: Args['minchunksize']  = minchunksize
    if DurationScale is not None: Args['DurationScale'] = DurationScale
    # Remember Pixels are [Phi,Theta,Time,PulseDuration]
    Pixels = Pixels[Pixels[:,2].argsort()]
    Adjacency = IsNeighboursDurationTime(Pixels,**Args)
    
    # Get used pixels
    UsedPixels = torch.nonzero(Adjacency.sum(dim=1)).flatten()
    # Get Edges 
    Edges = torch.nonzero(Adjacency[:,UsedPixels][UsedPixels,:])
    if ReturnPixelsMask:
        PixelsMask = Adjacency.sum(dim=1).bool()
        return PixelsMask,Edges
    else:
       return UsedPixels,Edges


