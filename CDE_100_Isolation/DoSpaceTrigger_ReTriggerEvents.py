import sys, os
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
import glob
import pickle


# Redo the trigger using spatial analysis instead of time box analysis

def threshold_relaxation_coefficient(Guaranteed_Threshold,Neighbour_signals,coefficient_decay = 5.0):
    if type(Neighbour_signals) != np.ndarray: Neighbour_signals = np.array(Neighbour_signals)
    N_Neighbours = len(Neighbour_signals)
    # fractional_neighbour_signals = Neighbour_signals/Guaranteed_Threshold


    coefficient = np.exp(-coefficient_decay*np.sum(Neighbour_signals)/N_Neighbours)
    return max(0,coefficient)


def ReTriggerEvent(Event):
    # Extract the relevant data from the event

    N_pixels = Event['TotalPixels']

    PixelData = Event['PixelData']
    pix_Theta = PixelData['Theta']
    pix_Phi = PixelData['Phi']
    pix_Trace = PixelData['Trace']
    pix_RecTrigger = PixelData['RecTrigger']

    pix_MyyTrigger = np.zeros_like(pix_RecTrigger)

    trace_length = pix_Trace.shape[1]

    # First Construct the neighbour map
    neighbours_list = np.zeros((N_pixels,N_pixels), dtype=bool)
    for i_pix in range(N_pixels):
        for j_pix in range(N_pixels):
            if i_pix == j_pix: continue
            pix_distance = np.sqrt( (pix_Theta[i_pix]-pix_Theta[j_pix])**2 + (pix_Phi[i_pix]-pix_Phi[j_pix])**2 )
            if pix_distance < 2.0:  # degrees
                neighbours_list[i_pix,j_pix] = True
    

                
    # Now we can find the approximate trigger window from the RecTrigger
    triggered_positions = np.where(pix_RecTrigger)[1]
    if triggered_positions.size > 0:
        min_pos = np.min(triggered_positions)
        max_pos = np.max(triggered_positions)
    else:
        min_pos = 0
        max_pos = 99999
    min_pos = max(0,min_pos-5)
    max_pos = min(trace_length,max_pos+10)

    
    Norm_Signal_Array  = pix_Trace / np.std(pix_Trace[:,:200].astype(np.float64), axis=1, keepdims=True)
    # print(f'Normed Signal Array : {Norm_Signal_Array}')
    GuaranteedThreshold = 5 # In units of std deviations

    # Now go over each time bin and ReTrigger
    for t in range(min_pos, max_pos):
        signals = Norm_Signal_Array[:,t]
        # print(f"Time bin {t}, Signals: {signals}")
        for i_pix in range(N_pixels):
            # print(f'Checking Singal {signals[i_pix]:.2f} for pixel {i_pix} at time {t}')
            if signals[i_pix] > GuaranteedThreshold:
                pix_MyyTrigger[i_pix,t] = True
                continue

            neighbour_indices = np.where(neighbours_list[i_pix])[0]
            neighbour_signals = signals[neighbour_indices]
            neighbour_signals = np.clip(neighbour_signals, -GuaranteedThreshold, GuaranteedThreshold)

            relative_threshold = GuaranteedThreshold * threshold_relaxation_coefficient(GuaranteedThreshold, neighbour_signals)
            if signals[i_pix] > relative_threshold:
                pix_MyyTrigger[i_pix,t] = True
                
    
    # Now we must prune the trigger 

    # First we make sure that once triggered, a pixel stays triggered for at least 3 time bins
    for i_pix in range(N_pixels):
        triggered_times = np.where(pix_MyyTrigger[i_pix])[0]
        # print(f"Pixel {i_pix} initially triggered at times: {triggered_times}")
        for t in triggered_times:
            trigger_bins = pix_MyyTrigger[i_pix, max(0,t-2):min(trace_length,t+3)]
            if len(trigger_bins) < 5: continue
            elif (np.sum(trigger_bins[0:3]) == 3) or (np.sum(trigger_bins[1:4]) == 3) or (np.sum(trigger_bins[2:5]) == 3): continue
            else: 
                # print(f"Pruning pixel {i_pix} at time {t} for not having 3 consecutive triggers.")
                pix_MyyTrigger[i_pix, t] = False
        # print(f"Pixel {i_pix} finally triggered at times: {np.where(pix_MyyTrigger[i_pix])[0]}")
    # Other Prunning Steps are not added yet

    # Second we make sure that the triggered pixels are all contiguous.
    # To do that, we will use the neighbours map to find pixels connected to the one with the highest signal

    Triggered_at_any_time = np.where(np.any(pix_MyyTrigger, axis=1))[0]
    highest_Signal_pix = np.where(pix_Trace == np.max(pix_Trace))[0]
    
    # using BFS we visit neighbours of triggered pixels
    Is_Safe_map = np.zeros(N_pixels,dtype=bool)

    queue = [highest_Signal_pix[0]]
    while queue:
        current_pix = queue.pop(0)
        if Is_Safe_map[current_pix]: continue
        Is_Safe_map[current_pix] = True
        neighbour_indices = np.where(neighbours_list[current_pix])[0]
        for neighbour in neighbour_indices:
            if neighbour in Triggered_at_any_time and not Is_Safe_map[neighbour]:
                queue.append(neighbour)
    # Now make everything not in Is_Safe_map False in the trigger map
    for i_pix in range(N_pixels):
        if not Is_Safe_map[i_pix]:
            pix_MyyTrigger[i_pix,:] = False

    Event['MyyTrigger'] = pix_MyyTrigger




if __name__ == '__main__':
    data_path = "./Pickled_Data/*"



    all_files = sorted(glob.glob(data_path ))
    print(f'Found {len(all_files)} files')
        
    for file in all_files:
        print(f'Processing file: {file}')
        with open(file, 'rb') as f:
            Data = pickle.load(f)
            
            for i,Event in enumerate(Data):
                print(f"    ReTriggering event {i+1}/{len(Data)}", end='\r')
                ReTriggerEvent(Event)
            print()
            print(f'Finished processing file: {file}, saving results...')
        with open(file, 'wb') as f:
            pickle.dump(Data, f)