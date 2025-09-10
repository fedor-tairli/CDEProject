import sys, os

# Import ADST Reader
adst_advanced_path = '/home/fedor-tairli/work/OfflineInstallation/fortuna/'
sys.path.append(adst_advanced_path)
from gui_utils.adst_advanced import *
from gui_utils.auger_data_handler import AugerDataHandler

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
import glob

def IndexToXY(index):
    index -=1
    Xs = index//22
    Ys = index%22

    return Xs,Ys

ADST_File = '/home/fedor-tairli/work/CDEs/CDE_100_Isolation/ADST.PCGF.400000000.root'
filelist = [ADST_File,]

actual_file_list_path = '/home/fedor-tairli/work/CDEs/ReadingData/FilesToRead.txt'
with open(actual_file_list_path, 'r') as f:
    filelist = f.readlines()
    filelist = ["/home/fedor-tairli/work" + f.strip()[2:] for f in filelist]

use_event = (str(400000000),str(5172))
use_event = None
# filelist = glob.glob('/home/fedor-tairli/work/Data/MC/low/b01/PCGF/*')


do_plot = False
set_std_norm_for_all = False
plot_log_scale = True
skip_long_events = True
debug = True
superdebug = True

# import ROOT

# ROOT.gdebug = 1

for file in filelist:
    geometry = GetDetectorGeometry(file)

    for Event in RecEventProvider(file,0):
        if (not use_event is None):
            if use_event[0] not in str(Event.GetEventId()) or use_event[1] not in str(Event.GetEventId()): continue
        FdEvents = Event.GetFDEvents()
        
        if skip_long_events: # Can use status 4 to skip long events, its a bit faster, but requires Time Fit from rec
            First_Status_4_trigger = 1000
            Last_Status_4_trigger  = -1
        for eye in FdEvents:
            if eye.GetEyeId() == 6: continue
            if debug: print('------------------------------------------------------------------')
            if debug: print(f"Processing Event {Event.GetEventId()}, Eye {eye.GetEyeId()}")

            Bin_Width = 100 if eye.GetEyeId() in [1,2,3,4] else 50
            
            if debug: print( '    Getting Pixels')
            rec_pixel = eye.GetFdRecPixel()
            if debug: print(f'    Got {rec_pixel.GetNumberOfPixels()}')

            N_pixels_in_event = rec_pixel.GetNumberOfPixels()


            Signal_array = []
            ID_array     = []
            Pos_array    = []
            Status_array = []

            Rec_Trigger_array = []
            My_Trigger_array  = []
            
            Eye_Geometry = geometry.GetEye(eye.GetEyeId())

            for iPix in range(N_pixels_in_event):
                try:
                    if debug : 
                        print_str = f"  Processing Pixel {iPix+1}/{N_pixels_in_event}" +\
                                    f" with Pixel Id {rec_pixel.GetPixelId(iPix)}-{rec_pixel.GetID(iPix)}" +\
                                    f" and Tel_ID {rec_pixel.GetTelescopeId(iPix)}"
                        print(print_str, end='\r')
                        
                    if superdebug  or (debug and iPix == N_pixels_in_event-1): print()


                    pix_ID    = rec_pixel.GetID(iPix)
                    if superdebug: print(f"    Pixel unique ID: {pix_ID}")
                    tel_ID = rec_pixel.GetTelescopeId(iPix)
                    if superdebug: print(f"    Telescope ID: {tel_ID}")
                    if not rec_pixel.HasADCTrace(iPix):
                        if superdebug: print(f"    Warning: Pixel {pix_ID} has no ADC trace, skipping.")
                        N_pixels_in_event -= 1
                        continue
                    pix_geom_id = rec_pixel.GetPixelId(iPix) -1 
                    if superdebug: print(f"    Pixel geom id: {pix_geom_id}")
                    pix_status = rec_pixel.GetStatus(iPix)
                    if superdebug: print(f"    Pixel status: {pix_status}")
                    tel_pointing = "upward" if eye.GetEyeId() == 5 or tel_ID in [7,8,9] else "downward"
                    if superdebug: print(f"    Telescope pointing: {tel_pointing}")
                    
                    # pix_Trace = np.array(rec_pixel.GetTrace(iPix))
                    pix_Trace = rec_pixel.GetTrace(iPix)
                    if pix_Trace is None or len(pix_Trace) == 0:
                        if superdebug: print(f"    Warning: Pixel {pix_ID} has no valid trace data, skipping.")
                        N_pixels_in_event -= 1
                        continue
                    else:
                        pix_Trace = np.array(pix_Trace)
                    if superdebug: print(f"    Pixel trace length: {len(pix_Trace)}")
                    tel_Geometry = Eye_Geometry.GetTelescope(tel_ID)
                    if superdebug: print(f"    Got telescope geometry for telescope {tel_ID}")
                    pix_Theta       = tel_Geometry.GetPixelOmega(pix_geom_id,tel_pointing)
                    pix_Phi         = tel_Geometry.GetPixelPhi(pix_geom_id,tel_pointing)
                    if superdebug: print(f"    Pixel Theta: {pix_Theta}, Pixel Phi: {pix_Phi}")
                    pix_Time_Offset = eye.GetMirrorTimeOffset(tel_ID)
                    if superdebug: print(f"    Pixel Time Offset: {pix_Time_Offset}")
                    pix_Bin_Offset  = int(pix_Time_Offset/Bin_Width)
                    if superdebug: print(f"    Pixel Bin Offset: {pix_Bin_Offset}")

                    # cycle the trace to account for the time offset
                    if pix_Bin_Offset > 0:
                        pix_Trace[pix_Bin_Offset:] = pix_Trace[:len(pix_Trace)-pix_Bin_Offset]
                        pix_Trace[:pix_Bin_Offset] = 0
                        if superdebug: print(f"    Applied positive bin offset of {pix_Bin_Offset}")

                    # Check triggers from Rec
                    pix_triggered = np.zeros(len(pix_Trace))
                    pix_Pulse_Start = rec_pixel.GetPulseStart(iPix)
                    if superdebug: print(f"    Pixel Pulse Start: {pix_Pulse_Start}")
                    pix_Pulse_Stop  = rec_pixel.GetPulseStop(iPix)
                    if superdebug: print(f"    Pixel Pulse Stop: {pix_Pulse_Stop}")
                    if skip_long_events and pix_status == 4:
                        if pix_Pulse_Start < First_Status_4_trigger: First_Status_4_trigger = pix_Pulse_Start
                        if pix_Pulse_Stop  > Last_Status_4_trigger:  Last_Status_4_trigger  = pix_Pulse_Stop

                    if pix_Pulse_Start < pix_Pulse_Stop:
                        pix_triggered[pix_Pulse_Start:pix_Pulse_Stop] = 1
                    if superdebug: print("Appending pixel data to arrays.")
                    Signal_array.append(pix_Trace)
                    if superdebug: print("Successfully appended signal array.")
                    ID_array.append(pix_ID)
                    if superdebug: print("Successfully appended ID array.")
                    Pos_array.append( (pix_Phi,pix_Theta) )
                    if superdebug: print("Successfully appended Position array.")
                    Status_array.append(pix_status)
                    if superdebug: print("Successfully appended Status array.")

                    Rec_Trigger_array.append(pix_triggered)
                    if superdebug: print("Successfully appended Rec Trigger array.")

                    if superdebug: print("Moving to next pixel.")
                except Exception as e:
                    if "ReferenceError: attempt to access a null-pointer" in str(e):
                        print(f"    Warning: Skipping pixel {iPix} due to null pointer error.")
                    else:
                        print(f"    Error processing pixel {iPix}: {e}")
                    continue
            
            if skip_long_events:
                if Last_Status_4_trigger - First_Status_4_trigger > 20:
                    if debug: print(f"Skipping event {Event.GetEventId()}, eye {eye.GetEyeId()} with long trigger streak of {Last_Status_4_trigger - First_Status_4_trigger} bins.")
                    continue
            if superdebug:
                print(f' Processing pixel info into arrays:')
                print(f'  Total pixels in event: {N_pixels_in_event}')
                print(f'  Signal List Length: {len(Signal_array)}, First Signal Length: {len(Signal_array[0]) if len(Signal_array)>0 else "N/A"}')
                print(f'  ID List Length: {len(ID_array)}')
                print(f'  Pos List Length: {len(Pos_array)}, First Pos: {Pos_array[0] if len(Pos_array)>0 else "N/A"}')
                print(f'  Status List Length: {len(Status_array)}, First Status: {Status_array[0] if len(Status_array)>0 else "N/A"}')
                print(f'  Rec Trigger List Length: {len(Rec_Trigger_array)}, First Rec Trigger Length: {len(Rec_Trigger_array[0]) if len(Rec_Trigger_array)>0 else "N/A"}')
                print(f'  My Trigger List Length: {len(My_Trigger_array)}')
                print('  Sample of pixel details:')
                for i in range(N_pixels_in_event):
                    print(f'  Pixel {i}: ID={ID_array[i]}, Pos={Pos_array[i]}, Status={Status_array[i]}, Signal length={len(Signal_array[i])}, Rec Trigger sum={np.sum(Rec_Trigger_array[i])}')
            Signal_array = np.array(Signal_array)
            ID_array     = np.array(ID_array)
            Pos_array    = np.array(Pos_array)
            Status_array = np.array(Status_array)

            Rec_Trigger_array = np.array(Rec_Trigger_array)
            My_Trigger_array = np.zeros_like(Rec_Trigger_array)


            # Skip this event if too long
            # This method should work, but we can use status to 
            # skip_this_event = False
            # if skip_long_events:
            #     Rec_Trigger_Sum = np.sum(Rec_Trigger_array, axis=0)
                
            #     triggered_positions = np.where(Rec_Trigger_Sum > 0)[0]
            #     earliest_cont_trigger = triggered_positions[0]
            #     current_streak = 1
            #     for triggered_Pos in triggered_positions[1:]:
            #         if triggered_Pos == earliest_cont_trigger + current_streak:
            #             current_streak += 1
            #             if current_streak > 20:
            #                 print(f"Skipping event {Event.GetEventId()} with long trigger streak of {current_streak} bins.")
            #                 continue
            #         else:
            #             earliest_cont_trigger = triggered_Pos
            #             current_streak = 1
            # if skip_this_event: continue

            

            # Window of Tiggeing by rec
            mask = Status_array == 4
            triggered_positions = np.where((Rec_Trigger_array[mask] == 1))
            if triggered_positions[1].size > 0:
                min_pos = triggered_positions[1].min()
                max_pos = triggered_positions[1].max()
                print(f"Min triggered position (dim 2): {min_pos}")
                print(f"Max triggered position (dim 2): {max_pos}")
            else:
                print("No triggered positions found with status==4.")
                continue
            
            # Adjust min and max to be 3 bins wider
            min_pos = max(0, min_pos - 3)
            max_pos = min(Signal_array.shape[1] - 1, max_pos + 3)
            #######################################################################
            
            # Construct my own trigger here
            
            # Make a neighbourhood graph
            neighbours_list = np.zeros([N_pixels_in_event,N_pixels_in_event])

            for pix_i in range(N_pixels_in_event):
                for pix_j in range(N_pixels_in_event):
                    if pix_i == pix_j: continue
                    pix_distance = (Pos_array[pix_i,0] - Pos_array[pix_j,0])**2 + (Pos_array[pix_i,1] - Pos_array[pix_j,1])**2
                    pix_distance = np.sqrt(pix_distance)
                    if pix_distance < 2:
                        neighbours_list[pix_i,pix_j] = 1


            # Make a plot, to see if everything works correctly
            signals = Signal_array[:,min_pos+11]
            # plt.figure(figsize=(10,10))
            # for i_pix in range(N_pixels_in_event):
            #     for j_pix in range(N_pixels_in_event):
            #         if neighbours_list[i_pix,j_pix] == 1:
            #             plt.plot([Pos_array[i_pix,0],Pos_array[j_pix,0]],[Pos_array[i_pix,1],Pos_array[j_pix,1]],c='gray',alpha=0.3)
            # plt.scatter(Pos_array[:,0],Pos_array[:,1], c = signals, cmap='inferno', s=300, vmin=0, vmax=500)
            # plt.show()

            # for i_pix in range(N_pixels_in_event):
            #     num_neighbours = int(np.sum(neighbours_list[i_pix]))
            #     print(f"Pixel {ID_array[i_pix]} has {num_neighbours} neighbours.")
            # exit
            # Now make the trigger
            My_Trigger_array = np.zeros_like(Rec_Trigger_array)
            Norm_Signal_array = Signal_array / np.std(Signal_array[:,:200])
            # plt.figure()
            # plt.hist(Norm_Signal_array.flatten(),bins = 100,range = (-10,10))
            # plt.yscale('log')
            # plt.show()
            # exit()
            Guaranteed_threshold = 5
            # relative_threshold = 0.5

            def threshold_relaxation_coefficient(Guaranteed_Threshold,Neighbour_signals):
                if type(Neighbour_signals) != np.ndarray: Neighbour_signals = np.array(Neighbour_signals)
                N_Neighbours = len(Neighbour_signals)
                # fractional_neighbour_signals = Neighbour_signals/Guaranteed_Threshold


                coefficient = np.exp(-2.5*np.sum(Neighbour_signals)/N_Neighbours)
                return max(0,coefficient)
            
            for t in range(min_pos, max_pos+1):
                signals = Norm_Signal_array[:,t]

                for i_pix in range(N_pixels_in_event):
                    if signals[i_pix] > Guaranteed_threshold: 
                        # print('Passed Guaranteed Threshold')
                        My_Trigger_array[i_pix,t] = 1
                        continue
                    
                    neighbour_indices = np.where(neighbours_list[i_pix] == 1)[0]
                    neighbour_signals = signals[neighbour_indices]
                    neighbour_signals = np.clip(neighbour_signals,a_min = -Guaranteed_threshold, a_max = Guaranteed_threshold).tolist()


                    relative_threshold = Guaranteed_threshold * threshold_relaxation_coefficient(Guaranteed_threshold,neighbour_signals)
                    if signals[i_pix] > relative_threshold:
                        # print(f'Passed relative Threshold: {relative_threshold} , with Neighbour sum: {total_neighbour_signal}')
                        My_Trigger_array[i_pix,t] = 1
                    
            # Trigger Pruning
            # Make sure that once triggered, stays triggered for 3 bins
            for i_pix in range(N_pixels_in_event):
                triggered_times = np.where(My_Trigger_array[i_pix] == 1)[0]
                for t in triggered_times:
                    trigger_bins = My_Trigger_array[i_pix,t-2:t+3]
                    if len(trigger_bins) < 5: continue
                    elif (np.sum(trigger_bins[0:3]) == 3) or (np.sum(trigger_bins[1:4]) == 3) or (np.sum(trigger_bins[2:5]) == 3): continue
                    else: My_Trigger_array[i_pix,t] = 0

            ######################################################################
            # Make the Plot
            if not do_plot: continue
            fig, ax = plt.subplots(figsize=(10, 10))
            if set_std_norm_for_all:
                select_vmax = 5
                select_vmin = 0
            elif plot_log_scale:
                select_vmax = np.log10(18000)
                select_vmin = 0
            else:
                select_vmax = 500
                select_vmin = 0
        
            scat = ax.scatter(Pos_array[:,0], Pos_array[:,1],
                              cmap='inferno',c = np.log10(np.clip(Signal_array[:,min_pos],a_min=1,a_max=None)),
                               s=350, label='AllPixels', alpha=1,
                               vmin = select_vmin, vmax = select_vmax)
            
            # Plot red cross at triggered pixels at this time bin
            rec_trigger_alpha = (Rec_Trigger_array[:,min_pos]*0.8).astype(float)
            print(len(rec_trigger_alpha), N_pixels_in_event)
            rec_triggered = ax.scatter(Pos_array[:,0], Pos_array[:,1], alpha = rec_trigger_alpha,
                                       s=50, label='Rec Triggered', facecolors='none', c=['cyan']*N_pixels_in_event, linewidths=2,marker='x')
            
            my_trigger_alpha = (My_Trigger_array[:,min_pos]).astype(float)
            my_triggered = ax.scatter(Pos_array[:,0], Pos_array[:,1], alpha = my_trigger_alpha,
                                       s=30, label='My Triggered', c = ['lime']*N_pixels_in_event, linewidths=2, marker='o')

            # Bin Label Text
            bin_text = ax.text(0.05, 0.95, f'Time Bin: {min_pos}', transform=ax.transAxes, fontsize=14,
                               verticalalignment='top', color='white', bbox=dict(facecolor='black', alpha=0.5))

            ax.set_xlabel('Phi')
            ax.set_ylabel('Theta')
            ax.set_title(f'Pixel Signals for event {Event.GetEventId()} at time bin {min_pos}')
            ax.set_facecolor('midnightblue')
            plt.colorbar(scat, label='Pixel Signal')
            plt.legend()
            
            # Set Limits here if needed
            # ax.set_xlim([-1.5,1.5])
            # ax.set_ylim([-1.5,1.5])
            ax.invert_xaxis()

            # Construct the animation updates

            def update(frame):
                if set_std_norm_for_all or not plot_log_scale : signals = Signal_array[:, frame]
                else : signals = signals = np.log10(np.clip(Signal_array[:,frame], a_min=1, a_max=None))
                
                scat.set_array(signals)
                bin_text.set_text(f'Time Bin: {frame}')

                rec_trigger_alpha = (Rec_Trigger_array[:,frame]*0.8).astype(float)
                rec_triggered.set_alpha(rec_trigger_alpha)
                my_trigger_alpha = (My_Trigger_array[:,frame]).astype(float)
                my_triggered.set_alpha(my_trigger_alpha)

                return scat, bin_text, rec_triggered
            
            ani = FuncAnimation(
                fig, update, frames=range(min_pos, max_pos + 1),
                interval=500, blit=False, repeat=True
            )
            ani.save('trigger_animation.gif', writer='pillow', fps=2)
            plt.show()
