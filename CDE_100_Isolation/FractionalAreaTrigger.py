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
use_event = (str(400000000),str(5172))
# filelist = glob.glob('/home/fedor-tairli/work/Data/MC/low/b01/PCGF/*')


time_trig = True
space_trig = True
set_std_norm_for_all = True
plot_log_scale = True

for file in filelist:
    geometry = GetDetectorGeometry(file)

    for Event in RecEventProvider(file,0):
        if use_event[0] not in str(Event.GetEventId()) or use_event[1] not in str(Event.GetEventId()): continue
        FdEvents = Event.GetFDEvents()

        for eye in FdEvents:
            if eye.GetEyeId() == 6: continue
            Bin_Width = 100 if eye.GetEyeId() in [1,2,3,4] else 50
            
            rec_pixel = eye.GetFdRecPixel()

            N_pixels_in_event = rec_pixel.GetNumberOfPixels()


            Signal_array = []
            ID_array     = []
            Pos_array    = []
            Status_array = []

            Rec_Trigger_array = []
            My_Trigger_array  = []
            
            Eye_Geometry = geometry.GetEye(eye.GetEyeId())

            for iPix in range(N_pixels_in_event):

                pix_status = rec_pixel.GetStatus(iPix)

                tel_ID = rec_pixel.GetTelescopeId(iPix)
                tel_pointing = "upward" if eye.GetEyeId() == 5 or tel_ID in [7,8,9] else "downward"
                
                pix_Trace = rec_pixel.GetTrace(iPix)
                pix_ID    = rec_pixel.GetPixelId(iPix)
                

                tel_Geometry = Eye_Geometry.GetTelescope(tel_ID)
                pix_Theta       = tel_Geometry.GetPixelOmega(pix_ID-440*(tel_ID-1),tel_pointing)
                pix_Phi         = tel_Geometry.GetPixelPhi(pix_ID-440*(tel_ID-1),tel_pointing)
                pix_Time_Offset = eye.GetMirrorTimeOffset(tel_ID)
                pix_Bin_Offset  = int(pix_Time_Offset/Bin_Width)

                # cycle the trace to account for the time offset
                if pix_Bin_Offset > 0:
                    pix_Trace[pix_Bin_Offset:] = pix_Trace[:len(pix_Trace)-pix_Bin_Offset]
                    pix_Trace[:pix_Bin_Offset] = 0

                # Check triggers from Rec
                pix_triggered = np.zeros(len(pix_Trace))
                pix_Pulse_Start = rec_pixel.GetPulseStart(iPix)
                pix_Pulse_Stop  = rec_pixel.GetPulseStop(iPix)

                if pix_Pulse_Start < pix_Pulse_Stop:
                    pix_triggered[pix_Pulse_Start:pix_Pulse_Stop] = 1
                
                Signal_array.append(pix_Trace)
                ID_array.append(pix_ID)
                Pos_array.append( (pix_Phi,pix_Theta) )
                Status_array.append(pix_status)

                Rec_Trigger_array.append(pix_triggered)
            
            Signal_array = np.array(Signal_array)
            ID_array     = np.array(ID_array)
            Pos_array    = np.array(Pos_array)
            Status_array = np.array(Status_array)

            Rec_Trigger_array = np.array(Rec_Trigger_array)
            My_Trigger_array = np.zeros_like(Rec_Trigger_array)
                    
            
            # print(f'Event ID: {Event.GetEventId()}, Eye ID: {eye.GetEyeId()}, N_pixels: {N_pixels_in_event}')
            

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
                break
            
            # Adjust min and max to be 3 bins wider
            min_pos = max(0, min_pos - 3)
            max_pos = min(Signal_array.shape[1] - 1, max_pos + 3)
            #######################################################################
            
            # Construct my own trigger here
            PixelSTDs = np.std(Signal_array[:,:200],axis=1)
            Norm_Signal_array = Signal_array / PixelSTDs[:,np.newaxis]

            Threshold = 1
            my_rec_trigger = Norm_Signal_array > Threshold
            My_Trigger_array = my_rec_trigger.astype(int)

            # now check that each pixel is triggered for at least 3 consecutive bins

            # Can hack becasue i already know where the signal is
            My_Trigger_array[:, :min_pos-3] = 0
            My_Trigger_array[:, max_pos +3:] = 0

            if time_trig:
                # Checks that at least 3 consecutive bins are above threshold
                for i in range(N_pixels_in_event):
                    for j in range(max(min_pos-3,0), min(max_pos+4,Signal_array.shape[1])):
                        if My_Trigger_array[i,j] == 1:
                            active_streak = 1
                            for k in [j-2,j-1,j+1,j+2]:
                                if My_Trigger_array[i,k] == 1:
                                    active_streak += 1
                            if active_streak < 3:
                                My_Trigger_array[i,j] = 0

            # Space Trigger to be done here
            if space_trig:
                for i in range(N_pixels_in_event):
                    # find neighbors - can use approx linear phi-theta projection
                    pix_distances = (Pos_array[:,0] - Pos_array [i,0])**2+ (Pos_array[:,1] - Pos_array [i,1])
                    pix_distances = np.sqrt(pix_distances)
                    pix_neighnours = (pix_distances < 2) & (pix_distances > 1e-5) # Neighbour Pixels should be < 2 degrees, and obviously not the same pix

                    # Find pixels indexes of the neighbours
                    neighbour_indexes = np.where(pix_neighnours)[0]

                    for j in range(max(min_pos-3,3), min(max_pos+4,Signal_array.shape[1]-3)):
                        if My_Trigger_array[i,j] == 1:
                            n_neigh_trig = np.sum(My_Trigger_array[np.ix_(neighbour_indexes, [j-3, j-2, j-1, j, j+1, j+2, j+3])])
                            if n_neigh_trig < 2:
                                My_Trigger_array[i,j] = 0
                            
            if set_std_norm_for_all:
                Signal_array = Norm_Signal_array

                    


            ######################################################################
            # Make the Plot
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
                else : signals = signals = np.log10(np.clip(Signal_array, a_min=1, a_max=None))
                
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
