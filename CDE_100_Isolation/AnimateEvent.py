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



ADST_File = '/home/fedor-tairli/work/CDEs/CDE_100_Isolation/ADST.PCGF.400000000.root'

filelist = glob.glob('/home/fedor-tairli/work/Data/MC/low/b01/PCGF/*')

Nevents_Skipped = 0
for file in filelist:
    geometry = GetDetectorGeometry(ADST_File)
    for Event in RecEventProvider(file,0):

        # 27 , 37 , 40
        # Event = Rec_events[37]
        # print(f'Event Id : {Event.GetEventId()}')

        FDEvents = Event.GetFDEvents()


        Plot_Ani = True
        Plot_COM_Scatter = False
        Plot_Total_Signal = True




        for eye in FDEvents:
            if eye.GetEyeId() != 6:
                continue

            # Check if the eye has high cherenkov fraction
            slected_mirror = None
            for mirror in range(10):
                if eye.MirrorIsInEvent(mirror):
                    if eye.GetTelescopeData(mirror).GetGenApertureLight().GetCherenkovFraction() > 95:
                        cherenkov_fraction = eye.GetTelescopeData(mirror).GetGenApertureLight().GetCherenkovFraction()
                        slected_mirrorNormally = mirror
                        break
            else:
                Nevents_Skipped += 1
                continue

            

            eye_geometry = geometry.GetEye(eye.GetEyeId())
            
            print('Event ID',Event.GetEventId() ,'EyeId:', eye.GetEyeId(), 'N_Skipped:', Nevents_Skipped)
            rec_pixel = eye.GetFdRecPixel()

            total_pixels_in_eye = rec_pixel.GetNumberOfPixels()
            total_pilsed_pixels = rec_pixel.GetNumberOfSDPFitPixels()

            # print('Total pixels in eye:', total_pixels_in_eye)
            # print('Total pulsed pixels:', total_pilsed_pixels)

            all_pixel_traces = np.zeros([total_pixels_in_eye,1000]) 
            all_pixel_locs   = np.zeros([total_pixels_in_eye,2])
            all_pixel_status = np.zeros([total_pixels_in_eye])
            all_trace_stds   = np.zeros([total_pixels_in_eye])
            
            min_trigger_time = float('inf')
            max_trigger_time = float('-inf')

            
            for i in range(total_pixels_in_eye):
                pix_status = rec_pixel.GetStatus(i)
                pix_ID     = rec_pixel.GetID(i)
                tel_ID     = rec_pixel.GetTelescopeId(i)

                tel_geometry = eye_geometry.GetTelescope(tel_ID)
                tel_time_offset = eye.GetMirrorTimeOffset(tel_ID)
                tel_bin_offset = int(tel_time_offset // 100) if tel_ID <7 else int(tel_time_offset // 50)

                pix_phi   = tel_geometry.GetPixelPhi  (pix_ID-440*(tel_ID-1),"upward")
                pix_theta = tel_geometry.GetPixelOmega(pix_ID-440*(tel_ID-1),"upward")

                pix_trace = np.array(rec_pixel.GetTrace(i))
                shifted_trace = np.zeros_like(pix_trace)
                if tel_bin_offset > 0:
                    shifted_trace[tel_bin_offset:] = pix_trace[:len(pix_trace)-tel_bin_offset]
                else:
                    shifted_trace = pix_trace  # No shift for zero, negative cannot exist


                if len(shifted_trace) == 2000: # need to sum up two bins
                    shifted_trace = shifted_trace[::2] + shifted_trace[1::2]

                all_pixel_traces[i,:] += shifted_trace
                all_pixel_locs  [i,0]  = pix_theta
                all_pixel_locs  [i,1]  = pix_phi
                all_pixel_status[i]    = pix_status
                all_trace_stds[i] = np.std(pix_trace[:200])

                if pix_status == 4:
                    pix_pulse_start = rec_pixel.GetPulseStart(i)+ tel_bin_offset
                    pix_pulse_stop  = rec_pixel.GetPulseStop(i) + tel_bin_offset
                    if tel_ID >6: 
                        pix_pulse_start //= 2
                        pix_pulse_stop  //= 2
                    # print(f' Pixel {i}: Status={pix_status}, Start={pix_pulse_start}, Stop={pix_pulse_stop}, TelID={tel_ID}, PixID={pix_ID}, Theta={pix_theta:.3f}, Phi={pix_phi:.3f}')
                    if pix_pulse_start < min_trigger_time: min_trigger_time = pix_pulse_start
                    if pix_pulse_stop > max_trigger_time:  max_trigger_time = pix_pulse_stop

            # Expand Trigger window by 3 bins on each side
            # Make sure we dont go below 0 or above 1000
            min_trigger_time = max(0, min_trigger_time - 3)
            max_trigger_time = min(999, max_trigger_time + 3)
            # print(f'Trigger Window: {min_trigger_time} to {max_trigger_time}')

            # For each timebin, we want to find the location of the COM of the signal
            all_signal_COMs = np.zeros((1000,2))
            for t in range(int(min_trigger_time), int(max_trigger_time)+1):
                signals = all_pixel_traces[:, t]
                total_signal = np.sum(signals)
                if total_signal > 0:
                    com_x = np.sum(all_pixel_locs[:,1] * signals) / total_signal
                    com_y = np.sum(all_pixel_locs[:,0] * signals) / total_signal
                    all_signal_COMs[t] = [com_x, com_y]
                else:
                    all_signal_COMs[t] = [np.nan, np.nan]


            all_pixel_traces = np.log10(np.clip(all_pixel_traces, a_min=1, a_max=None))

            if Plot_Ani:
                fig, ax = plt.subplots(figsize=(10, 10))
                sc = ax.scatter(all_pixel_locs[:, 1], all_pixel_locs[:, 0],
                                cmap = 'inferno', c=all_pixel_traces[:, int(min_trigger_time)],
                                s=500,  alpha=1,
                                vmin = 0, vmax = np.log10(20000))
                COM = ax.scatter([], [], s=150,label = 'Signal COM', c='red')
                
                ax.set_xlabel('Phi')
                ax.set_ylabel('Theta')
                ax.set_title(f'Pixel Signal Animation for event {Event.GetEventId()}')
                ax.set_facecolor('midnightblue')
                
                # ax.set_xlim(90,120)
                # ax.set_ylim(15,45)
                ax.invert_xaxis()
                    
                bin_text = ax.text(0.15, 0.02, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=16, color='blue')

                def update(frame):
                    signals = all_pixel_traces[:, frame]
                    sc.set_array(signals)
                    bin_text.set_text(f'Bin: {frame}')
                    COM.set_offsets([all_signal_COMs[frame, 0], all_signal_COMs[frame, 1]])
                    return sc,bin_text, COM

                ani = FuncAnimation(
                    fig, update, frames=range(int(min_trigger_time), int(max_trigger_time)+1),
                    interval=500, blit=True
                )

                plt.colorbar(sc, ax=ax, label='Signal')
                plt.show()

            
            
            # Scatterplot of COMs
            valid_indices = ~np.isnan(all_signal_COMs[:, 0])
            com_x = all_signal_COMs[valid_indices, 0]
            com_y = all_signal_COMs[valid_indices, 1]
            timesteps = np.arange(1000)[valid_indices]
            total_signals = np.sum(all_pixel_traces[:,valid_indices], axis=0)



            x_lims = (np.min(all_pixel_locs[:, 1]), np.max(all_pixel_locs[:, 1]))
            y_lims = (np.min(all_pixel_locs[:, 0]), np.max(all_pixel_locs[:, 0]))
            valid_com_mask = (
                (com_x >= x_lims[0]) & (com_x <= x_lims[1]) &
                (com_y >= y_lims[0]) & (com_y <= y_lims[1])
            )
            com_x = com_x[valid_com_mask]
            com_y = com_y[valid_com_mask]
            timesteps = timesteps[valid_com_mask]
            total_signals = total_signals[valid_com_mask]

            plt.figure(figsize=(8, 8))
            plt.xlim(x_lims)
            plt.ylim(y_lims)
            plt.scatter(all_pixel_locs[:, 1], all_pixel_locs[:, 0], c='gray', s=3, alpha=0.5, label='Pixels')
            scatter = plt.scatter(com_x, com_y, c=timesteps, s=total_signals, cmap='plasma', alpha=0.7,vmin=np.min(timesteps), vmax=np.max(timesteps), label='COM Path')
            if Plot_COM_Scatter:


                
                plt.gca().invert_xaxis()
                plt.xlabel('Phi (COM)')
                plt.ylabel('Theta (COM)')
                plt.title('COM Scatterplot: Color=Time, Size=Total Signal')
                plt.colorbar(scatter, label='Timestep')
                plt.show()

            if Plot_Total_Signal:
                # Figure of the total signal vs time
                plt.figure(figsize=(8, 8))
                plt.plot(timesteps, total_signals, marker='o', linestyle='-', color='purple')
                plt.xlabel('Timestamp')
                plt.ylabel('Total Signal')
                plt.title('Total Signal vs Timestamp')
                plt.grid(True)
                plt.show()