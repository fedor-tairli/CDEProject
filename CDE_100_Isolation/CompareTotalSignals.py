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



ADST_File = '/home/fedor-tairli/work/CDEs/CDE_100_Isolation/ADST.PCGF.400000000.root'


geometry   = GetDetectorGeometry(ADST_File)
# Rec_events = RecEventProvider(ADST_File,0)


for Event in RecEventProvider(ADST_File,0):
    print(f'Event Id : {Event.GetEventId()}')

    FDEvents = Event.GetFDEvents()

    for eye in FDEvents:
        if eye.GetEyeId() == 6:
            continue
        
        eye_geometry = geometry.GetEye(eye.GetEyeId())
        rec_pixel = eye.GetFdRecPixel()

        total_pixels_in_eye = rec_pixel.GetNumberOfPixels()
        total_pilsed_pixels = rec_pixel.GetNumberOfSDPFitPixels()

        
        all_pixel_traces = np.zeros([total_pixels_in_eye,1000]) 
        all_pixel_locs   = np.zeros([total_pixels_in_eye,2])
        all_pixel_status = np.zeros([total_pixels_in_eye])
        
        min_trigger_time = float('inf')
        max_trigger_time = float('-inf')

        # Initialise a large figure
        for i in range(total_pixels_in_eye):
            pix_status = rec_pixel.GetStatus(i)
            pix_ID     = rec_pixel.GetID(i)
            tel_ID     = rec_pixel.GetTelescopeId(i)

            tel_geometry = eye_geometry.GetTelescope(tel_ID)
            pix_phi   = tel_geometry.GetPixelPhi  (pix_ID-440*(tel_ID-1),"upward")
            pix_theta = tel_geometry.GetPixelOmega(pix_ID-440*(tel_ID-1),"upward")

            pix_trace = np.array(rec_pixel.GetTrace(i))
            
            if len(pix_trace) == 2000: # need to sum up two bins
                pix_trace = pix_trace[::2] + pix_trace[1::2]

            all_pixel_traces[i,:] += pix_trace
            all_pixel_locs  [i,0]  = pix_theta
            all_pixel_locs  [i,1]  = pix_phi
            all_pixel_status[i]    = pix_status
            
            if pix_status == 4:
                pix_pulse_start = rec_pixel.GetPulseStart(i)
                pix_pulse_stop  = rec_pixel.GetPulseStop(i) 
                if pix_pulse_start < min_trigger_time: min_trigger_time = pix_pulse_start
                if pix_pulse_stop > max_trigger_time:  max_trigger_time = pix_pulse_stop

        min_trigger_time = int(min_trigger_time - 10)
        max_trigger_time = int(max_trigger_time + 10)

        # For each time bin between min/max trigger time, plot the total signal in all pixels, and total signal in pixels with status 4
        Total_Signal = []
        Triggered_Signal = []

        for t in range(min_trigger_time,max_trigger_time):
            Total_Signal.append(np.sum(all_pixel_traces[:,t]))
            Triggered_Signal.append(np.sum(all_pixel_traces[np.where(all_pixel_status==4)[0],t]))
        Total_Signal = np.array(Total_Signal)
        Triggered_Signal = np.array(Triggered_Signal)

        timesteps = np.arange(min_trigger_time, max_trigger_time)
        plt.figure(figsize=(12, 10))
        plt.plot(timesteps, Total_Signal, label='Total Signal in all Pixels')
        plt.plot(timesteps, Triggered_Signal, label='Total Signal in Triggered Pixels')
        plt.axvline(x=min_trigger_time+10, color='k', linestyle='--', label='Start Trigger Window')
        plt.axvline(x=max_trigger_time-10, color='k', linestyle='--', label='End Trigger Window')
        plt.xlabel('Time Bins (100 ns each)')
        plt.ylabel('Total Signal in Bin')
        plt.title(f'Event {Event.GetEventId()}, Eye {eye.GetEyeId()}')
        plt.legend()
        plt.grid()
        plt.show()