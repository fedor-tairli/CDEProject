import sys, os

# Import ADST Reader
adst_advanced_path = '/home/fedor-tairli/work/OfflineInstallation/fortuna/'
sys.path.append(adst_advanced_path)
from gui_utils.adst_advanced import *
from gui_utils.auger_data_handler import AugerDataHandler

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def IndexToXY(index):
    index -=1
    Xs = index//22
    Ys = index%22

    return Xs,Ys


ADST_File = '/home/fedor-tairli/work/CDEs/CDE_100_Isolation/ADST.PCGF.400000000.root'


Rec_events = RecEventProvider(ADST_File,0)

Event = Rec_events[27]
print(f'Event Id : {Event.GetEventId()}')

FDEvents = Event.GetFDEvents()
geometry = GetDetectorGeometry(ADST_File)







for eye in FDEvents:
    if eye.GetEyeId() != 6:
        continue
    
    eye_geometry = geometry.GetEye(eye.GetEyeId())
    
    print('EyeId:', eye.GetEyeId())
    rec_pixel = eye.GetFdRecPixel()

    total_pixels_in_eye = rec_pixel.GetNumberOfPixels()
    total_pilsed_pixels = rec_pixel.GetNumberOfSDPFitPixels()

    print('Total pixels in eye:', total_pixels_in_eye)
    print('Total pulsed pixels:', total_pilsed_pixels)

    all_pixel_traces = np.zeros([total_pixels_in_eye,1000]) 
    all_pixel_locs   = np.zeros([total_pixels_in_eye,2])
    all_pixel_status = np.zeros([total_pixels_in_eye])
    all_trace_stds = np.zeros([total_pixels_in_eye])
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
        all_trace_stds[i] = np.std(pix_trace[:200])
    # Create the Triggering Mechanism

    # 1. Find the highest signal in the traces
    # 2. Keep Adding the next highest bin
    # 3. something, idk what yet

    Singal_Integral = [0,]
    locations = []

    if True:
        all_pixel_traces_full = all_pixel_traces.copy()
        # Lets see what happens if we devide each signal by its STD
        all_pixel_traces = all_pixel_traces / all_trace_stds[:,np.newaxis]
        

    current_signal = 1000000000
    N_iter = 0
    print(f'Total Possible Iterations: {total_pixels_in_eye*1000}')
    while current_signal > 0:
        # Find the next highest signal in the traces
        current_signal_loc = np.unravel_index(np.argmax(all_pixel_traces, axis=None), all_pixel_traces.shape)
        current_signal_dir = all_pixel_locs[current_signal_loc[0],:]
        current_signal_tim = current_signal_loc[1]
    
        locations.append(current_signal_dir+current_signal_tim)

        current_signal = all_pixel_traces[current_signal_loc]
        Singal_Integral.append(Singal_Integral[-1]+current_signal)

        all_pixel_traces[current_signal_loc] = 0 # Set to zero so we dont pick it again
        N_iter += 1
        if N_iter%10 == 0:
            print(f'    Iteration {N_iter}, Current Signal: {current_signal}, Integral: {Singal_Integral[-1]}',end='\r')
        
        if N_iter > 1000: # Hacking for quickness
            break
    

    

    print()
    Singal_Integral = Singal_Integral[1:]
    locations = np.array(locations)
    print(locations.shape)
    print()

    plt.figure(figsize=(10,6))
    plt.plot(Singal_Integral,label='Signal Integral')
    ax = plt.gca()

    # plot a linear fit for the last 500 points
    def linear_func(x, a, b):
        return a*x + b
    x_data = np.arange(len(Singal_Integral)-500,len(Singal_Integral),1)
    y_data = Singal_Integral[-500:]
    popt, pcov = curve_fit(linear_func, x_data, y_data)

    ax.plot(np.arange(len(Singal_Integral)), linear_func(np.arange(len(Singal_Integral)), *popt), 'r--', label='Linear Fit (last 500 points)')
    print(f'Linear Fit Parameters: a={popt[0]}, b={popt[1]}')  


    ax.set_xlabel('Number of bins added')
    ax.set_ylabel('Signal Integral')
    ax.legend(loc='upper left')




    ax2 = ax.twinx()
    ax2.plot(np.diff(Singal_Integral,prepend=0),color='orange',label='Signal Derivative')
    ax2.set_ylabel('Signal Derivative')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')

    plt.title(f'Event {Event.GetEventId()} Eye {eye.GetEyeId()}')
    plt.legend()

    plt.show()

    # plt.figure(figsize=(10,6))
    # plt.hist(all_trace_stds,bins=50)
    # plt.show()


    
    