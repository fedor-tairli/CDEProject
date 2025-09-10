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

pixel_average = 0
pixel_average_std = 24.5
pixel_average_std_std = 5.5


plt.figure(figsize=(10,6))

sample_N = 100
sample_len = 1000

# each pixel will have a mean of 0
each_pixel_mean = np.zeros(sample_N)
# need to sample a gausian for each pixel's std
each_pixel_std = np.random.normal(pixel_average_std,pixel_average_std_std,sample_N)

# now we sample each pixel's values 
all_pixel_traces = np.zeros([sample_N,sample_len])
for i in range(sample_N):
    all_pixel_traces[i,:] = np.random.normal(each_pixel_mean[i],each_pixel_std[i],sample_len)


Singal_Integral = [0,]


current_signal = 1000000000
N_iter = 0
# while current_signal > 0:
for i in range(sample_N*sample_len):
    # Find the next highest signal in the traces
    current_signal_loc = np.unravel_index(np.argmax(all_pixel_traces, axis=None), all_pixel_traces.shape)
    

    current_signal = all_pixel_traces[current_signal_loc]
    Singal_Integral.append(Singal_Integral[-1]+current_signal)

    all_pixel_traces[current_signal_loc] = -999999990 # Set to zero so we dont pick it again
    N_iter += 1
    if N_iter%10 == 0:
        print(f'    Iteration {N_iter}, Current Signal: {current_signal}, Integral: {Singal_Integral[-1]}',end='\r')
    


plt.plot(Singal_Integral,label='Signal Integral')
ax = plt.gca()

# plot a linear fit for the last 500 points
def linear(x, a, b):
    return a*x + b

def upper_circle(x, x0, y0, r):
    """
    Returns the upper half of a circle for given x, center (x0, y0), and radius r.
    Only real values are returned (where |x - x0| <= r).
    """
    x = np.asarray(x)
    y = y0 + np.sqrt(np.maximum(0, r**2 - (x - x0)**2))
    return y

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def sinusoid(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


fitFunc = sinusoid
x_data = np.arange(len(Singal_Integral))
y_data = Singal_Integral
popt, pcov = curve_fit(fitFunc, x_data, y_data,p0=[1e6,1e-6,0.5e6,0])

ax.plot(x_data, fitFunc(x_data, *popt), 'r--', label='Linear Fit (last 500 points)')
print(f'Linear Fit Parameters: a={popt[0]}, b={popt[1]}')  


ax.set_xlabel('Number of bins added')
ax.set_ylabel('Signal Integral')
ax.legend(loc='upper left')
plt.show()

