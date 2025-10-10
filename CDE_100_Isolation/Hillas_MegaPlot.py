# File operations
import sys, os
import glob
import pickle

# Plotting
from matplotlib import pyplot as plt
from matplotlib import cm

# Processing
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

# Runtime warnings
print("Ignoring RuntimeWarnings")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



hostname = os.uname()

############################################### Processing Functions
# Processing_Functions

def Gradient_and_GOF(x,y,w):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var  = np.var(x)

    m = np.sum(w*(x - x_mean)*(y - y_mean)) / np.sum(w) / x_var

    t = y_mean - m*x_mean

    chi_squared = np.sum(w*(y - (m*x + t))**2) / np.sum(w)

    return m,chi_squared





def Calculate_Hillas_Values(Event):
    # Info
    Possible_Telscope_OA_Phis = np.array([44.45,89.87,132.83])
    Possible_Telescope_OA_Thetas = np.array([45,45,45])

    Pixel_Data = Event['PixelData']
    Triggered_Mask = Event['MyyTrigger'].sum(axis=1)>0  


    Pixel_Amplitudes = np.sum(Pixel_Data['Trace']*Event['MyyTrigger'], axis=1).astype(float)

    pix_X_Cam = Pixel_Data['Phi']
    pix_Y_Cam = Pixel_Data['Theta']

    pix_X_Cam_Mean = np.mean(pix_X_Cam*Pixel_Amplitudes) / np.mean(Pixel_Amplitudes)
    pix_Y_Cam_Mean = np.mean(pix_Y_Cam*Pixel_Amplitudes) / np.mean(Pixel_Amplitudes)

    # Transfrom the pix_X,Y_Cam to pix_X,Y
    # Minimise the spread in Y
    # X = cos(alpha)*(X_Cam - X_Cam_Mean) - sin(alpha)*(Y_Cam - Y_Cam_Mean)
    # Y = sin(alpha)*(X_Cam - X_Cam_Mean) + cos(alpha)*(Y_Cam - Y_Cam_Mean)


    def Y_Spread(alpha):
        Y = np.sin(alpha)*(pix_X_Cam - pix_X_Cam_Mean) + np.cos(alpha)*(pix_Y_Cam - pix_Y_Cam_Mean)
        spread = np.sum(Pixel_Amplitudes*(Y)**2) / np.sum(Pixel_Amplitudes)
        return spread
    
    res = minimize_scalar(Y_Spread, bounds=[0,2*np.pi], method='bounded')
    alpha = res.x
    # res = minimize(Y_Spread, bounds=[(-2 * np.pi, 2 * np.pi)], method='L-BFGS-B')
    # alpha = res.x[0]
    

    X = np.cos(alpha)*(pix_X_Cam - pix_X_Cam_Mean) - np.sin(alpha)*(pix_Y_Cam - pix_Y_Cam_Mean)
    Y = np.sin(alpha)*(pix_X_Cam - pix_X_Cam_Mean) + np.cos(alpha)*(pix_Y_Cam - pix_Y_Cam_Mean)

    # 1
    H_Amplitude = np.sum(Pixel_Amplitudes)

    # 2
    H_Npix = np.sum(Pixel_Amplitudes > 0)

    # 3
    H_Distance = np.min(np.sqrt((pix_X_Cam_Mean-Possible_Telscope_OA_Phis)**2 + (pix_Y_Cam_Mean-Possible_Telescope_OA_Thetas)**2))

    # 4 
    H_Width = np.sqrt(np.sum(Pixel_Amplitudes*(Y)**2) / H_Amplitude)

    # 5
    H_Length = np.sqrt(np.sum(Pixel_Amplitudes*(X)**2) / H_Amplitude)

    # 6
    H_Skewness = np.sum(Pixel_Amplitudes*(X)**3) / H_Amplitude / H_Length**3

    # 7
    H_Kurtosis = np.sum(Pixel_Amplitudes*(X)**4) / H_Amplitude / H_Length**4

    # Composite Values

    # 8
    r = np.sqrt((X/H_Length)**2 + (Y/H_Width)**2)
    H_Grad_Profile, H_GOF_Profile = Gradient_and_GOF(r, np.log10(np.clip(Pixel_Amplitudes,a_min = 1, a_max = np.inf)), Pixel_Amplitudes)


    # 9 - Time Profiles    
    
    # Calculate Pixel Centroids here
    Centroids = np.sum(Pixel_Data['Trace']*Event['MyyTrigger']*np.arange(Pixel_Data['Trace'].shape[1]), axis=1) / np.sum(Pixel_Data['Trace']*Event['MyyTrigger'], axis=1)
    Centroids = Centroids[Triggered_Mask]
    Centroids -= np.min(Centroids)
    # print(Triggered_Mask)
    # print(Centroids)

    H_Grad_Time_Major, H_GOF_Time_Major = Gradient_and_GOF(X[Triggered_Mask], Centroids, Pixel_Amplitudes[Triggered_Mask])
    H_Grad_Time_Minor, H_GOF_Time_Minor = Gradient_and_GOF(Y[Triggered_Mask], Centroids, Pixel_Amplitudes[Triggered_Mask])

    Hillas_Values = {   "H_Amplitude": H_Amplitude,
                        "H_Npix"     : H_Npix,
                        "H_Distance" : H_Distance,
                        "H_Width"    : H_Width,
                        "H_Length"   : H_Length,
                        "H_Skewness" : H_Skewness,
                        "H_Kurtosis" : H_Kurtosis,
                        "H_Grad_Profile"    : H_Grad_Profile,
                        "H_GOF_Profile"     : H_GOF_Profile,
                        "H_Grad_Time_Major" : H_Grad_Time_Major,
                        "H_GOF_Time_Major"  : H_GOF_Time_Major,
                        "H_Grad_Time_Minor" : H_Grad_Time_Minor,
                        "H_GOF_Time_Minor"  : H_GOF_Time_Minor,
                        # "Triggered"         : Triggered_Mask,
                        "H_alpha"           : alpha,
                        }

    Event['HillasValues'] = Hillas_Values




############################################### Variable Lists which will be plotted

All_H_Amplitude       = []
All_H_Distance        = []
All_H_Width           = []
All_H_Length          = []
All_H_Skewness        = []
All_H_Kurtosis        = []
All_H_GOF_Profile     = []
All_H_Grad_Profile    = []
All_H_GOF_Time_Major  = []
All_H_Grad_Time_Major = []
All_H_GOF_Time_Minor  = []
All_H_Grad_Time_Minor = []
All_H_Npix            = []
All_H_alpha           = []

# Descriptive Parameters

All_GenLogE              = []
All_GenCosZen            = []
All_GenXmax              = []
All_GenSDPTheta          = []
All_GenSDPPhi            = []
All_GenChi0              = []
All_GenRp                = []
All_GenCherenkovFraction = []
All_GenPrimary           = []
All_EventClass           = []
All_Highest_Bins         = []



data_path = "./PickledData/*"
all_files = sorted(glob.glob(data_path))
all_files = [f for f in all_files if 'DoSpaceTrigger_Data_Batch' in f]

print(f'Found {len(all_files)} files')
# print(all_files)

## Reading Loop

Force_Recalculation = False
if Force_Recalculation:
    print("Forcing recalculation of Hillas values for all events.")

Suppress_ZeroSizeArray_Errors = True
if Suppress_ZeroSizeArray_Errors:
    print("Suppressing errors due to zero-size arrays when calculating Hillas values.")

# all_files = all_files[:3]
# print("Processing only first 3 files for testing.")

Use_Saturation_Events = False
Pixel_Signal_Threshold = 10000

for file in all_files:
    MadeChangesToData = False
    with open(file, 'rb') as f:
        Data = pickle.load(f)
        print(f'Processing file: {file} with {len(Data)} events')

        for i_Event, Event in enumerate(Data):
            if (not 'HillasValues' in Event) or Force_Recalculation:
                try: 
                    Calculate_Hillas_Values(Event)
                    MadeChangesToData = True
                except Exception as e:
                    if ("zero-size array" in str(e)) and Suppress_ZeroSizeArray_Errors:
                        continue
                    else:
                        print(f"Error calculating Hillas values for event {i_Event} in file {file}: {e}")
                        continue
                if not MadeChangesToData: print("    Will update this file with new Hillas values.")
            Pixel_Traces = Event['PixelData']['Trace']
            Highest_Bin_Signal = np.max(Pixel_Traces)

            if Use_Saturation_Events and (Highest_Bin_Signal < Pixel_Signal_Threshold): continue
                
            # Hillas Values 
            All_H_Amplitude      .append(Event['HillasValues']['H_Amplitude']      )
            All_H_Distance       .append(Event['HillasValues']['H_Distance']       )
            All_H_Width          .append(Event['HillasValues']['H_Width']          )
            All_H_Length         .append(Event['HillasValues']['H_Length']         )
            All_H_Skewness       .append(Event['HillasValues']['H_Skewness']       )
            All_H_Kurtosis       .append(Event['HillasValues']['H_Kurtosis']       )
            All_H_GOF_Profile    .append(Event['HillasValues']['H_GOF_Profile']    )
            All_H_Grad_Profile   .append(Event['HillasValues']['H_Grad_Profile']   )
            All_H_GOF_Time_Major .append(Event['HillasValues']['H_GOF_Time_Major'] )
            All_H_Grad_Time_Major.append(Event['HillasValues']['H_Grad_Time_Major'])
            All_H_GOF_Time_Minor .append(Event['HillasValues']['H_GOF_Time_Minor'] )
            All_H_Grad_Time_Minor.append(Event['HillasValues']['H_Grad_Time_Minor'])
            All_H_Npix           .append(Event['HillasValues']['H_Npix']           )
            All_H_alpha          .append(Event['HillasValues']['H_alpha']          )

            # Descriptive Parameters

            All_GenLogE             .append(Event['Gen_LogE']             )
            All_GenCosZen           .append(Event['Gen_CosZen']           )
            All_GenXmax             .append(Event['Gen_Xmax']             )
            All_GenSDPTheta         .append(Event['Gen_SDPTheta']         )
            All_GenSDPPhi           .append(Event['Gen_SDPPhi']           )
            All_GenChi0             .append(Event['Gen_Chi0']             )
            All_GenRp               .append(Event['Gen_Rp']               )
            All_GenCherenkovFraction.append(Event['Gen_CherenkovFraction'])
            All_GenPrimary          .append(Event['Gen_Primary']          )
            All_EventClass          .append(Event['EventClass']           )
            All_Highest_Bins        .append(Highest_Bin_Signal            )

            if (i_Event+1) % 1000 == 0:
                print(f"    Processed {i_Event+1}/{len(Data)} events", end='\r')
        print(f"    Processed {i_Event+1}/{len(Data)} events", end='\r')
    
    if MadeChangesToData:
        with open(file, 'wb') as f:
            pickle.dump(Data, f)
        print(f"\n    Updated file: {file} with new Hillas values.")

    print()
    print('------------------------------------------------------------------------------------------')
print("Finished processing all files.")





################################################################## Convert to numpy arrays

# Hillas Values
All_H_Amplitude       = np.array(All_H_Amplitude       )
All_H_Distance        = np.array(All_H_Distance        )
All_H_Width           = np.array(All_H_Width           )
All_H_Length          = np.array(All_H_Length          )
All_H_Skewness        = np.array(All_H_Skewness        )
All_H_Kurtosis        = np.array(All_H_Kurtosis        )
All_H_GOF_Profile     = np.array(All_H_GOF_Profile     )
All_H_Grad_Profile    = np.array(All_H_Grad_Profile    )
All_H_GOF_Time_Major  = np.array(All_H_GOF_Time_Major  )
All_H_Grad_Time_Major = np.array(All_H_Grad_Time_Major )
All_H_GOF_Time_Minor  = np.array(All_H_GOF_Time_Minor  )
All_H_Grad_Time_Minor = np.array(All_H_Grad_Time_Minor )
All_H_Npix            = np.array(All_H_Npix            )
All_H_alpha           = np.array(All_H_alpha           )

# Descriptive Parameters
All_GenLogE              = np.array(All_GenLogE              )
All_GenCosZen            = np.array(All_GenCosZen            )
All_GenXmax              = np.array(All_GenXmax              )
All_GenSDPTheta          = np.array(All_GenSDPTheta          )
All_GenSDPPhi            = np.array(All_GenSDPPhi            )
All_GenChi0              = np.array(All_GenChi0              )
All_GenRp                = np.array(All_GenRp                )
All_GenCherenkovFraction = np.array(All_GenCherenkovFraction )
All_GenPrimary           = np.array(All_GenPrimary           )
All_EventClass           = np.array(All_EventClass           )
All_Highest_Bins         = np.array(All_Highest_Bins         )
print("Converted all lists to numpy arrays.")



All_H_Grad_Time_Major = np.abs(All_H_Grad_Time_Major)
print("Converted All_H_Grad_Time_Major to absolute values.")
print("This is done because the sign of the gradient depends on the")
print("direction of the shower in the camera, which is decided by alpha fitter preference.")

All_WL_Ratio = All_H_Width / All_H_Length
print("Calculated WL_Ratio = Width / Length")
print("This is a measure of the elongation of the image.")

All_Highest_Bins = np.log10(np.clip(All_Highest_Bins, a_min=1, a_max=np.inf))


##################################################################  Plotting

All_H_Values = {"Amplitude": All_H_Amplitude,
                "Distance" : All_H_Distance,
                "Width"    : All_H_Width,
                "Length"   : All_H_Length,
                "WL_Ratio" : All_WL_Ratio,
                "Skewness" : All_H_Skewness,
                "Kurtosis" : All_H_Kurtosis,
                "GOF_Profile" : All_H_GOF_Profile,
                "Grad_Profile": All_H_Grad_Profile,
                "GOF_Time_Major" : All_H_GOF_Time_Major,
                "Grad_Time_Major": All_H_Grad_Time_Major,
                "GOF_Time_Minor" : All_H_GOF_Time_Minor,
                "Grad_Time_Minor": All_H_Grad_Time_Minor,
                "Npix"     : All_H_Npix,
                "alpha"    : All_H_alpha
                }



All_Desc_Values = {  "GenLogE"   : All_GenLogE,
                     "GenCosZen" : All_GenCosZen,
                     "GenXmax"   : All_GenXmax,
                     "GenSDPTheta": All_GenSDPTheta,
                     "GenSDPPhi"  : All_GenSDPPhi,
                     "GenChi0"    : All_GenChi0,
                     "GenRp"      : All_GenRp,
                     "GenCherenkovFraction": All_GenCherenkovFraction,
                     "Log Highest Signal / Bin": All_Highest_Bins,
                 }



All_Desc_Values_CMAPS = {"GenLogE"   : cm.Purples,
                         "GenCosZen" : cm.Blues  ,
                         "GenXmax"   : cm.Greens ,
                         "GenSDPTheta": cm.Oranges,
                         "GenSDPPhi"  : cm.Oranges,
                         "GenChi0"    : cm.Reds   ,
                         "GenRp"      : cm.Reds   ,
                         "GenCherenkovFraction": cm.BuPu,
                         "Log Highest Signal / Bin": cm.GnBu
                          }
from matplotlib.colors import LogNorm

All_H_Values_limits = { "GOF_Time_Major": [0,10],
                        "Grad_Time_Major": [0,1],
                        "GOF_Time_Minor": [0,10],
                        "Grad_Time_Minor": [-1,1],
                        "Skewness": [-2,2],
                        "Kurtosis": [0,10],
                        "alpha"   : [0.5*np.pi, 1.5*np.pi],
                        "Npix": [0,50],
                         }

All_H_Values_scales = {"Amplitude": 'log',
                    #    "Length": 'log',
                    #    "Kurtosis": 'log',
                       "GOF_Profile": 'log',
                    #    "Npix": 'log',
                        }


All_Desc_Values_limits = {
                          "GenCherenkovFraction": [80,100],
                        }

All_Desc_Values_scales = {
                            "Log Highest Signal / Bin": 'linear',
                        }

UseLogNorm = False

fig, ax = plt.subplots(len(All_H_Values)+1,len(All_Desc_Values)+1, figsize=[80,10*len(All_H_Values)])

for v,value in enumerate(All_H_Values.keys()):

    X = All_H_Values[value]
    if value in All_H_Values_limits:
        limits = All_H_Values_limits[value]
        X_Finite_Mask = (X > limits[0]) & (X < limits[1]) & np.isfinite(X)
    else:
        X_Finite_Mask = np.isfinite(X)
    
    if value in All_H_Values_scales:
        if All_H_Values_scales[value] == 'log':
            X_Finite_Mask = X_Finite_Mask & (X>0)
            X = X[X_Finite_Mask]
            X = np.log10(X)
    else:
        X = X[X_Finite_Mask]
    
    ax[v+1,0].hist(X, bins=50, density=True, orientation='horizontal')
    ax[v+1,0].set_ylabel(value)
    ax[v+1,0].set_xlabel("Density")
    # ax[v+1,0].set_xscale('log')
    ax[v+1,0].grid()
    ax[v+1,0].invert_xaxis()



    for i, (desc_name, desc_value) in enumerate(All_Desc_Values.items()):
        desc_value = desc_value[X_Finite_Mask]
        if desc_name in All_Desc_Values_limits:
            limits = All_Desc_Values_limits[desc_name]
            desc_value_limit_mask = (desc_value > limits[0]) & (desc_value < limits[1]) & np.isfinite(desc_value)
            desc_value = desc_value[desc_value_limit_mask]
            X_limited = X[desc_value_limit_mask]
        else:
            X_limited = X

        if UseLogNorm:
            ax[v+1,i+1].hist2d(desc_value, X_limited, bins=50, density=True, cmap=All_Desc_Values_CMAPS[desc_name], norm = LogNorm())
        else:
            ax[v+1,i+1].hist2d(desc_value, X_limited, bins=50, density=True, cmap=All_Desc_Values_CMAPS[desc_name])

        ax[v+1,i+1].set_xlabel(desc_name)
        # ax[v+1,i+1].set_ylabel(value)
        # ax[v+1,i+1].grid()

for i, (desc_name, desc_value) in enumerate(All_Desc_Values.items()):
    if desc_name in All_Desc_Values_limits:
        limits = All_Desc_Values_limits[desc_name]
        desc_value = desc_value[(desc_value > limits[0]) & (desc_value < limits[1]) & np.isfinite(desc_value)]
        
    ax[0,i+1].hist(desc_value, bins=50, density=True,color=All_Desc_Values_CMAPS[desc_name](0.75))
    ax[0,i+1].set_xlabel(desc_name)
    ax[0,i+1].set_ylabel("Density")
    ax[0,i+1].grid()

ax[0,0].axis('off')

plt.tight_layout()

if Use_Saturation_Events:
    FileName = "Hillas_Parameters_Distribution_MegaPlot_SaturationEvents"
else:
    FileName = "Hillas_Parameters_Distribution_MegaPlot_AllEvents"

if UseLogNorm:
    FileName += "_LgNorm"



plt.savefig(f"{FileName}.png")
plt.savefig(f"{FileName}.pdf")


print(f"Saved {FileName}.png/pdf")



