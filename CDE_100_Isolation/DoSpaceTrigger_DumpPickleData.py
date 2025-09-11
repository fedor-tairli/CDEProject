import os
import numpy as np
import glob
import pickle



def ReadFile(filename,Data):
    with open(filename, 'r') as file:
        lines = file.readlines()
        This_Event = {}
        while len(lines) > 0:
            line = lines.pop(0)
            
            if line.startswith('# EventID:'): # Example -> # EventID: Batch_300001999:Shower_5515

                parts = line.strip().split(':')
                This_Event['Batch'] = int(parts[1].split('_')[1].strip())
                This_Event['Shower'] = int(parts[2].split('_')[1].strip())

            elif line.startswith('# EyeID:'): # Example -> # EyeID: 5
                parts = line.strip().split(':')
                This_Event['EyeID'] = parts[1].strip()
            
            elif line.startswith('# Gen_LogE:'):
                parts = line.strip().split(':')
                This_Event['Gen_LogE'] = float(parts[1].strip())

            elif line.startswith('# Gen_CosZen:'):
                parts = line.strip().split(':')
                This_Event['Gen_CosZen'] = float(parts[1].strip())

            elif line.startswith('# Gen_Xmax:'):
                parts = line.strip().split(':')
                This_Event['Gen_Xmax'] = float(parts[1].strip())

            elif line.startswith('# Gen_SDPPhi:'):
                parts = line.strip().split(':')
                This_Event['Gen_SDPPhi'] = float(parts[1].strip())

            elif line.startswith('# Gen_SDPTheta:'):
                parts = line.strip().split(':')
                This_Event['Gen_SDPTheta'] = float(parts[1].strip())

            elif line.startswith('# Gen_Chi0:'):
                parts = line.strip().split(':')
                This_Event['Gen_Chi0'] = float(parts[1].strip())

            elif line.startswith('# Gen_Rp:'):
                parts = line.strip().split(':')
                This_Event['Gen_Rp'] = float(parts[1].strip())

            elif line.startswith('# Gen_T0:'):
                parts = line.strip().split(':')
                This_Event['Gen_T0'] = float(parts[1].strip())

            elif line.startswith('# Gen_CoreEyeDist:'):
                parts = line.strip().split(':')
                This_Event['Gen_CoreEyeDist'] = float(parts[1].strip())

            elif line.startswith('# Gen_Primary:'):
                parts = line.strip().split(':')
                This_Event['Gen_Primary'] = int(parts[1].strip())

            elif line.startswith('# EventClass:'):
                parts = line.strip().split(':')
                This_Event['EventClass'] = parts[1].strip()

            elif line.startswith('# TotalPixels:'): # Example -> # TotalPixels: 134
                parts = line.strip().split(':')
                This_Event['TotalPixels'] = int(parts[1].strip())
            
            elif line.startswith('# PixelID, Status, Theta, Phi'):
                # Then the next N lines are lines of pixel data
                pixel_data = {'PixelID': [], 'Status': [], 'Theta': [], 'Phi': [] , 'Trace': [], 'RecTrigger': []}
                
                for _ in range(This_Event['TotalPixels']):
                    pixel_line = lines.pop(0).strip()
                    pixel_parts = pixel_line.split(',')
                    
                    pixel_data['PixelID'].append(int(pixel_parts[0].strip()))
                    pixel_data['Status'].append(int(pixel_parts[1].strip()))
                    pixel_data['Theta'].append(float(pixel_parts[2].strip()))
                    pixel_data['Phi'].append(float(pixel_parts[3].strip()))

                # Then the next N lines are lines of Trace data
                line = lines.pop(0)
                if not line.startswith('# Signal Array (rows: pixels, columns: time bins)'):
                    print(f"Expected Trace header line, got: {line.strip()}")
                    return
                
                for _ in range(This_Event['TotalPixels']):
                    trace_line = lines.pop(0).strip()
                    trace_parts = trace_line.split(',')
                    if trace_parts[-1] == '':trace_parts = trace_parts[:-1]
                    # Try to make a numpy float array
                    trace_array = np.array([float(x) for x in trace_parts[1:]], dtype=np.float32)
                    pixel_data['Trace'].append(trace_array)

                # Then the next N lines are lines of RecTrigger data
                line = lines.pop(0)
                if not line.startswith('# Rec Trigger Array (rows: pixels, columns: time bins)'):
                    print(f"Expected RecTrigger header line, got: {line.strip()}")
                    return
                
                for i in range(This_Event['TotalPixels']):
                    rec_line = lines.pop(0).strip()
                    # sometimes a single zero is written instead of a full line of zeros
                    if rec_line == '0':
                        rec_array = np.zeros(len(pixel_data['Trace'][i]), dtype=np.int32)
                        pixel_data['RecTrigger'].append(rec_array)
                    else: 

                        rec_parts = rec_line.split(',')
                        if rec_parts[-1] == '':rec_parts = rec_parts[:-1]
                        rec_array = np.array([int(x) for x in rec_parts[1:]], dtype=np.int32)
                        pixel_data['RecTrigger'].append(rec_array)
                    
                This_Event['PixelData'] = pixel_data
            else:
                print(f"Unrecognized line: {line.strip()}")
        
        # Check if the Trace and RecTrigger lengths match for all pixels
        # Due to some pixels being in differet mirrors, there may be an offset
        LongestTrace = max(len(t) for t in This_Event['PixelData']['Trace'])
        
        for i in range(This_Event['TotalPixels']):
            # Pad Trace at the back with zeros if needed
            trace_len = len(This_Event['PixelData']['Trace'][i])
            if trace_len < LongestTrace:
                pad_width = LongestTrace - trace_len
                This_Event['PixelData']['Trace'][i] = np.pad(This_Event['PixelData']['Trace'][i], (0, pad_width), mode='constant')
            # Pad RecTrigger at the back with zeros if needed
            rec_len = len(This_Event['PixelData']['RecTrigger'][i])
            if rec_len < LongestTrace:
                pad_width = LongestTrace - rec_len
                This_Event['PixelData']['RecTrigger'][i] = np.pad(This_Event['PixelData']['RecTrigger'][i], (0, pad_width), mode='constant')
            
            # Check that the lengths of rec and trace match now
            if len(This_Event['PixelData']['Trace'][i]) != len(This_Event['PixelData']['RecTrigger'][i]):
                print(f"Length mismatch for pixel {This_Event['PixelData']['PixelID'][i]}: Trace length {len(This_Event['PixelData']['Trace'][i])}, RecTrigger length {len(This_Event['PixelData']['RecTrigger'][i])}")
        
        # Now that the lengths match, convert lists to numpy arrays
        This_Event['PixelData']['PixelID']    = np.array(This_Event['PixelData']['PixelID'], dtype=np.int32)
        This_Event['PixelData']['Status']     = np.array(This_Event['PixelData']['Status'], dtype=np.int32)
        This_Event['PixelData']['Theta']      = np.array(This_Event['PixelData']['Theta'], dtype=np.float32)
        This_Event['PixelData']['Phi']        = np.array(This_Event['PixelData']['Phi'], dtype=np.float32)
        This_Event['PixelData']['Trace']      = np.array(This_Event['PixelData']['Trace'], dtype=np.float16)
        This_Event['PixelData']['RecTrigger'] = np.array(This_Event['PixelData']['RecTrigger'], dtype=bool)

        Data.append(This_Event)

if __name__ == "__main__":

    data_path = "./ReadEvents/*"

    all_files = sorted(glob.glob(data_path ))
    print(f"Found {len(all_files)} files to process.")


    Data = []  # List to hold data from all files

    LoadPickle = False
    PickleName = 'DoSpaceTrigger_Data.pkl'

    if LoadPickle and os.path.exists(PickleName):
        print(f"Pickle file {PickleName} exists, loading data from it.")
        with open(PickleName, 'rb') as f:
            Data = pickle.load(f)
        print(f"Loaded data from {PickleName}, total events: {len(Data)}")
        
    else:
        for i,file in enumerate(all_files):
            print(f"Reading file {i+1}/{len(all_files)}: {file}", end='\r')
            ReadFile(file,Data)
        
        print(f"\nFinished reading files, total events: {len(Data)}")
        with open(PickleName, 'wb') as f:
            pickle.dump(Data, f)
