import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

AllPossiblePixelDirections = pd.read_csv('../../ReadingData/camera_view_table_with_HEAT_down.txt',sep=' ',header=None,names = ['EyeID','TelID','PixID','Theta','Phi'])



# Select a telescope Skeleton
TelescopeID = 1
EyeID       = 5
TelescopePixels = AllPossiblePixelDirections[ (AllPossiblePixelDirections['TelID'] == TelescopeID) & (AllPossiblePixelDirections['EyeID'] == EyeID) ]
All_Thetas = TelescopePixels['Theta'].values
All_Phis   = TelescopePixels['Phi'].values


def IndexToXY(indices,return_tensor=False):
    if isinstance(indices, int) or  isinstance(indices, float):
        indices -= 1
        X = indices//22
        Y = indices%22
        if return_tensor: return torch.tensor([X]).int(),torch.tensor([Y]).int()
        else:             return int(X),int(Y)
    else:

        indices -=1
        Xs = indices//22
        Ys = indices%22
        if return_tensor: return Xs.int(),Ys.int()
        else:             return Xs.int().tolist(),Ys.int().tolist()


def XYToIndex(Xs, Ys, return_tensor=False):
    """
    Convert X, Y coordinates back to pixel indices.
    Inverse of IndexToXY function.
    
    Parameters:
    -----------
    Xs : torch.Tensor or list
        X coordinates (row indices)
    Ys : torch.Tensor or list
        Y coordinates (column indices)
    return_tensor : bool
        If True, return as tensor; if False, return as list
    
    Returns:
    --------
    indices : torch.Tensor or list
        Pixel indices (1-indexed)
    """
    # Convert to tensors if needed
    if not isinstance(Xs, torch.Tensor):
        Xs = torch.tensor(Xs)
    if not isinstance(Ys, torch.Tensor):
        Ys = torch.tensor(Ys)
    
    # Reverse the operation: indices = Xs * 22 + Ys + 1
    indices = Xs * 22 + Ys + 1
    
    if return_tensor:
        return indices.int()
    else:
        return indices.int().tolist()
    
# range_threshold = 2 #deg

# for idx in range(1,441):
#     print(f'Pixel Index: {idx}', end ='')
#     pixel_theta = TelescopePixels[TelescopePixels['PixID'] == idx]['Theta'].values[0]
#     pixel_phi   = TelescopePixels[TelescopePixels['PixID'] == idx]['Phi'].values[0]
#     # print(f'  Theta: {pixel_theta}, Phi: {pixel_phi}')

#     i_three_dir = np.array([np.sin(np.deg2rad(pixel_theta))*np.cos(np.deg2rad(pixel_phi)),
#                             np.sin(np.deg2rad(pixel_theta))*np.sin(np.deg2rad(pixel_phi)),
#                             np.cos(np.deg2rad(pixel_theta))])
#     i_three_dir = i_three_dir / np.linalg.norm(i_three_dir)
#     neighbor_pixels = []
#     for jdx in range(1,441):
#         if jdx == idx:
#             continue
#         neighbor_theta = TelescopePixels[TelescopePixels['PixID'] == jdx]['Theta'].values[0]
#         neighbor_phi   = TelescopePixels[TelescopePixels['PixID'] == jdx]['Phi'].values[0]
    
#         j_three_dir = np.array([np.sin(np.deg2rad(neighbor_theta))*np.cos(np.deg2rad(neighbor_phi)),
#                                 np.sin(np.deg2rad(neighbor_theta))*np.sin(np.deg2rad(neighbor_phi)),
#                                 np.cos(np.deg2rad(neighbor_theta))])
#         j_three_dir = j_three_dir / np.linalg.norm(j_three_dir)


#         # print(f'i_three_dir: {i_three_dir}, j_three_dir: {j_three_dir}')
#         ang_div = np.rad2deg(np.arccos(np.clip(np.dot(i_three_dir,j_three_dir),-1.0,1.0)))
#         if ang_div <= range_threshold:
#             neighbor_pixels.append(jdx)
            
#             # print(f'    Checking against Pixel Index: {jdx}')
#             # print(f'      Angular Divergence: {ang_div} deg')    
#             # print(f'      Neighbor Theta: {neighbor_theta}, Neighbor Phi: {neighbor_phi}')
#             # print(f'      Is Neighbor: {ang_div <= range_threshold}')
#     print(f'  Neighbor Pixels: {neighbor_pixels}')
def Get_Pixel_Neighbour_Dict():

    neighbour_dict = {}

    for idx in range(1,441):

        n_l = idx - 22
        n_r = idx + 22
            
        neighbour_same_col_u = idx - 1        
        neighbour_same_col_d = idx + 1

        neighbour_side_col_u = n_l - 1 if idx % 2 == 0 else n_r - 1
        neighbour_side_col_d = n_l + 1 if idx % 2 == 0 else n_r + 1

        neighbor_pixels = []
        
        # Check boundaries and append neighbours
        
        # Left most column
        if idx <23:
            n_l = None
            if idx % 2 == 0:
                neighbour_side_col_u = None
                neighbour_side_col_d = None
        
        # Right most column
        if idx > 418:
            n_r = None
            if idx % 2 == 1:
                neighbour_side_col_u = None
                neighbour_side_col_d = None
            
        # Top row
        if idx % 22 == 1:
            neighbour_same_col_u = None
            neighbour_side_col_u = None
        # Bottom row
        if idx % 22 == 0:
            neighbour_same_col_d = None
            neighbour_side_col_d = None
        
        for n in [n_l, n_r, neighbour_same_col_u, neighbour_same_col_d, neighbour_side_col_u, neighbour_side_col_d]:
            if n is not None:
                neighbor_pixels.append(n)
        

        neighbour_dict[idx] = neighbor_pixels
    return neighbour_dict

        
        