import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np

trans = mtransforms.Affine2D().rotate_deg(30)
hexagon_marker = mpath.Path.unit_regular_polygon(6)
hexagon_marker = hexagon_marker.transformed(trans)

def plot_hex_array(array1, array2, mask, S=2500):
    nrows, ncols = array1.shape
    assert array1.shape == array2.shape == mask.shape, "All arrays must have the same shape"
    
    # Define offsets for odd and even columns
    Xpos  = np.linspace(-np.cos(np.pi/6)*ncols,np.cos(np.pi/6)*ncols,ncols)
    Ypos = np.linspace(-10,10,nrows)
    
    # Create the color map
    color_map = plt.get_cmap('viridis')
    colors1 = color_map(array1.flatten())
    colors2 = color_map(array2.flatten())

    fig, axs = plt.subplots(ncols = 2,nrows=1,figsize=(20,10))
    for ax, array, colors, title in zip(axs, [array1, array2], [colors1, colors2], ['Signal', 'Arrival Time']):
        for i in range(nrows):
            for j in range(ncols):
                color = 'gray' if mask[i, j] == 0 else colors[i * ncols + j]
                if j % 2 == 0:
                    ax.scatter(Xpos[j],Ypos[i]+1,s=S,edgecolors='k',color=color,marker = hexagon_marker)
                else:
                    ax.scatter(Xpos[j],Ypos[i]  ,s=S,edgecolors='k',color=color,marker = hexagon_marker)

        ax.axis('off')
        ax.set_aspect('equal') # ensure hexagons aren't distorted
        ax.set_title(title)
        fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(np.min(array), np.max(array))), ax=ax, shrink=0.7)

        ax.set_xlim(-11,11)
        ax.set_ylim(-12,13)
    plt.show()





def plot_mirror(Array,Mask = None):
    assert Array.shape == (22,20), 'Array shape is {}'.format(Array.shape)

    if Mask is None:
        Mask = np.zeros_like(Array)
    # Define offsets for odd and even rows
    Xpos_even = np.linspace(-15.75,14.75,20)
    Xpos_odd  = np.linspace(-14.75,15.75,20)
    Ypos = np.linspace(1.6,30.4,22)

    color_map = plt.get_cmap('viridis')
    colors = color_map(Array.flatten())

    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(22):
        for j in range(20):
            color = 'gray' if Mask[i, j] == 0 else colors[j * 20 + i]
            if i+1 % 2 == 0:
                ax.scatter(Xpos_even[j],Ypos[i],s=2500,edgecolors='k',color=color,marker = hexagon_marker)
            else:
                ax.scatter(Xpos_odd[j],Ypos[i],s=2500,edgecolors='k',color=color,marker = hexagon_marker)
    ax.axis('off')
    ax.set_aspect('equal') # ensure hexagons aren't distorted
    fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(np.min(Array), np.max(Array))), ax=ax, shrink=0.7)
    plt.show()



