#!/usr/bin/env python

'''
File name: visualize_dataset.py
Author: Thomas Haas, KUL (thomas.haas@kuleuven.be)
Date: 14.12.2018
Editor: Eske Freiesleben, DTU
Date: 16.03.2026

Description:
Functions for import, scaling and processing of AWESCO Wind Field Datasets

Additional: Visualization functions of AWESCO Wind Field Datasets
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_metadata(fname, case, **kwargs):

    # Initialize metadata
    metadata = {'case':case, 'fname':fname}

    # Scaling
    scale = {}

    if case == 'c1':
        z0 = 0.03
        scale['default'] = {'L': 1000.0, 'u':9.0, 'z':2.0, 'z0':0.03, 'utau':0.44380}
    elif case == 'c2':
        z0 = 0.1
        scale['default'] = {'L': 1000.0, 'u':8.0, 'z':2.0, 'z0':0.1, 'utau':0.46325}
    if kwargs:
        print('User defined scaling')
        print(len(kwargs))
        if len(kwargs)==2:
            utau = kwargs['u']*0.4/(np.log(kwargs['z']/z0))
            scale['user'] = {'L': 1000.0, 'u': kwargs['u'], 'z': kwargs['z'], 'z0': z0, 'utau': utau}
        else:
            print('scaling: missing argument - required arguments: u, z')

    # Define scale to use
    key = 'user'
    if key in scale.keys():
        s = scale['user']
    else:
        s = scale['default']
    metadata['scaling'] = s

    # Retrieve scaled metadata from dataset file
    f = h5py.File(fname, 'r')
    # > Time and space vectors
    metadata['t'] = (s['L']/s['utau'])*np.array(f['time_array'])
    metadata['x'] = s['L']*np.array(f['xvec'])
    metadata['y'] = s['L']*np.array(f['yvec'])
    metadata['z'] = s['L']*np.array(f['zvec'])
    # > Time and space resolution
    metadata['Nt'] = len(metadata['t'])
    metadata['Nx'] = len(metadata['x'])
    metadata['Ny'] = len(metadata['y'])
    metadata['Nz'] = len(metadata['z'])
    # > Time and space discretization
    metadata['dt'] = metadata['t'][1]-metadata['t'][0]
    metadata['dx'] = metadata['x'][1]-metadata['x'][0]
    metadata['dy'] = metadata['y'][1]-metadata['y'][0]
    metadata['dz'] = metadata['z'][1]-metadata['z'][0]
    f.close()

    # Min/max velocity
    metadata['scaling']['vmin'] = s['u']/0.4*np.log(metadata['z'][0]/s['z0'])
    metadata['scaling']['vmax'] = s['u']/0.4*np.log(s['L']/s['z0'])

    return metadata

def get_volume(metadata, var, nt):
    fname = metadata['fname']
    with h5py.File(fname, 'r') as f:
        # Load from your 4D slice: [Time, Z, Y, X]
        field_3d = f[var][nt, :, :, :] 
    # Scale by utau and return in Z, Y, X order
    return field_3d * metadata['scaling']['utau']

def plot_plane(ax, u_slice, metadata, var, q, i_idx, ttl, um):
    # Create physical coordinate grids based on metadata
    # metadata['x'], ['y'], ['z'] are in meters
    X, Y, Z = np.meshgrid(metadata['x'], metadata['y'], metadata['z'], indexing='ij')
    
    if q == 'x': # Front View (Y-Z plane)
        # i_idx is the index along the X-axis
        x = np.squeeze(Y[i_idx, :, :])
        y = np.squeeze(Z[i_idx, :, :])
        ax.set(xlabel='y [m]', ylabel='z [m]')
        
    elif q == 'y': # Side View (X-Z plane)
        # i_idx is the index along the Y-axis
        x = np.squeeze(X[:, i_idx, :])
        y = np.squeeze(Z[:, i_idx, :])
        ax.set(xlabel='x [m]', ylabel='z [m]')
        
    elif q == 'z': # Top View (X-Y plane)
        # i_idx is the index along the Z-axis
        x = np.squeeze(X[:, :, i_idx])
        y = np.squeeze(Y[:, :, i_idx])
        ax.set(xlabel='x [m]', ylabel='y [m]')

    # Use pcolormesh for a smooth CFD-style plot
    # Note: u_slice must be 2D
    pcm = ax.pcolormesh(x, y, u_slice, vmin=um[0], vmax=um[1], 
                        shading='gouraud', cmap='viridis')
    ax.set_title(ttl)
    return pcm, ax

def plot_slices(V, metadata, var, nt):
    """
    V is a 3D volume with shape (Nz, Ny, Nx) or (Nx, Ny, Nz)
    Based on our new 'get_volume', it is (Nz, Ny, Nx)
    """
    titles = {'u': 'Streamwise', 'v': 'Spanwise', 'w': 'Vertical'}
    ttl = f"{titles.get(var, '')} velocity at t={metadata['t'][nt]:.2f}s "

    # Calculate center indices
    # Assumes V shape is (Nz, Ny, Nx)
    nz, ny, nx = V.shape
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2
    
    um = [np.floor(np.min(V)), np.ceil(np.max(V))]

    # --- 1. FRONT VIEW (Y-Z Plane at middle of X) ---
    figx, ax = plt.subplots(figsize=(6, 4))
    # Extracting slice: All Z, All Y, at middle X index
    u_slice_x = V[:, :, mid_x] 
    pcm, ax = plot_plane(ax, u_slice_x, metadata, var, 'x', mid_x, 
                         ttl + f'(Front view x={metadata["x"][mid_x]:.1f}m)', um)
    plt.colorbar(pcm, ax=ax, label=f'{var} [m/s]')

    # --- 2. SIDE VIEW (X-Z Plane at middle of Y) ---
    figy, ax = plt.subplots(figsize=(6, 4))
    # Extracting slice: All Z, at middle Y index, All X
    u_slice_y = V[:, mid_y, :]
    pcm, ax = plot_plane(ax, u_slice_y, metadata, var, 'y', mid_y, 
                         ttl + f'(Side view y={metadata["y"][mid_y]:.1f}m)', um)
    plt.colorbar(pcm, ax=ax, label=f'{var} [m/s]')

    # --- 3. TOP VIEW (X-Y Plane at middle of Z) ---
    figz, ax = plt.subplots(figsize=(6, 4))
    # Extracting slice: At middle Z index, All Y, All X
    u_slice_z = V[mid_z, :, :]
    pcm, ax = plot_plane(ax, u_slice_z, metadata, var, 'z', mid_z, 
                         ttl + f'(Top view z={metadata["z"][mid_z]:.1f}m)', um)
    plt.colorbar(pcm, ax=ax, label=f'{var} [m/s]')

    return [figx, figy, figz]

def plot_pod_energy(relative_energy, cumulative_energy):
    # plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Individual Mode Energy (Scree Plot) - Blue Side
    ax1.set_xlabel('Mode Number', fontsize=12)
    ax1.set_ylabel('Individual Energy (%)', color='tab:blue', fontsize=12)
    line1, = ax1.plot(range(1, 101), relative_energy[:100], 'o-', color='tab:blue', markersize=4, label='Individual Energy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, max(relative_energy[:100]) * 1.1) # Give some headroom

    # Cumulative Energy - Red Side
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Energy (%)', color='tab:red', fontsize=12)
    line2, = ax2.plot(range(1, 101), cumulative_energy[:100], 's--', color='tab:red', markersize=4, label='Cumulative Energy')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(min(cumulative_energy[:100]), 105)

    # 90% Threshold Line
    thresh_line = ax2.axhline(y=90, color='gray', linestyle=':', linewidth=2, label='90% Threshold')

    # Combine legends from both axes
    lines = [line1, line2, thresh_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('POD Energy Decay and Cumulative Energy (First 100 Modes)', fontsize=14)
    plt.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()

def plot_pod_modes(U_modes, Nz, Ny, Nx, modes_to_show=[0, 4, 9, 19, 29, 49, 79, 99, 149], z_slice = 3):
    # Create the 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten() # Flatten to 1D for easy looping
    
    for i, mode_idx in enumerate(modes_to_show):
        # 1. Reshape the column back to (X, Y, Z)
        mode_shape = U_modes[:, mode_idx].reshape(Nz, Ny, Nx)
        
        # 2. Extract the horizontal slice at the chosen height
        # This gives us a (Ny, Nx) array
        slice_2d = mode_shape[z_slice, :, :]
        
        # 3. Plotting
        im = axes[i].imshow(slice_2d, 
                            cmap='RdBu_r', 
                            interpolation='bicubic', 
                            origin='lower',
                            aspect='auto') # 'auto' helps if the 500x500 area has unequal grid points
        
        # Add labels and styling
        axes[i].set_title(f"POD Mode {mode_idx + 1}", fontsize=12)
        if i >= 6: axes[i].set_xlabel("X (Length)")
        if i % 3 == 0: axes[i].set_ylabel("Y (Width)")
        
        # Optional: Add a colorbar to each subplot
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f"Top 9 POD Modes - Horizontal Slice at Z-Index {z_slice}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()