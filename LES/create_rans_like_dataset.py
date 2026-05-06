"""
Create an xarray dataset similar to RANS format but with LES data.

This script loads LES velocity data and creates a synthetic RANS-like dataset with:
- Data variables: u, v, w, tke_x, tke_y, tke_z, epsilon
- Coordinates: t, x, y, z

The velocity components are derived from the available LES data,
and turbulent quantities are computed from fluctuations.
"""

import numpy as np
import xarray as xr
from tools import get_metadata, get_volume


def create_rans_like_dataset(h5_file, case, u_ref=None, z_ref=None, n_samples=None):
    """
    Create a RANS-like xarray dataset from LES HDF5 data.
    
    Parameters
    ----------
    h5_file : str
        Path to the HDF5 file (e.g., 'golf_slice_c1.h5')
    case : str
        Case identifier ('c1' or 'c2')
    u_ref : float, optional
        Reference velocity for scaling. If None, uses default for case.
    z_ref : float, optional
        Reference height for scaling. If None, uses default for case.
    n_samples : int, optional
        Number of time samples to load. If None, loads all available.
    
    Returns
    -------
    xr.Dataset
        Dataset with variables (u, v, w, tke_x, tke_y, tke_z, epsilon)
        and coordinates (t, x, y, z)
    """
    
    # Get metadata with optional user scaling
    kwargs = {}
    if u_ref is not None and z_ref is not None:
        kwargs = {'u': u_ref, 'z': z_ref}
    metadata = get_metadata(h5_file, case, **kwargs)
    
    # Determine number of samples
    Nt = metadata['Nt']
    if n_samples is not None:
        Nt = min(n_samples, Nt)
    
    print(f"Loading {Nt} time samples from {h5_file}")
    print(f"Grid dimensions: Nz={metadata['Nz']}, Ny={metadata['Ny']}, Nx={metadata['Nx']}")
    
    # Extract coordinates
    t = metadata['t'][:Nt]
    x = metadata['x']
    y = metadata['y']
    z = metadata['z']
    
    # Load velocity data: u-component
    print("Loading u-component velocity...")
    u_data = np.array([get_volume(metadata, 'u', nt=nt) for nt in range(Nt)])  # Shape: (Nt, Nz, Ny, Nx)
    v_data = np.array([get_volume(metadata, 'v', nt=nt) for nt in range(Nt)])
    w_data = np.array([get_volume(metadata, 'w', nt=nt) for nt in range(Nt)])
    z0 = metadata['scaling']['z0']
    u_ref = metadata['scaling']['u']
    z_ref = metadata['scaling']['z']
    kappa = 0.4
    u_star = u_ref * kappa / np.log(z_ref / z0)
    u_log = u_star/kappa * np.log(z / z0)
    u_log = u_log[np.newaxis, :, np.newaxis, np.newaxis]  # Shape: (1, Nz, 1, 1) for broadcasting

    u_prime = u_data - u_log  # Fluctuations
    v_prime = v_data  # Assuming mean v is zero
    w_prime = w_data  # Assuming mean w is zero
    
    # Compute turbulent kinetic energy components
    # tke_x = 0.5 * <u'^2>, tke_y = 0.5 * <v'^2>, tke_z = 0.5 * <w'^2>
    print("Computing turbulent kinetic energy components...")
    tke_x = 0.5 * (u_prime ** 2).mean(axis=0)  # Shape: (Nz, Ny, Nx)
    tke_y = 0.5 * (v_prime ** 2).mean(axis=0)
    tke_z = 0.5 * (w_prime ** 2).mean(axis=0)
    
    # Compute total TKE
    tke_total = tke_x + tke_y + tke_z
    
    # Estimate epsilon (dissipation rate) using Kolmogorov theory
    # epsilon ~ k^1.5 / l, where l is a length scale
    # Simplified: use grid spacing as length scale
    print("Computing dissipation rate...")
    dx = metadata['dx']
    epsilon = np.where(tke_total > 0, 
                       0.09 * (tke_total ** 1.5) / dx,  # Simplified estimate
                       0)
    
    # Create xarray Dataset
    print("Creating xarray Dataset...")
    ds = xr.Dataset(
        data_vars={
            'u': (['t', 'z', 'y', 'x'], u_data),
            'v': (['t', 'z', 'y', 'x'], v_data),
            'w': (['t', 'z', 'y', 'x'], w_data),
            'tke_x': (['z', 'y', 'x'], tke_x),
            'tke_y': (['z', 'y', 'x'], tke_y),
            'tke_z': (['z', 'y', 'x'], tke_z),
            'epsilon': (['z', 'y', 'x'], epsilon),
        },
        coords={
            't': t,
            'x': x,
            'y': y,
            'z': z,
        },
        attrs={
            'case': case,
            'source_file': h5_file,
            'Nt': Nt,
            'dt': metadata['dt'],
            'dx': metadata['dx'],
            'dy': metadata['dy'],
            'dz': metadata['dz'],
        }
    )
    
    # Add metadata as attributes
    scaling = metadata['scaling']
    ds.attrs['u_ref'] = scaling['u']
    ds.attrs['z_ref'] = scaling['z']
    ds.attrs['z0'] = scaling['z0']
    ds.attrs['utau'] = scaling['utau']
    ds.attrs['L'] = scaling['L']
    
    return ds


def save_dataset(ds, output_file):
    """Save the dataset to a NetCDF file."""
    print(f"Saving dataset to {output_file}...")
    ds.to_netcdf(output_file)
    print(f"Dataset saved successfully.")
    return output_file


def load_rans_like_dataset(netcdf_file):
    """Load a saved RANS-like dataset."""
    return xr.open_dataset(netcdf_file)


if __name__ == "__main__":
    # Example usage
    import os
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_file = os.path.join(script_dir, 'golf_slice_c1.h5')
    output_file = os.path.join(script_dir, 'rans_like_c1.nc')
    
    # Create dataset
    ds = create_rans_like_dataset(h5_file, case='c1', n_samples=100, u_ref=8, z_ref=90)
    
    # Display dataset info
    print("\nDataset created:")
    print(ds)
    
    # # Save to NetCDF
    save_dataset(ds, output_file)
    
    # # Verify by loading
    # ds_loaded = load_rans_like_dataset(output_file)
    # print("\nDataset loaded from NetCDF:")
    # print(ds_loaded)
