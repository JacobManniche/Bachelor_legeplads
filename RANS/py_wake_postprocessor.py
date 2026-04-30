
import numpy as np
import xarray as xr
import os

from py_wake_ellipsys.wind_farm_models.ellipsys import EllipSys
from pyellipsys.inversemap import InverseMap

def process_rans(file, res=10.0, size=(512,512,100), filename=None):
    """
    Post-process RANS data to create a cartesian grid netcdf file.
    Parameters:
    - file: Path to the input RANS netcdf file.
    - res: Resolution in meters for the new x-y plane (default: 10.0).
    - size: Tuple specifying the size in each direction of x,y of the cartesian grid and size from 0 of z (default: (512,512,100)).
        - z will be logarithmically spaced from 0.1 to 100 meters with size[2] points.
    - filename: Optional name for the output cartesian netcdf file. If None, it will be generated from the input filename.
    """
    if filename is None:
        filename = os.path.splitext(os.path.basename(file))[0] + '_cartesian.nc'

    # 2. Open the dataset
    ds = xr.open_dataset(file)

    # Create new cartesian grid based on the bounds and resolution
    x_vec = np.arange(-size[0]//2, size[0]//2+res, res)
    y_vec = np.arange(-size[1]//2, size[1]//2+res, res)
    z_vec = np.logspace(-1, 2, num=size[2], base=10)  # Logarithmic spacing in z to capture near-ground details

    # 3D grid points
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
    n_points = X.size 
    
    # Add this after creating X, Y, Z
    print(f"Target grid size: {X.shape}")
    print(f"Total points per variable: {X.size:,}")
    if X.size > 10_000_000:
        raise ValueError("The target grid is too large. Consider increasing the resolution or using a smaller domain.")

    points = np.zeros((n_points, 3)) # Create a 3D array to hold the flattened points
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    points[:, 2] = Z.flatten()

    # Remap everything to cartesian grid using InverseMap
    IM = InverseMap()
    print('Interpreting...')
    Ui_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['U'], points[:, 0], points[:, 1], points[:, 2],
                    add_ghost_layer=False)

    Vi_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['V'], points[:, 0], points[:, 1], points[:, 2],
                    add_ghost_layer=False, make_inversemap=False, locate_points=False)

    Wi_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['W'], points[:, 0], points[:, 1], points[:, 2],
                    add_ghost_layer=False, make_inversemap=False, locate_points=False)

    Pi_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['P'], points[:, 0], points[:, 1], points[:, 2],
                    add_ghost_layer=False, make_inversemap=False, locate_points=False)

    muT_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['muT'], points[:, 0], points[:, 1], points[:, 2],
                        add_ghost_layer=False, make_inversemap=False, locate_points=False)
    tke_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['tke'], points[:, 0], points[:, 1], points[:, 2],
                        add_ghost_layer=False, make_inversemap=False, locate_points=False)

    epsilon_flat = IM.interp(ds['x'], ds['y'], ds['z'], ds['epsilon'], points[:, 0], points[:, 1], points[:, 2],
                            add_ghost_layer=False, make_inversemap=False, locate_points=False)


    shape = X.shape # take the shape of the 3D grid for reshaping
    
    # Reshape to 3D arrays
    U_3d = Ui_flat.reshape(shape)
    V_3d = Vi_flat.reshape(shape)
    W_3d = Wi_flat.reshape(shape)
    P_3d = Pi_flat.reshape(shape)
    muT_3d = muT_flat.reshape(shape)
    tke_3d = tke_flat.reshape(shape)
    epsilon_3d = epsilon_flat.reshape(shape)

    ds_cart = xr.Dataset(
        coords={
            'x': x_vec,
            'y': y_vec,
            'z': z_vec
        },
        data_vars={
            'U': (['x', 'y', 'z'], U_3d),
            'V': (['x', 'y', 'z'], V_3d),
            'W': (['x', 'y', 'z'], W_3d),
            'P': (['x', 'y', 'z'], P_3d),
            'muT': (['x', 'y', 'z'], muT_3d),
            'tke': (['x', 'y', 'z'], tke_3d),
            'epsilon': (['x', 'y', 'z'], epsilon_3d)
        }
    )

    ds_cart.attrs = ds.attrs
    ds_cart.attrs['resolution'] = res

    ds_cart.to_netcdf(filename)
    print(f"Cartesian grid netcdf file '{filename}' created successfully.")
    return ds_cart

if __name__ == "__main__":
    
    # Post process the RANS data to create a cartesian grid netcdf from curvilinear grid data
    file = os.path.abspath('../nc files/flowdata_terrain_mb.nc')
    process_rans(file, res=10.0)

