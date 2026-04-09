import xarray as xr
import xesmf as xe
import numpy as np

# 1. Load your curvilinear NetCDF
ds = xr.open_dataset('flowdata_terrain_mb.nc')

# 2. Define the 'Target' Cartesian Grid
# xESMF expects the target to be a dataset with 'lat' and 'lon' 
# (or 'x' and 'y') coordinates.
ds_out = xr.Dataset(
    {
        "x": (["x"], np.linspace(ds.x.min(), ds.x.max(), 500)),
        "y": (["y"], np.linspace(ds.y.min(), ds.y.max(), 500)),
    }
)

# 3. Create the Regridder
# 'bilinear' or 'patch' are common; 'patch' is better for preserving 
# the shape of a Gaussian hill as it minimizes noise.
regridder = xe.Regridder(ds, ds_out, method='patch')

# 4. Transform the data
ds_cartesian = regridder(ds)

# Save back to a new NetCDF
ds_cartesian.to_netcdf('cartesian_output.nc')