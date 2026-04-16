# --- Replacement for InverseMap ---
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

infile = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/RANS/flowdata_terrain_mb.nc'
data = xr.open_dataset(infile)

H=300.0

# 1. Flatten the 3D curvilinear coordinates (i, j, k) into a list of points
# We use .values to get the numpy arrays from the xarray dataset
points_3d = np.array([
    data['x'].values.flatten(), 
    data['y'].values.flatten(), 
    data['z'].values.flatten()
]).T

# 2. Define the target points (where you want to extract data)
npoints = 1
xi = np.array([0.0, 0.0, H + 20.0]) # 20m above hill top

# 3. Interpolate U, V, and W
# Note: 'linear' is much faster for 3D than 'cubic'
Ui = griddata(points_3d, data['U'].values.flatten(), xi, method='linear')
Vi = griddata(points_3d, data['V'].values.flatten(), xi, method='linear')
Wi = griddata(points_3d, data['W'].values.flatten(), xi, method='linear')

print(f"Interpolated Velocities at {xi}:")
print(f"U: {Ui}, V: {Vi}, W: {Wi}")