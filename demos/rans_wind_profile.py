import os
import xarray as xr
import matplotlib.pyplot as plt


file = os.path.abspath('/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/RANS/flow_flat_2m_2m.nc')
ds = xr.open_dataset(file)

plt.plot(ds.sel(y=0, x=0).U.values*8, ds.sel(y=0, x=0).z.values)
plt.ylabel('X [m]')
plt.xlabel('U [m/s]')
plt.title('Wind Profile')
plt.show()