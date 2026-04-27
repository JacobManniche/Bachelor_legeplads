import os
import xarray as xr
import matplotlib.pyplot as plt
from Tracer.field import Field


file = os.path.abspath('/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/RANS/nc files/flow_flat_2m_2m.nc')
ds = xr.open_dataset(file)

# synthetic log profile
wind_synt = Field(profile='log', direction=0, U_ref=8, z_ref=90, z0=0.03)

# RANS profile
wind_rans = Field(ds=ds, profile='rans', scale_factor=8)

plt.plot(wind_synt.ds.sel(y=0, x=0).U.values, wind_synt.ds.sel(y=0, x=0).z.values)
plt.plot(wind_rans.ds.sel(x=0, y=0).U.values, wind_rans.ds.sel(x=0, y=0).z.values)

plt.ylabel('Z [m]')
plt.xlabel('U [m/s]')
plt.title('Wind Profile')
plt.show()