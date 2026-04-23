# Demo for trajectory model using rans field

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Tracer.windfield import WindField
from Tracer.tracer import solver, initial_velocity, fetch_wind_data, initial_spin_rate
from Tracer.RK45 import solver_rk45
# play field parameters - dont touch unless necessary
P0=np.array([-200, 250, 1])
nx = 500        # length of play field
ny = 200        # width of play field
nz = 100        # height of play field
dt = 0.01       # time step

# Basis shot parameters - if not otherwise specified, these are used
shot_speed = 76.44384   # intital velocity
shot_angle = 10.4       # initial angle for trajectory	
shot_spin = 2545        # initial spin
spin_axis = 1.25

# initialize the rans field
infile = r'RANS/flow_gaussian_cartesian.nc'
dataset = xr.open_dataset(infile)

# 
infile = r'RANS/flowdata_terrain_mb.nc'
dataset_raw = xr.open_dataset(infile)

# trajectory with log wind
rans_wind = WindField(nx=nx, ny=ny, nz=nz, profile='rans',dataset=dataset)
rans_V0 = initial_velocity(speed=shot_speed, angle=shot_angle)
rans_W0 = initial_spin_rate(spin_rate=shot_spin, spin_axis=spin_axis)
rans_t, rans_p, rans_v, rans_w = solver(rans_V0, rans_W0, P0=P0, wind=rans_wind, dt=dt)

print(rans_p[-2])

