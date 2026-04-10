import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tracer import WindField, solver, initial_velocity

# Data from tour averages for the men's (PGA) and women's (LPGA) golf turnaments
# from https://www.trackman.com/blog/golf/introducing-updated-tour-averages

pga_data = [
    ["Driver", 115, -0.9, 171, 1.49, 10.4, 2545, 32, 39, 258],
    ["3-wood", 110, -2.3, 162, 1.47, 9.3, 3663, 29, 44, 228],
    ["5-wood", 106, -2.5, 156, 1.47, 9.7, 4322, 30, 48, 216],
    ["Hybrid", 102, -2.4, 149, 1.47, 10.2, 4587, 28, 49, 211],
    ["3 Iron", 100, -2.5, 145, 1.46, 10.3, 4404, 27, 48, 199],
    ["4 Iron", 98, -2.9, 140, 1.44, 10.8, 4782, 28, 49, 192],
    ["5 Iron", 96, -3.4, 135, 1.41, 11.9, 5280, 30, 50, 182],
    ["6 Iron", 94, -3.7, 130, 1.39, 14.0, 6204, 29, 50, 172],
    ["7 Iron", 92, -3.9, 123, 1.34, 16.1, 7124, 31, 51, 161],
    ["8 Iron", 89, -4.2, 118, 1.33, 17.8, 8078, 30, 51, 150],
    ["9 Iron", 87, -4.3, 112, 1.29, 20.0, 8793, 29, 52, 139],
    ["PW", 84, -4.7, 104, 1.24, 23.7, 9316, 29, 52, 130],
]

lpga_data = [
    ["Driver", 96, 2.8, 143, 1.49, 12.6, 2506, 24, 36, 204],
    ["3-wood", 92, -0.8, 135, 1.47, 11.6, 2595, 23, 38, 183],
    ["5-wood", 90, -1.6, 130, 1.46, 12.3, 4320, 23, 43, 173],
    ["Hybrid", 87, -1.9, 125, 1.44, 13.9, 4504, 23, 45, 163],
    ["4 Iron", 82, -1.7, 118, 1.43, 13.9, 4608, 23, 43, 160],
    ["5 Iron", 81, -2.0, 114, 1.42, 14.6, 4966, 23, 45, 152],
    ["6 Iron", 80, -2.3, 111, 1.41, 16.7, 5904, 23, 46, 142],
    ["7 Iron", 78, -2.5, 106, 1.38, 18.5, 6630, 24, 47, 131],
    ["8 Iron", 76, -2.8, 102, 1.36, 20.8, 7413, 25, 47, 122],
    ["9 Iron", 74, -3.2, 95, 1.30, 23.5, 7605, 25, 48, 112],
    ["PW", 72, -3.2, 88, 1.25, 25.2, 8465, 25, 48, 101],
]

columns = [
    "Club",
    "Club Speed (mph)",
    "Attack Angle (deg)",
    "Ball Speed (mph)",
    "Smash Factor",
    "Launch Angle (deg)",
    "Spin Rate (rpm)",
    "Max Height (m)",
    "Land Angle (deg)",
    "Carry (m)"
]

df_pga = pd.DataFrame(pga_data, columns=columns)
df_lpga = pd.DataFrame(lpga_data, columns=columns)


# convert from mph to m/s
df_pga['Ball Speed (mph)'] = df_pga['Ball Speed (mph)'] * 0.44704
df_pga = df_pga.rename(columns={'Ball Speed (mph)':'Ball Speed (m/s)'})
df_pga['Club Speed (mph)'] = df_pga['Club Speed (mph)'] * 0.44704
df_pga = df_pga.rename(columns={'Club Speed (mph)':'Club Speed (m/s)'})

df_lpga['Ball Speed (mph)'] = df_lpga['Ball Speed (mph)'] * 0.44704
df_lpga = df_lpga.rename(columns={'Ball Speed (mph)':'Ball Speed (m/s)'})
df_lpga['Club Speed (mph)'] = df_lpga['Club Speed (mph)'] * 0.44704
df_lpga = df_lpga.rename(columns={'Club Speed (mph)':'Club Speed (m/s)'})

P0 = [0,0,0]
directions = [0, 45, 135, 180] #[*range(0, 360, 45)]
DT = 0.01

def calculate_trajectory_metrics(df, wind_profile):
    """Calculate carry and max height for all clubs in dataframe."""
    carry = []
    max_height = []
    
    for i in range(len(df)):
        total_carry = 0
        total_height = 0
        
        V0 = initial_velocity(speed=df['Ball Speed (m/s)'][i], angle=df['Launch Angle (deg)'][i])
        W0 = np.array([0, -df['Spin Rate (rpm)'][i], 0])
        
        for dir in directions:
            wind = WindField(nx=300, ny=50, nz=50, direction=dir, profile=wind_profile, z0=0.003, U_ref=0)
            t, p, v, w = solver(P0, V0, W0, wind, dt=DT)
            total_carry += ((p[-1][0] - p[0][0])**2 + (p[-1][1] - p[0][1])**2)**0.5
            total_height += max(p[:,2])
        
        num_directions = len(directions)
        carry.append(total_carry / num_directions)
        max_height.append(total_height / num_directions)
    
    return carry, max_height

# %%
carry_pga, maxheight_pga = calculate_trajectory_metrics(df_pga, 'log')
carry_lpga, maxheight_lpga = calculate_trajectory_metrics(df_lpga, 'uniform')

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# Calculate errors
pga_carry_error = np.abs((df_pga['Carry (m)'] - carry_pga) / df_pga['Carry (m)']) * 100
lpga_carry_error = np.abs((df_lpga['Carry (m)'] - carry_lpga) / df_lpga['Carry (m)']) * 100
pga_height_error = np.abs((df_pga['Max Height (m)'] - maxheight_pga) / df_pga['Max Height (m)']) * 100
lpga_height_error = np.abs((df_lpga['Max Height (m)'] - maxheight_lpga) / df_lpga['Max Height (m)']) * 100

# Top-left
axes[0, 0].scatter(df_pga['Club'], df_pga['Carry (m)'], label='PGA data')
axes[0, 0].scatter(df_pga['Club'], carry_pga, marker='x', label='PGA Trajectory')
axes[0, 0].scatter(df_lpga['Club'], df_lpga['Carry (m)'], label='LPGA data')
axes[0, 0].scatter(df_lpga['Club'], carry_lpga, marker='x', label='LPGA Trajectory')
axes[0, 0].set_title("Carry comparison")
axes[0, 0].legend()

# Top-right
axes[0, 1].scatter(df_pga['Club'], df_pga['Max Height (m)'], label='PGA data')
axes[0, 1].scatter(df_pga['Club'], maxheight_pga, marker='x', label='PGA Trajectory')
axes[0, 1].scatter(df_lpga['Club'], df_lpga['Max Height (m)'], label='LPGA data')
axes[0, 1].scatter(df_lpga['Club'], maxheight_lpga, marker='x', label='LPGA Trajectory')
axes[0, 1].set_title("Max height")
axes[0, 1].legend()

# Bottom-left
axes[1, 0].scatter(df_pga['Club'], pga_carry_error, label='PGA error')
axes[1, 0].scatter(df_lpga['Club'], lpga_carry_error, label='LPGA error')
axes[1, 0].set_title("Carry error (absolute)")
axes[1, 0].set_ylabel('Error (%)')
axes[1, 0].legend()

# Bottom-right
axes[1, 1].scatter(df_pga['Club'], pga_height_error, label='PGA error')
axes[1, 1].scatter(df_lpga['Club'], lpga_height_error, label='LPGA error')
axes[1, 1].set_title("Max height error (absolute)")
axes[1, 1].set_ylabel('Error (%)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
