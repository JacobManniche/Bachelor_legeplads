import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tracer import WindField, solver, initial_velocity, initial_spin_rate

# read csv file
file = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/Confidential/shotdata.csv'
with open(file, 'r') as f:
    lines = f.readlines()
    f.close()

data_points = []
for line in lines[1:]:
    ca, bs, ld, la, sa, sr, mh = [float(l) for l in line.replace(',', '.').split('\t')[1:]]
    data_points.append([f"{bs:.0f}_{la:.0f}_{sa:.0f}", bs, la, sr, sa, mh, ca])


columns = [
    "Club",
    "Ball Speed (m/s)",
    "Launch Angle (deg)",
    "Spin Rate (rpm)",
    "Spin Axis (deg)",
    "Max Height (m)",
    "Carry (m)"
]

df_pga = pd.DataFrame(data_points, columns=columns)

directions = [0, 45, 90, 75, 180]#, [*range(0, 360, 45)]
DT = 0.01

def calculate_trajectory_metrics(df, wind_profile):
    """Calculate carry and max height for all clubs in dataframe."""
    carry = []
    max_height = []
    
    for i in range(len(df)):
        total_carry = 0
        total_height = 0
        
        V0 = initial_velocity(speed=df['Ball Speed (m/s)'][i], angle=df['Launch Angle (deg)'][i])
        W0 = initial_spin_rate(spin_rate=df['Spin Rate (rpm)'][i], spin_axis=df['Spin Axis (deg)'][i])
        
        for dir in directions:
            wind = WindField(nx=300, ny=50, nz=50, direction=dir, profile=wind_profile, z0=0.003, U_ref=6)
            p = solver(V0, W0, wind=wind, dt=DT)[1]
            total_carry += ((p[-1][0] - p[0][0])**2 + (p[-1][1] - p[0][1])**2)**0.5
            total_height += max(p[:,2])
        
        num_directions = len(directions)
        carry.append(total_carry / num_directions)
        max_height.append(total_height / num_directions)
        print('.', end='', flush=True)
    
    return carry, max_height

def plot_comparison():
    carry_pga, maxheight_pga = calculate_trajectory_metrics(df_pga, 'log')

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=-1.0)

    # Calculate errors
    pga_carry_error = ((df_pga['Carry (m)'] - carry_pga) / df_pga['Carry (m)']) * 100
    pga_height_error = ((df_pga['Max Height (m)'] - maxheight_pga) / df_pga['Max Height (m)']) * 100

    # Top-left
    axes[0, 0].scatter(df_pga['Club'], df_pga['Carry (m)'], label='PGA data', color='tab:blue', zorder=1)
    axes[0, 0].scatter(df_pga['Club'], carry_pga, marker='+', label='PGA Trajectory', color='tab:green', zorder=2)
    axes[0, 0].set_xticks(df_pga['Club'], "")
    axes[0, 0].set_title("Carry comparison (m)")

    # Top-right

    axes[0, 1].scatter(df_pga['Club'], df_pga['Max Height (m)'], label='PGA data', color='tab:blue', zorder=1)
    axes[0, 1].scatter(df_pga['Club'], maxheight_pga, marker='+', label='PGA Trajectory', color='tab:green', zorder=2)
    axes[0, 1].set_title("Max height (m)")
    axes[0, 1].set_xticks(df_pga['Club'], "")
    axes[0, 1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Bottom-left
    axes[1, 0].scatter(df_pga['Club'], pga_carry_error, label='PGA error', color='tab:blue', marker='x')
    axes[1, 0].axhline(color='darkgray', linestyle='--', linewidth=1.5, zorder=-1.0)
    axes[1, 0].set_title("Carry error percentage")
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].set_xticks(df_pga['Club'])
    axes[1, 0].set_xticklabels(labels=df_pga['Club'], rotation=45)

    # Bottom-right
    axes[1, 1].scatter(df_pga['Club'], pga_height_error, label='PGA error', color='tab:blue', marker='x')
    axes[1, 1].axhline(color='darkgray', linestyle='--', linewidth=1.5, zorder=-1.0)
    axes[1, 1].set_title("Max height error percentage")
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    axes[1, 1].set_xticks(df_pga['Club'])
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison()
