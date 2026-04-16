from Tracer import solver, initial_velocity, initial_spin_rate
import Tracer.debug_tools as d
import matplotlib.pyplot as plt

# read csv file
file = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/Confidential/shotdata.csv'
with open(file, 'r') as f:
    lines = f.readlines()
    f.close()

trajectories = []
data_points = []

for line in lines[1:]:
    ca, bs, ld, la, sa, sr, mh = [float(l) for l in line.replace(',', '.').split('\t')[1:]]
    t, p, v, w = solver(initial_velocity(speed=bs, angle=la), initial_spin_rate(spin_rate=sr, spin_axis=sa), wind='log')

    sim_carry = (p[-1][0]**2 + p[-1][1]**2)**.5
    sim_height = max(p[:, 2])
    print(f"Simulated Carry: {sim_carry:.2f} m, Simulated Max Height: {sim_height:.2f} m")
    print(f"Target Carry: {ca:.2f} m, Target Max Height: {mh:.2f} m")
    print('='*30)

    data_points.append({'carry': ca, 'ball_speed': bs, 'launch_angle': la, 'spin_rate': sr, 'max_height': mh, 'sim_carry': sim_carry, 'sim_height': sim_height})
    trajectories.append((t, p, v, w))

ax = d.plot_trajectories(trajectories, plot=False)
for point in data_points:
    ax[2].legend([f'Carry: {point["carry"]:.1f}m, Ball Speed: {point["ball_speed"]:.1f}m/s, Launch Angle: {point["launch_angle"]}°, Spin Rate: {point["spin_rate"]} rpm' for point in data_points[:4]], loc='lower right', fontsize='small', bbox_to_anchor=(1.04, 1))
    ax[0].axhline(point["max_height"], linestyle='--', label='Target Max Height')
    ax[1].axvline(point["carry"], linestyle='--', label='Target carry')

plt.tight_layout()
plt.show()

for i, point in enumerate(data_points):
    plt.scatter(i, (point['sim_carry'] - point['carry']), marker='x', label=f'Carry: {point["carry"]:.1f}m, Ball Speed: {point["ball_speed"]:.1f}m/s, Launch Angle: {point["launch_angle"]}°, Spin Rate: {point["spin_rate"]} rpm')
    plt.scatter(i, (point['sim_height'] - point['max_height']), marker='o', label=f'Carry: {point["carry"]:.1f}m, Ball Speed: {point["ball_speed"]:.1f}m/s, Launch Angle: {point["launch_angle"]}°, Spin Rate: {point["spin_rate"]} rpm')

plt.xlabel('Shot')
plt.ylabel('Difference in carry and max height (m)')
plt.title('Simulated vs Target')
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
plt.tight_layout()
plt.show()