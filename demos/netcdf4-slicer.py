
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_wind_field(file):
	# 1. Load your Cartesian NetCDF (created in the previous steps)
	# If you haven't made it yet, use your existing 'ds' from the RANS output.
	ds = xr.open_dataset(file)

	# Extract the components.
	# Assuming dimensions are (x, y, z).
	U = ds.U.values * 8
	V = ds.V.values * 8
	W = ds.W.values * 8
	T = (U**2 + V**2 + W**2)**0.5  # Total velocity magnitude
	x_vec = ds.x.values
	y_vec = ds.y.values
	z_vec = ds.z.values

	# 2. Setup the figure.
	fig, ax = plt.subplots(2,2, figsize=(10, 8))
	plt.subplots_adjust(bottom=0.2)

	# Initial slice index (middle of the height).
	initial_idx = len(z_vec) // 2

	# 3. Create the initial plot.
	X, Y = np.meshgrid(x_vec, y_vec)
	meshU = ax[0, 0].pcolormesh(X, Y, U[:, :, initial_idx].T, shading='auto', cmap='viridis')
	meshV = ax[0, 1].pcolormesh(X, Y, V[:, :, initial_idx].T, shading='auto', cmap='viridis')
	meshW = ax[1, 0].pcolormesh(X, Y, W[:, :, initial_idx].T, shading='auto', cmap='viridis')
	meshT = ax[1, 1].pcolormesh(X, Y, T[:, :, initial_idx].T, shading='auto', cmap='viridis')

	# Add formatting.
	plt.colorbar(meshU, ax=ax[0, 0], label='Velocity $U$ [m/s]')
	plt.colorbar(meshV, ax=ax[0, 1], label='Velocity $V$ [m/s]')
	plt.colorbar(meshW, ax=ax[1, 0], label='Velocity $W$ [m/s]')
	plt.colorbar(meshT, ax=ax[1, 1], label='Total Velocity $T$ [m/s]')

	title = ax[0, 0].set_title(f'Wind Velocity at Height: {z_vec[initial_idx]:.2f} m')
	ax[0, 0].set_xlabel('X [m]')
	ax[0, 0].set_ylabel('Y [m]')
	ax[0, 0].set_aspect('equal')
	ax[0, 1].set_xlabel('X [m]')
	ax[0, 1].set_ylabel('Y [m]')
	ax[0, 1].set_aspect('equal')
	ax[1, 0].set_xlabel('X [m]')
	ax[1, 0].set_ylabel('Y [m]')
	ax[1, 0].set_aspect('equal')
	ax[1, 1].set_xlabel('X [m]')
	ax[1, 1].set_ylabel('Y [m]')
	ax[1, 1].set_aspect('equal')

	# 4. Add a slider to move through z-slices.
	slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
	slice_slider = Slider(
		ax=slider_ax,
		label='z slice',
		valmin=0,
		valmax=len(z_vec) - 1,
		valinit=initial_idx,
		valstep=1,
	)


	def update_slice(val):
		slice_idx = int(slice_slider.val)
		meshU.set_array(U[:, :, slice_idx].T.ravel())
		meshV.set_array(V[:, :, slice_idx].T.ravel())
		meshW.set_array(W[:, :, slice_idx].T.ravel())
		meshT.set_array(T[:, :, slice_idx].T.ravel())
		title.set_text(f'Wind Velocity at Height: {z_vec[slice_idx]:.2f} m')
		fig.canvas.draw_idle()


	slice_slider.on_changed(update_slice)
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	file = 'temp/flow_gaussian_cartesian.nc'
	visualize_wind_field(file)