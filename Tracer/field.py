import numpy as np
import xarray as xr

class Field:
    def __init__(self, ds=None, profile="log", **kwargs):
        """
        ds : xarray.Dataset
            Internal dataset containing coordinates 'x', 'y', 'z' 
            and variables 'U', 'V', 'W', 'tke'.
        profile : str
            Type of wind field to create. Options: 'log', 'uniform', 'rans'.
        kwargs : dict
            Additional parameters for synthetic profiles (direction, U_ref, z0) or RANS dataset
        """
        
        kwargs = {
            "x_range": kwargs.get("x", (-250, 250)),
            "y_range": kwargs.get("y", (-250, 250)),
            "z_range": kwargs.get("z", (0, 100)),
            "res": kwargs.get("res", 10.0),
            "direction": kwargs.get("direction", 270),
            "U_ref": kwargs.get("U_ref", 10.0),
            "z0": kwargs.get("z0", 0.03)
        }

        # make sure profile is lowercase for consistency
        self.profile = profile.lower()
        
        if self.profile in ["log", "uniform"]:
            self.synthesize(**kwargs)
        elif self.profile == "rans" and ds is not None:
            self.from_rans(ds)
        else:
            raise ValueError("Must provide a valid profile type and/or dataset.")

    def synthesize(self, profile="log", x_range=(-250, 250), y_range=(-250, 250), z_range=(0, 100), 
                       res=10.0, direction=270, U_ref=10.0, z0=0.03):
        """Constructor for Log or Uniform profiles."""
        x = np.arange(x_range[0], x_range[1] + res, res)
        y = np.arange(y_range[0], y_range[1] + res, res)
        z = np.arange(z_range[0], z_range[1] + res, res)
        
        # Create 3D mesh for the vertical profile
        # Velocity only changes with Z in synthetic profiles
        z_grid = np.tile(z, (len(x), len(y), 1))
        
        if profile == "uniform":
            mag = np.full_like(z_grid, U_ref)
        elif profile == "log":
            kappa = 0.4
            z_ref = 10.0
            z_safe = np.maximum(z_grid, z0 + 1e-6)
            u_star = U_ref * kappa / np.log(z_ref / z0)
            mag = (u_star / kappa) * np.log(z_safe / z0)
        
        angle = np.radians(direction)

        self.ds = xr.Dataset(
            data_vars={
                'U': (['x', 'y', 'z'], mag * np.cos(angle)),
                'V': (['x', 'y', 'z'], mag * np.sin(angle)),
                'W': (['x', 'y', 'z'], np.zeros_like(mag)),
                'tke': (['x', 'y', 'z'], 0.1 * (mag**2) * np.exp(-z_grid / 50))
            },
            coords={'x': x, 'y': y, 'z': z}
        )

    def from_rans(self, path_or_ds):
        """Constructor for existing Cartesian RANS NetCDF files."""
        if isinstance(path_or_ds, str):
            ds = xr.open_dataset(path_or_ds)
        else:
            ds = path_or_ds
        self.ds = ds

    def get_velocity_at(self, x, y, z):
        """
        Interpolates the wind field at an arbitrary spatial point.
        Returns a dict with velocity vector and tke.
        """
        # .interp handles the logic of finding the nearest 'boxes' and weighting them
        point = self.ds.interp(x=x, y=y, z=z, method='linear')
        
        return np.array([point.U.values, point.V.values, point.W.values])
    
    def get_tke_at(self, x, y, z):
        point = self.ds.interp(x=x, y=y, z=z, method='linear')
        return float(point.tke.values)
    
    def __repr__(self):
        # Simple representation showing profile type and dataset summary
        return f"Field(profile={self.profile}) \n {self.ds})"

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Create a synthetic log profile
    sim_field = Field(profile="log", direction=270)
    
    # 2. Get data at an arbitrary coordinate (not necessarily on the grid)
    data = sim_field.get_velocity_at(x=10.5, y=-100.2, z=150.0)
    print(f"Synthetic U: {data[0]:.2f} m/s")

    # 3. Load from your Cartesian NetCDF
    # rans_wind = WindField.from_rans('flow_gaussian_cartesian.nc')
    # data_rans = rans_wind.get_velocity_at(x=0, y=0, z=320)