import numpy as np

class WindField:
    def __init__(self, nx=500, ny=500, nz=100, direction=0, profile="log", U_ref=10.0, z0=0.03, dataset=None):
        """
        Synthetic simulation params:

        nx, ny, nz : int
            Grid size in x, y, z directions
        direction : float
            Wind direction in degrees (0 is from the south to north)
        profile : str
            Type of wind profile ("log" or "uniform")
        U_ref : float
            Reference wind speed (m/s) at 10 meters above ground
        z_ref : float
            Reference height for log profile (m)
        z0 : float
            Surface roughness length (m)
        
        RANS simulation params:

        dataset : xarray.Dataset
            Must contain variables U, V, W, x, y, z, tke
        """
        self.profile = profile

        if profile =="rans":
            
            self.ds = dataset
            self.nx = dataset.sizes["x"]
            self.ny = dataset.sizes["y"]
            self.nz = dataset.sizes["z"]

            # Store coordinates (important!)
            self.x_coords = dataset["x"].values
            self.y_coords = dataset["y"].values
            self.z_coords = dataset["z"].values

            # WHEN TKE IS EXTRACTED FROM THE NC FILES, IT SHOULD BE ADDED HERE
            self.tke = 0

        if profile in ["log", "uniform"]:  

            # Constants
            kappa = 0.4 # Von Karman constant
            zRef = 10   # reference height for wind speed measurement (usually 10 meters)

            self.nx = nx
            self.ny = ny
            self.nz = nz
            
            # Initialize arrays
            self.velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)  # u, v, w
            self.tke = np.zeros((nx, ny, nz), dtype=np.float32)
            
            # Coordinates
            x = np.arange(nx)
            y = np.arange(ny)
            z = np.arange(nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Initialize velocity based on profile
            if profile == "uniform":
                U = np.full((nx, ny, nz), U_ref, dtype=np.float32)
            elif profile == "log":
                # Avoid log(0) by ensuring z > z0
                Z_safe = np.maximum(Z, z0 + 1e-6)
                u_star= U_ref*(kappa) / (np.log(zRef/z0))
                U = (u_star / kappa) * np.log(Z_safe / z0)
            else:
                raise ValueError("Unsupported profile type. Use 'uniform' or 'log'.")
        
            # Assign velocity components

            angle = np.radians(direction)  # Convert direction to radians

            self.velocity[..., 0] = U * np.cos(angle)   # u-component
            self.velocity[..., 1] = U * np.sin(angle)   # v-component
            self.velocity[..., 2] = 0   # w-component
            
            # Simple TKE model
            self.tke = 0.1 * (self.velocity[..., 0]**2) * np.exp(-Z / 50)

    def get_point(self, i, j, k):
        """
        Return velocity vector and TKE at grid index (i, j, k)
        (same interface as WindField)
        """

        if self.profile == 'rans':

            x = self.x_coords[i]
            y = self.y_coords[j]
            z = self.z_coords[k]

            ds_point = self.ds.sel(x=x, y=y, z=z, method="nearest")

            U = ds_point["U"].item()
            V = ds_point["V"].item()
            W = ds_point["W"].item()

            velocity = np.array([U, V, W], dtype=np.float32)

            # Placeholder TKE
            tke = 0.5 * (U**2 + V**2 + W**2) * 0.1

            return {
                "velocity": velocity,
                "tke": tke
            }
        
        if self.profile in ["log", "uniform"]:
        
            return {
                "velocity": self.velocity[i, j, k],
                "tke": self.tke[i, j, k]
            }
        else:
            raise ValueError("Profile issue.")

    def set_velocity(self, field):
        self.velocity = field

    def set_tke(self, field):
        self.tke = field