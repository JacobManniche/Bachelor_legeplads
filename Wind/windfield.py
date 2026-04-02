import numpy as np

class WindField:
    def __init__(self, nx, ny, nz, profile="log", U_ref=10.0, z_ref=10.0, z0=0.1):
        """
        nx, ny, nz : int
            Grid size in x, y, z directions
        profile : str
            Type of wind profile ("log" or "uniform")
        U_ref : float
            Reference wind speed (m/s)
        z_ref : float
            Reference height for log profile (m)
        z0 : float
            Surface roughness length (m)
        """

        # Constants
        kappa = 0.4
        u_star = 0.4

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
            U = U_ref * (u_star / kappa) / np.log(z_ref / z0)
        else:
            raise ValueError("Unsupported profile type. Use 'uniform' or 'log'.")
        
        # Assign velocity components
        self.velocity[..., 0] = U   # u-component
        self.velocity[..., 1] = 0   # v-component
        self.velocity[..., 2] = 0   # w-component
        
        # Simple TKE model
        self.tke = 0.1 * (self.velocity[..., 0]**2) * np.exp(-Z / 50)

    def get_point(self, i, j, k):
        """Return velocity vector and TKE at a grid point"""
        return {
            "velocity": self.velocity[i, j, k],
            "tke": self.tke[i, j, k]
        }

    def set_velocity(self, field):
        self.velocity = field

    def set_tke(self, field):
        self.tke = field
        