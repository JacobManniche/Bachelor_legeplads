import numpy as np
import math as mt
import os
import xarray

# PyWake / EllipSys imports
from py_wake_ellipsys.wind_farm_models.ellipsys import EllipSys
from py_wake_ellipsys.wind_farm_models.ellipsys_lib import (
    FlatBoxGrid,
    WFPostFlow,
    set_cluster_vars,
    AD,
    WFRun,
    Cluster
)

from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake_ellipsys_examples.data.turbines.dummy_wt import Dummy
from pyellipsys.inversemap import InverseMap


def main():
    # ---------------------------------------------------------------
    # 1. Define CFD Grid Parameters (Flat Terrain)
    # ---------------------------------------------------------------
    Dref = 100 # scaling parameter
    zlen = 500 # height of domain
    zFirstCell = 0.025    # first cell height above ground

    # Dimensions of inner grid domain in units of Dref (e,w,n,s)
    m1_e_D = 2.56; m1_w_D = 2.56; m1_n_D = 2.56; m1_s_D = 2.56
    cells1_D = 25.0 # number of cells per Dref in inner domain

    # Use FlatBoxGrid for a completely flat terrain case (no .grd files needed)
    grid = FlatBoxGrid(Dref, 
                            cells1_D = cells1_D,
                            zFirstCell_D = zFirstCell / Dref, 
                            z_cells1_D = 100,                   # number of cells in vertical direction in inner domain (default: cells1_D = z_cells1_D)
                            zWakeEnd_D = 3.0,
                            bsize = 32,
                            zlen_D = zlen / Dref,
                            radius_D = 50,
                            m1_w_D = m1_w_D, 
                            m1_e_D = m1_e_D, 
                            m1_n_D = m1_n_D,
                            m1_s_D = m1_s_D, cluster = Cluster(gbar_mem = 6, walltime = '4:00:00'))
        
    # ---------------------------------------------------------------
    # 2. Define Simulation Parameters
    # ---------------------------------------------------------------
    wt = Dummy()
    wt_x = np.array([0.0]) 
    wt_y = np.array([0.0]) 
    type_i = np.array([0])          

    hub_height = 90.0               # Hub height in meters
    h_i = np.array([hub_height])

    wd = [270.0]                    # Wind direction in degrees (meteorological convention: 0 = from North, 90 = from East, etc.)
    ws = [6.0]                      # Wind speed in m/s at reference height (zRef)
    TI = 0.1                        # Turbulence intensity at reference height (zRef)
    zRef = 10.0                     # Reference height for wind speed and turbulence intensity (m)

    # ---------------------------------------------------------------
    # 3. Setup Cluster and Flow Model
    # ---------------------------------------------------------------
    run_machine = 'gbar'
    set_cluster_vars(run_machine, True, 'hpc', corespernode=32, maxnodes=3)

    wfpostflow = WFPostFlow(outputformat='netCDFmb', single_precision_netCDF=True, cluster = Cluster(gbar_mem = 6, walltime = '6:00:00'))

    flowmodel = EllipSys(Hornsrev1Site(), wt, grid,TI, zRef, ad=AD(force='0000', run_pre=False),
                                wfrun=WFRun(casename='Flat_z0025m_1m', cluster = Cluster(walltime = '56:00:00'), write_restart=True),
                                wfpostflow=wfpostflow, run_wd_con=False)
        

    # ---------------------------------------------------------------
    # 4. Run Simulation
    # ---------------------------------------------------------------
    flowmodel.run_grid = True
    flowmodel.run_cal = False # this is only relevant when simulating turbines
    flowmodel.run_wf = True
    flowmodel.run_post = True

    WS_eff_ilk, TI_eff_ilk, power_ilk, *dummy = flowmodel.calc_wt_interaction(wt_x, wt_y, h_i,
                    type_i, wd, ws)

    # Store 3D flow data
    iwd, iws = 0, 0

    flowmodel.post_windfarm_flow(wd[iwd], ws[iws], precursor=False)

if __name__ == '__main__':
    main()
