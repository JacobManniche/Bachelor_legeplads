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
    zlen = 3000 # height of domain
    zFirstCell = 0.5 # first cell height above ground

    # Dimensions of inner grid domain in units of Dref (e,w,n,s)
    m1_e_D = 10.0; m1_w_D = 10.0; m1_n_D = 10.0; m1_s_D = 10.0
    cells1_D = 4.0 # number of cells per Dref in inner domain

    # Use FlatBoxGrid for a completely flat terrain case (no .grd files needed)
    grid = FlatBoxGrid(Dref, 
                            cells1_D = cells1_D,
                            zFirstCell_D = zFirstCell / Dref, 
                            bsize = 32,
                            zlen_D = zlen / Dref,
                            m1_w_D = m1_w_D, 
                            m1_e_D = m1_e_D, 
                            m1_n_D = m1_n_D,
                            m1_s_D = m1_s_D)
        
    # ---------------------------------------------------------------
    # 2. Define Simulation Parameters
    # ---------------------------------------------------------------
    wt = Dummy()
    wt_x = np.array([0.0]) 
    wt_y = np.array([0.0]) 
    type_i = np.array([0])

    hub_height = 90.0
    h_i = np.array([hub_height])

    wd = [270.0]
    ws = [8.0]
    TI = 0.1
    zRef = 90.0

    # ---------------------------------------------------------------
    # 3. Setup Cluster and Flow Model
    # ---------------------------------------------------------------
    run_machine = 'gbar'
    set_cluster_vars(run_machine, True, 'hpc', corespernode=24, maxnodes=1)

    wfpostflow = WFPostFlow(outputformat='netCDFmb', single_precision_netCDF=True)

    flowmodel = EllipSys(Hornsrev1Site(), wt, grid,TI, zRef, ad=AD(force='0000', run_pre=False),
                                wfrun=WFRun(casename='FlatTerrain',  write_restart=True),
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

    # ---------------------------------------------------------------
    # 5. Extract and Interpolate Data
    # ---------------------------------------------------------------
    folder = flowmodel.get_name(grid_wd=wd[iwd])
    infile = '%s/post_flow_wd%g_ws%g/flowdata_mb.nc' % (folder, wd[iwd], ws[iws])

    print(f"Waiting for output file: {infile}")
    cluster = Cluster()
    cluster.wait_for_file(infile)
    data = xarray.open_dataset(infile)

    IM = InverseMap()
    # Extract data at hub height at the origin
    points = np.array([[0.0, 0.0, hub_height]]) # shape (1,3)

    Ui = IM.interp(data['x'], data['y'], data['z'], data['U'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False)
    Vi = IM.interp(data['x'], data['y'], data['z'], data['V'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False, make_inversemap=False, locate_points=False) 
    Wi = IM.interp(data['x'], data['y'], data['z'], data['W'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False, make_inversemap=False, locate_points=False)

    print("\nInterpolated velocities at hub height (U, V, W):")
    print(f"U: {Ui[0]:.2f} m/s, V: {Vi[0]:.2f} m/s, W: {Wi[0]:.2f} m/s")

if __name__ == '__main__':
    main()

