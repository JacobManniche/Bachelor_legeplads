import numpy as np
import math as mt
import os
from py_wake_ellipsys.wind_farm_models.ellipsys_lib import TerrainGrid,WFPostFlow,Cluster,set_cluster_vars,AD,WFRun
from py_wake_ellipsys.utils.terraingridutils import write_box_grd
from py_wake_ellipsys_examples.data.turbines.dummy_wt import Dummy


# Import
#import numpy as np
from py_wake_ellipsys.wind_farm_models.ellipsys import EllipSys
#from py_wake_ellipsys.wind_farm_models.ellipsys_lib.ellipsys_wind_turbines import EllipSysOneTypeWT
#from py_wake_ellipsys.wind_farm_models.ellipsys_lib import *
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
#from py_wake_ellipsys.utils.notebookutils import display_all_pdfs_in_directory, display_all_pngs_in_directory

# these functions are copied from the Gaussian test case
#from py_wake_ellipsys_examples.data.GaussianHill.gaussianhill import create_terrain
def deg2std(height, max_slope):
    return height * np.exp(-0.5) / (mt.tan(max_slope / 180.0 * np.pi))


def gauss2d(x, y, mu_x, mu_y, sigma_x, sigma_y, height):
    return height * np.exp(-(x - mu_x) ** 2 / (2.0 * sigma_x ** 2) - (y - mu_y) ** 2 / (2.0 * sigma_y ** 2))

def save_grd_map(mapoutfile, mapname, x, y, z):
    nx = len(x)
    ny = len(y)
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    zmin = z.min()
    zmax = z.max()
    f = open(mapoutfile, 'w')
    f.write(mapname + '\r\n')
    f.write('%i %i\r\n' % (nx, ny))
    f.write('%g %g\r\n' % (xmin, xmax))
    f.write('%g %g\r\n' % (ymin, ymax))
    f.write('%g %g\r\n' % (zmin, zmax))
    for j in range(ny):
        for i in range(nx):
            f.write('%g %s' % (z[j, i], ''))
        f.write('\n')
    f.close()


def write_gauss2d_grd(x, y, height, mu_x, mu_y, maxangle_x, maxangle_y,filename,mapname):
    # Create a 2D Gaussian grd file
    sigma_x = deg2std(height, maxangle_x)
    sigma_y = deg2std(height, maxangle_y)
    z = np.zeros((len(y), len(x)))
    for j in range(len(y)):
        for i in range(len(x)):
            z[j, i] = gauss2d(x[i], y[j], mu_x, mu_y, sigma_x, sigma_y, height) 

    # Save grd file
    save_grd_map(filename, mapname, x, y, z)


def main():
    if __name__ == '__main__':
        # define Gaussian hill parameters
        H=300.0
        cen_x=0.0;     cen_y= 0.0
        max_angle_x=20.0; max_angle_y=10.0
        print('sigma_x=',deg2std(H, max_angle_x))
        print('sigma_y=',deg2std(H, max_angle_y))
        x=np.arange(-2000.0, 2000.01, 10.0)
        y= np.arange(-4500.0, 4500.01, 10.0)
        print(x[0],x[-1],y[0],y[-1])
        # write grd file for Gaussian hill
        inner='gaussian_hill'
        mapname ='gaussian_hill'
        filename=inner+'.grd'
        write_gauss2d_grd(x, y,H, cen_x, cen_y, max_angle_x, max_angle_y,filename,mapname)

        # make a 2x2 outer grd file with constant height and with dimensions at least 
        # as big as the computational domain, which  will be generated later
        zouter=0.0 # height value far away from hill
        outer='gaussian_hill_outer'
        filename=outer+'.grd'
        write_box_grd(filename, zouter, -1e6, 1e6, -1e7, 1e7)
        # write grd files with roughness values. Their names should be consistent
        # with the names of the grd files defining the surface 
        z0Outer = 0.03
        z0Inner = 0.1
        filename=inner+'_z0.grd'
        write_box_grd(filename, z0Inner, 0.0, 1e6, 0.0, 1e7)
        filename=outer+'_z0.grd'
        write_box_grd(filename, z0Outer, -1e6, 1e6, -1e7, 1e7)
        
        #---------------------------------------------------------------
        # now define a 3D CFD grid using the GRD files as input
        #---------------------------------------------------------------
        # load grd files
        grid_terrain_map_inner = os.getcwd() + '/'+inner+'.grd'
        grid_terrain_map_outer = os.getcwd() + '/'+outer+'.grd'
        # load z0 files
        grid_terrain_z0_inner = os.getcwd() + '/'+inner+'_z0.grd'
        grid_terrain_z0_outer = os.getcwd() + '/'+outer+'_z0.grd'
        # define CFD grid parameters        
        Dref=100 # scaling parameter
        m1_e_D=10.0; m1_w_D=10.0; m1_n_D=10.0; m1_s_D=10.0 #dimensions of inner grid domain in units of Dref (e,w,n,s)
        cells1_D = 4.0 # number of cells per Dref in inner domain
        radius=50000.0 # radius of o-grid around inner domain
        zFirstCell=0.5 # first cell height above ground
        zlen=3000 # height of domain
        x0=0.0; y0=0.0 # origin
        grid = TerrainGrid(Dref, type='terrainogrid', cells1_D=cells1_D, radius_D=radius/Dref,
                             zFirstCell_D=zFirstCell/Dref, bsize=32, zlen_D=zlen/Dref,
                             terrain_h_grds=[grid_terrain_map_inner, grid_terrain_map_outer],
                             terrain_z0_grds=[grid_terrain_z0_inner, grid_terrain_z0_outer],
                             m1_w_D=m1_w_D, m1_e_D=m1_e_D, m1_n_D=m1_n_D, m1_s_D=m1_s_D,
                             terrain_surfgen='sfhill',origin=(x0,y0),run_wd_con=False)


        # Define the simulation
        wt = Dummy()
        wt_x = np.array([0.0]) 
        wt_y = np.array([0.0]) 
        type_i = np.array([0])
        h_i = np.array([H])
        wd = [270.0]
        ws = [8.0]
        TI = 0.1
        zRef = 90.0
        # Set format of output
        wfpostflow = WFPostFlow(outputformat='netCDFmb', single_precision_netCDF=True)
        # Set global cluster settings
        run_machine = 'gbar'
        maxnodes = 1
        corespernode = 24
        queue = 'hpc'
        set_cluster_vars(run_machine, True, queue, corespernode, maxnodes)
        # set up flow model
        flowmodel = EllipSys(Hornsrev1Site(), wt, grid,TI, zRef, ad=AD(force='0000', run_pre=False),
                             wfrun=WFRun(casename='GaussianHill',  write_restart=True),
                             wfpostflow=wfpostflow, run_wd_con=False)


        # Make grid (this may take a while)
        flowmodel.run_grid = True
        flowmodel.run_cal = False # this is only relevant when simulating turbines
        flowmodel.run_wf = True
        flowmodel.run_post = True
        WS_eff_ilk, TI_eff_ilk, power_ilk, *dummy = flowmodel.calc_wt_interaction(wt_x, wt_y, h_i,
                 type_i, wd, ws)

        # Store 3D flow data as a multi-block netCDF file for one flow case
        iwd=0
        iws=0
        flowmodel.post_windfarm_flow(wd[iwd], ws[iws], precursor=False) 

        # Load netCDF file
        import xarray
        folder = flowmodel.get_name(grid_wd=wd[iwd])
        infile = '%s/post_flow_wd%g_ws%g/flowdata_terrain_mb.nc' % (folder, wd[iwd], ws[iws])
        print(infile)
        cluster = Cluster()
        cluster.wait_for_file(infile)
        data = xarray.open_dataset(infile)

        print(data)

        # Python does not have interpolation tools for 3D curvilinear grids, so we 
        # use an EllipSys based tool (InverseMap)
        from pyellipsys.inversemap import InverseMap
        IM = InverseMap()
        # coordinates of points to extract (here only one point)
        npoints=1
        points=np.zeros((npoints,3))
        points[0,:]=np.array([0.0, 0.0, H+20.0]) #20 m above the top of the hill which is at x=y=0  
        # get U, V and W velocity
        Ui = IM.interp(data['x'], data['y'], data['z'], data['U'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False)
        # A new interpolation with the same interpolation points can be used more quickly as
        Vi = IM.interp(data['x'], data['y'], data['z'], data['V'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False, make_inversemap=False, locate_points=False)

        Wi = IM.interp(data['x'], data['y'], data['z'], data['W'], points[:, 0], points[:, 1], points[:, 2], add_ghost_layer=False, make_inversemap=False, locate_points=False)


        print(Ui,Vi,Wi)
main()


