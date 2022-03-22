from sys import getsizeof
from time import time
import numpy as np
import torch as tc
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from helicalc import helicalc_data
from .integrate import *
from .constants import *
from .tools import *


## STRAIGHT BARS (Longitudinal, Tangential)
class StraightIntegrator3D(object):
    def __init__(self, geom_df, dxyz, int_func=tc.trapz, lib=tc, dev=0):
        # library setup
        self.lib = lib
        self.int_func = int_func
        self.dev = dev
        # GPU setup
        self.start_dev_mem = get_gpu_memory_map()[dev]
        self.mem_err_expected = False
        if lib is tc:
            tc.cuda.set_device(dev)
        # store relevant parameters
        self.geom_df = geom_df
        self.L = geom_df.length
        self.I = geom_df.I
        self.I_flow = geom_df.I_flow
        self.W = geom_df.W
        self.T = geom_df['T']
        self.j = self.I / (self.W*self.T)
        self.mu_fac = mu_0 * self.j / (4*np.pi)
        self.dxyz = dxyz
        # local origin -- depends on "I_flow"
        if np.isclose(self.I_flow, 0.):
            self.xc = geom_df.x0
            self.yc = geom_df.y0
            self.zc = geom_df.z0
        else:
            self.xc = geom_df.x1
            self.yc = geom_df.y1
            self.zc = geom_df.z0
        # integration steps
        # use correct library
        if lib is tc:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1))).cuda()
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1))).cuda()
            self.zps = lib.linspace(0, self.L, abs(int(self.L/self.dxyz[2] + 1))).cuda()
        else:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1)))
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1)))
            self.zps = lib.linspace(0, self.L, abs(int(self.L/self.dxyz[2] + 1)))
        self.XP, self.YP, self.ZP = lib.meshgrid(self.xps, self.yps, self.zps, indexing='ij')
        # rotation
        self.euler2 = geom_df[['Phi2', 'theta2', 'psi2']].values
        self.rot = Rotation.from_euler('zyz', self.euler2[::-1], degrees=True)
        self.inv_rot = self.rot.inv()

    def integrate(self, x0, y0, z0):
        x0, y0, z0 = self.rot.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        RX = rx_str(x0, self.XP)
        RY = ry_str(y0, self.YP)
        RZ = rz_str(z0, self.ZP)
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        result = []
        for integrand_func in [straight_integrand_Bx, straight_integrand_By]:
            integrand_xyz = self.mu_fac * integrand_func(RX, RY, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.zps, integrand_xyz, self.int_func).item())
        B_vec = np.array(result+[0.])
        # rotate vector back to mu2e coordinates
        B_vec_rot = self.inv_rot.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        return self.last_result

    def integrate_vec(self, x0_vec, y0_vec, z0_vec):
        x0_vec, y0_vec, z0_vec = self.rot.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        RX = rx_str(x0_vec[:,None,None,None], self.XP[None,...])
        RY = ry_str(y0_vec[:,None,None,None], self.YP[None,...])
        RZ = rz_str(z0_vec[:,None,None,None], self.ZP[None,...])
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        result = []
        for integrand_func in [straight_integrand_Bx, straight_integrand_By]:
            integrand_xyz = self.mu_fac * integrand_func(RX, RY, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.zps, integrand_xyz, self.int_func))
        if self.lib is tc:
            B_vecs = tc.stack(result+[tc.zeros_like(x0_vec)]).cpu()
        else:
            B_vecs = np.array(result+[np.zeros_like(x0_vec)])
        # rotate vector back to mu2e coordinates
        B_vecs_rot = self.inv_rot.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        return self.last_result

    def integrate_grid(self, df, N_batch=10000, tqdm=tqdm):
        # initial time
        t0 = time()
        # print info about bus bar
        i = int(round(self.geom_df['cond N']))
        print(f'Straight Bus Bar {i}: grid with {len(df):E} points')
        # add dataframe to object
        self.df = df.copy()
        # calculate number of chunks
        N_per_chunk = N_batch
        N_chunks = int(len(df)/N_per_chunk) + 1
        # let user know chunk size
        print(f'Chunk size: {N_per_chunk}, Number of chunks: {N_chunks}')
        # generate padded arrays corresponding to chunk size and number of chunks
        vals_list = []
        for col in ['X', 'Y', 'Z']:
            vals = np.zeros(N_per_chunk * N_chunks)
            vals[:len(self.df)] = self.df[col]
            vals = vals.reshape((N_chunks, N_per_chunk))
            vals_list.append(vals)
        # loop through chunks and save results
        Bxs = []
        Bys = []
        Bzs = []
        for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])):
            Bx_, By_, Bz_ = self.integrate_vec(x_, y_, z_)
            Bxs.append(Bx_)
            Bys.append(By_)
            Bzs.append(Bz_)

        Bxs = np.array(Bxs).flatten()[:len(self.df)]
        Bys = np.array(Bys).flatten()[:len(self.df)]
        Bzs = np.array(Bzs).flatten()[:len(self.df)]
        xs = vals_list[0].flatten()[:len(self.df)]
        ys = vals_list[1].flatten()[:len(self.df)]
        zs = vals_list[2].flatten()[:len(self.df)]

        self.df.loc[:, f'Bx_bus_str_cn_{i}'] = Bxs
        self.df.loc[:, f'By_bus_str_cn_{i}'] = Bys
        self.df.loc[:, f'Bz_bus_str_cn_{i}'] = Bzs

        # final time, report total time
        tf = time()
        print(f'Calculation time: {(tf - t0):0.2f} s\n')

        return self.df

    def save_grid_calc(self, savetype='pkl', savename=f'Bmaps/helicalc_partial/Mau13.DS_region.standard-busbar.cond_N_57_straight_TEST', all_cols=False):
        # determine which columns to save
        i = int(round(self.geom_df['cond N']))
        cols = ['X', 'Y', 'Z']
        for col in self.df.columns:
            if all_cols:
                if 'bus_str' in col:
                    cols.append(col)
            else:
                if f'bus_str_cn_{i}' in col:
                    cols.append(col)
        # save
        df_to_save = self.df[cols]
        if savetype == 'pkl':
            df_to_save.to_pickle(f'{helicalc_data}{savename}.{savetype}')
        else:
            raise NotImplementedError('Allowed savetype: ["pkl"]')


## CIRCLE ARC BARS
class CircleIntegrator(object):
    def __init__(self, geom_coil, dxyz, layer=1, int_func=tc.trapz, lib=tc, dev=0):
        # if layer > geom_coil.N_layers:
        #     raise ValueError(f'Layer "{layer}" invalid. Please select layer in the range [1, {int(geom_coil.N_layers)}]')
        self.start_dev_mem = get_gpu_memory_map()[dev]
        self.mem_err_expected = False
        # x, y, z --> rho, zeta, phi
        tc.cuda.set_device(dev)
        # rho0 = geom_coil.rho0_a + (layer-1)*(geom_coil.h_cable + 2*geom_coil.t_ci + geom_coil.t_il)
        rho0 = geom_coil.rho0_a
        rho1 = rho0 + geom_coil.h_sc
        # rho_lims = [geom_coil.rho0_a, geom_coil.rho1_a]
        rho_lims = [rho0, rho1]
        # helicity
        self.helicity = int(geom_coil.helicity * (-1)**(layer-1))
        # phi lims
        if layer == geom_coil.N_layers:
            # in last layer, wind until phi1 reached
            if geom_coil.phi1 < geom_coil.phi0:
                self.last_turn_rad = 2*np.pi - geom_coil.phi0 + geom_coil.phi1
            elif geom_coil.phi1 > geom_coil.phi0:
                self.last_turn_rad = geom_coil.phi1 - geom_coil.phi0
            else:
                self.last_turn_rad = 2*np.pi
            if self.helicity == 1:
                self.phi_i = geom_coil.phi0
                self.phi_f = geom_coil.phi0 + 2*np.pi*(geom_coil.N_turns-1) + self.last_turn_rad
            else:
                self.phi_i = geom_coil.phi0 + (2*np.pi - self.last_turn_rad)
                self.phi_f = geom_coil.phi0 + 2*np.pi*geom_coil.N_turns
        else:
            self.phi_i = geom_coil.phi0
            self.phi_f = geom_coil.phi0 + 2*np.pi*geom_coil.N_turns
        zeta_lims = [geom_coil.zeta0, geom_coil.zeta1]
        # phi_lims = [geom_coil.phi_i, geom_coil.phi_f]
        phi_lims = [self.phi_i, self.phi_f]
        xyz_lims = [rho_lims, zeta_lims, phi_lims]
        self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
        self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
        self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
        # estimate memory
        per_el_rh = 1.1e-5
        per_el_z = 2.27e-5
        per_el_ph = 4e-6
        lr = len(self.rhos)
        lz = len(self.zetas)
        lp = len(self.phis)
        # try 1
        # self.est_mem_init_mb = (2*(lr + lz + lp) + 2*(lr*lz*lp))*per_el
        # self.est_mem_run_mb = (2*(lr + lz + lp) + 7*(lr*lz*lp))*per_el
        # try 2
        # self.est_mem_init_mb = (2*(lr*per_el_rh + lz*per_el_z + lp*per_el_ph) + 2*(lr*lz*lp)*per_el_ph)
        # self.est_mem_run_mb = (2*(lr*per_el_rh + lz*per_el_z + lp*per_el_ph) + 7*(lr*lz*lp)*per_el_ph)
        # end init if est. run memory too large (give a message)
        # try 3
        # per_el = 2.9e-6
        per_el = 7e-6
        N_el_init = 2*(lr + lz + lp) + 2*lr*lz*lp
        N_el_run = 2*(lr + lz + lp) + 7*lr*lz*lp
        # N_el_run = lr + lz + lp + 10*lr*lz*lp
        # N_el_run = lr + lz + lp + 12*lr*lz*lp
        self.est_mem_init_mb = N_el_init * per_el
        self.est_mem_run_mb = N_el_run * per_el
        mult = 0.95
        if self.est_mem_run_mb > mult*MAXMEM:
            print(f'Estimated Run Memory: {self.est_mem_run_mb*1e-3} GB >',
                  f'{mult}*{MAXMEM} GB = {mult*MAXMEM} GB')
            self.mem_err_expected = True
        # else:
        self.RHO, self.ZETA, self.PHI = lib.meshgrid(self.rhos,self.zetas,self.phis)
        self.SINPHI = lib.sin(self.PHI)
        self.COSPHI = lib.cos(self.PHI)
        self.actual_mem_init_mb = get_gpu_memory_map()[dev] - self.start_dev_mem
        self.geom_coil = geom_coil
        self.int_func = int_func
        self.layer = layer
        self.lib = lib
        self.dev = dev
        self.XYZ_rot = geom_coil[[f'rot{i:d}' for i in [0,1,2]]].values
        self.XYZ_rot_rad = np.radians(self.XYZ_rot)
        self.mu2e_to_coil = Rotation.from_euler('XYZ', -self.XYZ_rot_rad)
        self.coil_to_mu2e = self.mu2e_to_coil.inv()
        self.xc, self.yc, self.zc = geom_coil[['x','y','z']].values
        # check sizes
        sizes = []
        for o in [self.rhos, self.zetas, self.phis, self.RHO, self.ZETA, self.PHI,
                  self.SINPHI, self.COSPHI]:
            sizes.append(getsizeof(o.storage())*1e-6)
        self.getsizeof_init_mb = np.array(sizes)

    #def integrate(self, x0=0, y0=0, z0=0, index=None):
    #    if index is not None:
    #        x0,y0,z0 = self.field_points[index]
    #    # rotate based on geom
    #    #x0, y0, z0 = transform_field_point(x0,y0,z0)
    #    else:
    #        # rotate/translate
    #        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
    def integrate(self, x0=0, y0=0, z0=0):
        # rotate based on geom
        #x0, y0, z0 = transform_field_point(x0,y0,z0)
        # rotate/translate
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # calculate repeated calculations
        RX = rx(self.RHO, self.COSPHI, x0)
        RY = ry(self.RHO, self.SINPHI, self.helicity, y0)
        RZ = rz(self.ZETA, self.PHI, self.geom_coil.phi0, self.geom_coil.pitch_bar, self.geom_coil.L, z0)
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx, helix_integrand_By, helix_integrand_Bz]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.RHO, self.COSPHI,
                                                                   self.SINPHI, self.helicity,
                                                                   self.geom_coil.pitch_bar, self.geom_coil.L)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).item())
        B_vec = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vec_rot = self.coil_to_mu2e.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result
