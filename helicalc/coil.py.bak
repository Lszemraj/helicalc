from sys import getsizeof
import numpy as np
import torch as tc
from scipy.spatial.transform import Rotation
from .integrate import *
from .constants import *
from .tools import *

'''
class MultiCoilIntegrator(object):
    # MUST HAVE EQUIV COIL GEOM
    def __init__(self, geom_coils, dxyz, int_func=tc.trapz, lib=tc, dev=0):
        self.names = []
        self.CoilIGs = []
        for geom_coil in geom_coils:
            for layer in range(1,geom_coil.N_layers+1):
                self.CoilIGs.append(CoilIntegrator(geom_coil,dxyz, layer, dev=dev))
                self.names.append(f'Coil_{geom_coil.Coil_Num}_Layer_{layer}')

        self.rhos = self.CoilIGs[0].rhos
        self.zetas = self.CoilIGs[0].zetas
        self.phis = self.CoilIGs[0].phis
        self.RHO = np.array([CIG.RHO for CIG in self.CoilIGs])
        self.ZETA = np.array([CIG.ZETA for CIG in self.CoilIGs])
        self.PHI = np.array([PHI.RHO for CIG in self.CoilIGs])
        self.COSPHI = np.array([COSPHI.RHO for CIG in self.CoilIGs])
        self.SINPHI = np.array([SINPHI.RHO for CIG in self.CoilIGs])
        self.geom_coils = geom_coils
        self.geom_coil = self.geom_coils[0]
        # rho_lims = [geom_coil.rho0_a, geom_coil.rho1_a]
        rho_lims = [rho0, rho1]
        zeta_lims = [geom_coil.zeta0, geom_coil.zeta1]
        phi_lims = [geom_coil.phi_i, geom_coil.phi_f]
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

    def integrate(self, x0=0, y0=0, z0=0):
        # rotate based on geom
        #x0, y0, z0 = transform_field_point(x0,y0,z0)
        # rotate/translate
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # calculate repeated calculations
        RX = rx(self.RHO, self.COSPHI, x0)
        RY = ry(self.RHO, self.SINPHI, self.geom_coil.helicity, y0)
        RZ = rz(self.ZETA, self.PHI, self.geom_coil.pitch_bar, self.geom_coil.L, z0)
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx, helix_integrand_By, helix_integrand_Bz]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.RHO, self.COSPHI,
                                                                   self.SINPHI, self.geom_coil.helicity,
                                                                   self.geom_coil.pitch_bar, self.geom_coil.L)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).item())
        B_vec = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vec_rot = self.coil_to_mu2e.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result
'''


class CoilIntegrator(object):
    def __init__(self, geom_coil, dxyz, layer=1, int_func=tc.trapz, lib=tc, dev=0):
        if layer > geom_coil.N_layers:
            raise ValueError(f'Layer "{layer}" invalid. Please select layer in the range [1, {int(geom_coil.N_layers)}]')
        # set correct device
        tc.cuda.set_device(dev)
        # check device initial memory
        self.start_dev_mem = get_gpu_memory_map()[dev]
        self.mem_err_expected = False
        # x, y, z --> rho, zeta, phi
        tc.cuda.set_device(dev)
        rho0 = geom_coil.rho0_a + (layer-1)*(geom_coil.h_cable + 2*geom_coil.t_ci + geom_coil.t_il)
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
        # try 64-bit
        # does not improve
        # self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda().type('torch.DoubleTensor')
        # self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda().type('torch.DoubleTensor')
        # self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda().type('torch.DoubleTensor')
        # 32-bit default
        ### TRY USING ARANGE INSTEAD OF LINSPACE (GET CORRECT ENDPOINTS)
        ### NO^^ (linspace does it correctly)
        # self.rhos = lib.arange(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
        # self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
        # self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
        self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
        self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
        self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
        # # estimate memory
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
        RZ = rz(self.ZETA, self.PHI, self.geom_coil.phi0, self.geom_coil.pitch_bar, self.geom_coil.L, self.geom_coil.t_gi, z0)
        # print(RZ)
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
