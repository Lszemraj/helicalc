import numpy as np
import torch as tc
from scipy.spatial.transform import Rotation
from .integrate import *
from .constants import *
from .tools import *

class CoilIntegrator(object):
    def __init__(self, geom_coil, dxyz, int_func=tc.trapz, lib=tc, dev=0):
        self.start_dev_mem = get_gpu_memory_map()[dev]
        # x, y, z --> rho, zeta, phi
        tc.cuda.set_device(dev)
        rho_lims = [geom_coil.rho0_a, geom_coil.rho1_a]
        zeta_lims = [geom_coil.zeta0, geom_coil.zeta1]
        phi_lims = [geom_coil.phi_i, geom_coil.phi_f]
        xyz_lims = [rho_lims, zeta_lims, phi_lims]
        self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
        self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
        self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
        # estimate memory
        per_el = 7e-6
        lr = len(self.rhos)
        lz = len(self.zetas)
        lp = len(self.phis)
        self.est_mem_init_mb = (2*(lr + lz + lp) + 2*(lr*lz*lp))*per_el
        self.est_mem_run_mb = (2*(lr + lz + lp) + 7*(lr*lz*lp))*per_el
        # end init if est. run memory too large (give a message)
        mult = 0.95
        if self.est_mem_run_mb > mult*MAXMEM:
            print(f'Estimated Run Memory: {self.est_mem_run_mb*1e-3} GB >',
                  f'{mult}*{MAXMEM} GB = {mult*MAXMEM} GB')
        else:
            self.RHO, self.ZETA, self.PHI = lib.meshgrid(self.rhos,self.zetas,self.phis)
            self.SINPHI = lib.sin(self.PHI)
            self.COSPHI = lib.cos(self.PHI)
        self.actual_mem_init_mb = get_gpu_memory_map()[dev] - self.start_dev_mem
        self.geom_coil = geom_coil
        self.int_func = int_func
        self.lib = lib
        self.dev = dev
        self.XYZ_rot = geom_coil[[f'rot{i:d}' for i in [0,1,2]]].values
        self.XYZ_rot_rad = np.radians(self.XYZ_rot)
        self.mu2e_to_coil = Rotation.from_euler('XYZ', -self.XYZ_rot_rad)
        self.coil_to_mu2e = self.mu2e_to_coil.inv()
        self.xc, self.yc, self.zc = geom_coil[['x','y','z']].values

    def integrate(self, x0=0, y0=0, z0=0):
        # rotate based on geom
        #x0, y0, z0 = transform_field_point(x0,y0,z0)
        # rotate/translate
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # calculate repeated calculations
        RX = rx(self.RHO, self.COSPHI, x0)
        RY = ry(self.RHO, self.SINPHI, self.geom_coil.helicity, y0)
        RZ = rz(self.ZETA, self.PHI, self.geom_coil.pitch, self.geom_coil.L, z0)
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx, helix_integrand_By, helix_integrand_Bz]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.RHO, self.COSPHI,
                                                                   self.SINPHI, self.geom_coil.helicity,
                                                                   self.geom_coil.pitch, self.geom_coil.L)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).item())
        B_vec = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vec_rot = self.coil_to_mu2e.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        return self.last_result
