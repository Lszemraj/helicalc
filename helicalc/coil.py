from sys import getsizeof
import numpy as np
import torch as tc
from scipy.spatial.transform import Rotation
from .integrate import *
from .constants import *
from .tools import *


class CoilIntegrator(object):
    def __init__(self, geom_coil, dxyz, layer=1, int_func=tc.trapz, lib=tc, dev=0, interlayer_connect=True):
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
        rho_lims = [rho0, rho1]
        # helicity
        self.helicity = int(geom_coil.helicity * (-1)**(layer-1))
        # phi lims
        if layer == geom_coil.N_layers:
            # in last layer, wind until phi1 reached
            self.last_turn_rad = 2*np.pi - abs(geom_coil.phi1-geom_coil.phi0)
            self.phi_i = geom_coil.phi0
            self.phi_f = geom_coil.phi0 + self.helicity*(2*np.pi*(geom_coil.N_turns-1) + self.last_turn_rad)
        else:
            self.phi_i = geom_coil.phi0
            self.phi_f = geom_coil.phi0 + self.helicity*(2*np.pi*geom_coil.N_turns)
            if interlayer_connect:
                if self.helicity < 0:
                    # connect brick at phi_i
                    self.phi_i = self.phi_i + np.radians(36.)
                else:
                    # connect brick at phi_f
                    self.phi_f = self.phi_f - np.radians(36.)
        zeta_lims = [geom_coil.zeta0, geom_coil.zeta1]
        # phi_lims = [geom_coil.phi_i, geom_coil.phi_f]
        phi_lims = [self.phi_i, self.phi_f]
        xyz_lims = [rho_lims, zeta_lims, phi_lims]
        if lib is tc:
            # set up trapezoidal integration points (per dimension)
            self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
            self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
            self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int(abs(xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
        else:
            # set up trapezoidal integration points (per dimension)
            self.rhos = lib.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1)))
            self.zetas = lib.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1)))
            self.phis = lib.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int(abs(xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1)))
        # where to reference z from
        self.z_ref = - self.helicity * geom_coil.L / 2
        if self.helicity < 0:
            # sort of ad hoc... FIXME!
            self.z_start = self.z_ref - geom_coil.L - 2*geom_coil.t_ci + 5e-5
        else:
            self.z_start = self.z_ref
        # setup integration steps
        self.RHO, self.ZETA, self.PHI = lib.meshgrid(self.rhos,self.zetas,self.phis, indexing='ij')
        self.SINPHI = lib.sin(self.PHI)
        self.COSPHI = lib.cos(self.PHI)
        # extra calculations for r_i
        self.RHOCOSPHI = self.RHO * self.COSPHI
        self.RHOSINPHI = self.RHO * self.SINPHI
        self.HRHOCOSPHI = self.helicity * self.RHOCOSPHI
        self.HRHOSINPHI = self.helicity * self.RHOSINPHI
        self.ZTERM = self.z_start + self.ZETA + abs(self.PHI - self.phi_i) * geom_coil.pitch_bar + geom_coil.t_gi

        # # estimate memory
        # Note that this is basically obsolete. Either modernize or remove. FIXME!
        per_el_rh = 1.1e-5
        per_el_z = 2.27e-5
        per_el_ph = 4e-6
        lr = len(self.rhos)
        lz = len(self.zetas)
        lp = len(self.phis)
        # testing memory capacity
        per_el = 7e-6
        N_el_init = 2*(lr + lz + lp) + 2*lr*lz*lp
        N_el_run = 2*(lr + lz + lp) + 7*lr*lz*lp
        self.est_mem_init_mb = N_el_init * per_el
        self.est_mem_run_mb = N_el_run * per_el
        mult = 0.95
        if self.est_mem_run_mb > mult*MAXMEM:
            print(f'Estimated Run Memory: {self.est_mem_run_mb*1e-3} GB >',
                  f'{mult}*{MAXMEM} GB = {mult*MAXMEM} GB')
            self.mem_err_expected = True
        # estimate required memory
        self.actual_mem_init_mb = get_gpu_memory_map()[dev] - self.start_dev_mem
        # save extras to object
        self.geom_coil = geom_coil
        self.int_func = int_func
        self.layer = layer
        self.lib = lib
        self.dev = dev
        # rotation function
        self.XYZ_rot = geom_coil[[f'rot{i:d}' for i in [0,1,2]]].values
        self.XYZ_rot_rad = np.radians(self.XYZ_rot)
        self.mu2e_to_coil = Rotation.from_euler('XYZ', -self.XYZ_rot_rad)
        self.coil_to_mu2e = self.mu2e_to_coil.inv()
        self.xc, self.yc, self.zc = geom_coil[['x','y','z']].values
        # check sizes
        sizes = []
        for o in [self.rhos, self.zetas, self.phis, self.RHO, self.ZETA, self.PHI,
                  self.SINPHI, self.COSPHI]:
            # lazy...but avoids errors when using numpy. FIXME!
            try:
                sizes.append(getsizeof(o.storage())*1e-6)
            except:
                pass
        self.getsizeof_init_mb = np.array(sizes)

    def integrate(self, x0=0, y0=0, z0=0):
        # rotate/translate
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # calculate repeated calculations
        RX = rx(self.RHO, self.COSPHI, x0)
        RY = ry(self.RHO, self.SINPHI, y0)
        RZ = rz(self.ZETA, self.PHI, self.phi_i, self.geom_coil.pitch_bar, self.geom_coil.t_gi, self.z_start, z0)
        # test saving RX, RY, RZ to the object
        self.RX = RX
        self.RY = RY
        self.RZ = RZ
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
        # VERY SLOW! CAUTION!
        #self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result

    def integrate_v2(self, x0=0, y0=0, z0=0):
        # rotate/translate
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # calculate repeated calculations
        RX = rx_min(self.RHOCOSPHI, x0)
        RY = ry_min(self.RHOSINPHI, y0)
        RZ = rz_min(self.ZTERM, z0)
        # test saving RX, RY, RZ to the object
        self.RX = RX
        self.RY = RY
        self.RZ = RZ
        # print(RZ)
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx_min, helix_integrand_By_min, helix_integrand_Bz_min]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.HRHOCOSPHI,
                                                                   self.HRHOSINPHI, self.geom_coil.pitch_bar)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).item())
        B_vec = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vec_rot = self.coil_to_mu2e.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        # VERY SLOW! CAUTION!
        #self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result

    def integrate_vec_iterate(self, x0_vec=np.zeros(100), y0_vec=np.zeros(100), z0_vec=np.zeros(100)):
        # rotate/translate
        x0_vec, y0_vec, z0_vec = self.mu2e_to_coil.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        # loop over field points
        B_vecs = []
        for x0, y0, z0 in zip(x0_vec, y0_vec, z0_vec):
            # calculate repeated calculations
            RX = rx_min(self.RHOCOSPHI, x0)
            RY = ry_min(self.RHOSINPHI, y0)
            RZ = rz_min(self.ZTERM, z0)
            # test saving RX, RY, RZ to the object
            self.RX = RX
            self.RY = RY
            self.RZ = RZ
            # print(RZ)
            R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
            result = []
            # int_func must have params (x, y, z, x0, y0, z0)
            for integrand_func in [helix_integrand_Bx_min, helix_integrand_By_min, helix_integrand_Bz_min]:
                integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.HRHOCOSPHI,
                                                                       self.HRHOSINPHI, self.geom_coil.pitch_bar)
                result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func))
            if self.lib is tc:
                B_vec = tc.stack(result)
            else:
                B_vec = np.array(result)
            B_vecs.append(B_vec)
        if self.lib is tc:
            B_vecs = tc.stack(B_vecs).cpu().T
        else:
            B_vecs = np.array(B_vecs).T
        # rotate vector back to mu2e coordinates
        B_vecs_rot = self.coil_to_mu2e.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        # VERY SLOW! CAUTION!
        #self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result

    def integrate_vec(self, x0_vec=np.zeros(100), y0_vec=np.zeros(100), z0_vec=np.zeros(100)):
        x0_vec, y0_vec, z0_vec = self.mu2e_to_coil.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        RX = rx(self.RHO[None,...], self.COSPHI[None,...], x0_vec[:,None,None,None])
        RY = ry(self.RHO[None,...], self.SINPHI[None,...], y0_vec[:,None,None,None])
        RZ = rz(self.ZETA[None,...], self.PHI[None,...], self.phi_i, self.geom_coil.pitch_bar, self.geom_coil.t_gi, self.z_start, z0_vec[:,None,None,None])
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx, helix_integrand_By, helix_integrand_Bz]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.RHO[None,...], self.COSPHI[None,...],
                                                                  self.SINPHI[None,...], self.helicity, self.geom_coil.pitch_bar, self.geom_coil.L)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).cpu())
        B_vecs = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vecs_rot = self.coil_to_mu2e.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        # VERY SLOW! CAUTION!
        #self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result

    def integrate_vec_v2(self, x0_vec=np.zeros(100), y0_vec=np.zeros(100), z0_vec=np.zeros(100)):
        x0_vec, y0_vec, z0_vec = self.mu2e_to_coil.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        RX = rx_min(self.RHOCOSPHI[None,...], x0_vec[:,None,None,None])
        RY = ry_min(self.RHOSINPHI[None,...], y0_vec[:,None,None,None])
        RZ = rz_min(self.ZTERM[None,...], z0_vec[:,None,None,None])
        # test saving RX, RY, RZ to the object
        self.RX = RX
        self.RY = RY
        self.RZ = RZ
        R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [helix_integrand_Bx_min, helix_integrand_By_min, helix_integrand_Bz_min]:
            integrand_xyz = self.geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, self.HRHOCOSPHI[None,...],
                                                                   self.HRHOSINPHI[None,...], self.geom_coil.pitch_bar)
            result.append(trapz_3d(self.rhos, self.zetas, self.phis, integrand_xyz, self.int_func).cpu())
        if self.lib is tc:
            B_vecs = tc.stack(result).cpu()
        else:
            B_vecs = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vecs_rot = self.coil_to_mu2e.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        # VERY SLOW! CAUTION!
        #self.actual_mem_run_mb = get_gpu_memory_map()[self.dev] - self.start_dev_mem
        return self.last_result
