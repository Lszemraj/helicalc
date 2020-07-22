import torch as tc

def gpu_load(geom_coil, dxyz , dev=0):
    # set device
    tc.cuda.set_device(dev)
    # x, y, z --> rho, zeta, phi
    rho_lims = [geom_coil.rho0_a, geom_coil.rho1_a]
    zeta_lims = [geom_coil.zeta0, geom_coil.zeta1]
    phi_lims = [geom_coil.phi_i, geom_coil.phi_f]
    xyz_lims = [rho_lims, zeta_lims, phi_lims]
    rhos = tc.linspace(xyz_lims[0][0], xyz_lims[0][1], abs(int((xyz_lims[0][1]-xyz_lims[0][0])/dxyz[0] + 1))).cuda()
    zetas = tc.linspace(xyz_lims[1][0], xyz_lims[1][1], abs(int((xyz_lims[1][1]-xyz_lims[1][0])/dxyz[1] + 1))).cuda()
    phis = tc.linspace(xyz_lims[2][0], xyz_lims[2][1], abs(int((xyz_lims[2][1]-xyz_lims[2][0])/dxyz[2] + 1))).cuda()
    RHO, ZETA, PHI = tc.meshgrid(rhos,zetas,phis)
    SINPHI = tc.sin(PHI)
    COSPHI = tc.cos(PHI)
    #geom_coil = geom_coil
    #int_func = int_func
    #lib = lib
    #dev = dev
    XYZ_rot = geom_coil[[f'rot{i:d}' for i in [0,1,2]]].values
    XYZ_rot_rad = np.radians(XYZ_rot)
    mu2e_to_coil = Rotation.from_euler('XYZ', -XYZ_rot_rad)
    coil_to_mu2e = mu2e_to_coil.inv()
    xc, yc, zc = geom_coil[['x','y','z']].values
    #integrand_func = get_int_func(X,Y,Z, int_func)
    
    x0=0
    y0=0
    z0=0
    # rotate based on geom
    #x0, y0, z0 = transform_field_point(x0,y0,z0)
    # rotate/translate
    x0, y0, z0 = mu2e_to_coil.apply(np.array([x0-xc, y0-yc, z0-zc]))
    # calculate repeated calculations
    RX = rx(RHO, COSPHI, x0)
    RY = ry(RHO, SINPHI, geom_coil.helicity, y0)
    RZ = rz(ZETA, PHI, geom_coil.pitch, geom_coil.L, z0)
    R2_32 = (RX**2+RY**2+RZ**2)**(3/2)
    result = []
    # int_func must have params (x, y, z, x0, y0, z0)
    for integrand_func in [helix_integrand_Bx, helix_integrand_By, helix_integrand_Bz]:
        integrand_xyz = geom_coil.mu_fac * integrand_func(RX, RY, RZ, R2_32, RHO, COSPHI,
                                                               SINPHI, geom_coil.helicity,
                                                               geom_coil.pitch, geom_coil.L)
        result.append(trapz_3d(rhos, zetas, phis, integrand_xyz, tc.trapz).item())
    B_vec = np.array(result)
    # rotate vector back to mu2e coordinates
    B_vec_rot = coil_to_mu2e.apply(B_vec)
    last_result = B_vec_rot
    return last_result, B_vec
