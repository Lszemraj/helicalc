import numpy as np
import torch as tc

# main integrator
# 3D needed for helicalc
def trapz_3d(xs, ys, zs, integrand_xyz, int_func=tc.trapz):
    return int_func(int_func(int_func(integrand_xyz, axis=-1, x=zs), axis=-1, x=ys), axis=-1, x=xs)

# 2D needed for solcalc. Cylindrical integration
def trapz_2d(rs, zs, integrand_rz, int_func=tc.trapz):
    return int_func(int_func(integrand_rz, axis=-1, x=zs), axis=-1, x=rs)

# helpful functions
# maybe move into CoilIntegrator class? FIXME!
## HELIX
# wind in +phi
'''
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, hel, y):
    return y - hel*rho*SINPHI
def rz(zeta, phi, phi0, pitch_bar, L, t_gi, z_start, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    return z - (z_start + zeta + (phi-phi0) * pitch_bar + t_gi)
'''
# wind in -phi
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, y):
    return y - rho*SINPHI
def rz(zeta, phi, phi0, pitch_bar, t_gi, z_start, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    ##return z - (z_start + zeta + (phi-phi0) * pitch_bar + t_gi)
    # do we need "lib.abs"?
    return z - (z_start + zeta + abs(phi-phi0) * pitch_bar + t_gi)
# REF from plotting
# zs = z_start + np.abs(phis-phi0)/(2*np.pi) * df_.pitch

def helix_integrand_Bx(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (rho * COSPHI * RZ - hel * pitch_bar * RY) / R2_32 # for +helicity
    return (hel * rho * COSPHI * RZ - pitch_bar * RY) / R2_32
def helix_integrand_By(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (hel * rho * SINPHI * RZ + hel * pitch_bar * RX) / R2_32
    #return hel * (rho * SINPHI * RZ + pitch_bar * RX) / R2_32 # for +helicity
    return (hel * rho * SINPHI * RZ + pitch_bar * RX) / R2_32

def helix_integrand_Bz(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (-hel * rho * SINPHI * RY - rho * COSPHI * RX) / R2_32
    #return -rho * (hel * SINPHI * RY + COSPHI * RX) / R2_32 # for +helicity
    return - hel * rho * (SINPHI * RY + COSPHI * RX) / R2_32

# minimal terms
def rx_min(RHOCOSPHI, x):
    return x - RHOCOSPHI
def ry_min(RHOSINPHI, y):
    return y - RHOSINPHI
def rz_min(ZTERM, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    return z - ZTERM

def helix_integrand_Bx_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return (HRHOCOSPHI * RZ - pitch_bar * RY) / R2_32
def helix_integrand_By_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return (HRHOSINPHI * RZ + pitch_bar * RX) / R2_32

def helix_integrand_Bz_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return - (HRHOSINPHI * RY + HRHOCOSPHI * RX) / R2_32

####

## CIRCLE
def rx_circ(rho, COSPHI, x):
    return x - rho*COSPHI
def ry_circ(rho, SINPHI, y):
    return y - rho*SINPHI
# need to check rz...
def rz_circ(zeta, z):
    return z - zeta

def circle_integrand_Bx(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (rho * COSPHI * RZ) / R2_32
def circle_integrand_By(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (rho * SINPHI * RZ) / R2_32
def circle_integrand_Bz(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (-rho * SINPHI * RY - rho * COSPHI * RX) / R2_32

