import numpy as np
import torch as tc

# main integrator
def trapz_3d(xs, ys, zs, integrand_xyz, int_func=tc.trapz):
    return int_func(int_func(int_func(integrand_xyz, axis=-1, x=zs), axis=-1, x=ys), axis=-1, x=xs)

# helpful functions
# maybe move into CoilIntegrator class? FIXME!
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, hel, y):
    return y - hel*rho*SINPHI
def rz(zeta, phi, pitch, L, z):
    return z - (zeta + phi * pitch/(2*np.pi) - L/2)

def helix_integrand_Bx(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch, L):
    return (rho * COSPHI * RZ - hel * pitch/(2*np.pi) * RY) / R2_32
def helix_integrand_By(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch, L):
    return (hel * rho * SINPHI * RZ + hel* pitch/(2*np.pi) *RX) / R2_32
def helix_integrand_Bz(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch, L):
    return (-hel * rho * SINPHI * RY - rho * COSPHI * RX) / R2_32
