import numpy as np
import torch as tc

# main integrator
def trapz_3d(xs, ys, zs, integrand_xyz, int_func=tc.trapz):
    return int_func(int_func(int_func(integrand_xyz, axis=-1, x=zs), axis=-1, x=ys), axis=-1, x=xs)

# helpful functions
# maybe move into CoilIntegrator class? FIXME!
## HELIX
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, hel, y):
    return y - hel*rho*SINPHI
def rz(zeta, phi, phi0, pitch_bar, L, t_gi, z):
    return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)

def helix_integrand_Bx(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (rho * COSPHI * RZ - hel * pitch_bar * RY) / R2_32
def helix_integrand_By(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (hel * rho * SINPHI * RZ + hel* pitch_bar *RX) / R2_32
def helix_integrand_Bz(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    return (-hel * rho * SINPHI * RY - rho * COSPHI * RX) / R2_32
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

