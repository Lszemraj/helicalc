'''
Constants useful for Biot-Savart calculations
'''
import math
import numpy as np
from scipy.constants import mu_0

mu0 = mu_0 # 4*pi*1e-7    # permeability of free space

MAXMEM = 11019. # max memory per GPU, in MB
DEVICES = [0, 1, 2, 3] # GPU device numbers (should be able to get from pytorch) FIXME!

# Mu2e Bmap coordinate definitions
# taken from Mau13 in CVMFS
PS_grid = {'X0':2.804, 'Y0':-1.200, 'Z0':-9.929,
           'nX':89, 'nY':97, 'nZ':281,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSu_grid = {'X0':4.000, 'Y0':-1.200, 'Z0':-2.929,
            'nX':201, 'nY':97, 'nZ':149,
            'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSd_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':-0.829,
            'nX':205, 'nY':97, 'nZ':157,
            'dX':0.025, 'dY':0.025, 'dZ':0.025}
DS_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':3.071,
           'nX':97, 'nY':97, 'nZ':521,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
