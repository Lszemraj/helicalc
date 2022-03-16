'''
Constants useful for Biot-Savart calculations
'''
import math
import numpy as np
from scipy.constants import mu_0

mu0 = mu_0 # 4*pi*1e-7    # permeability of free space

MAXMEM = 11019. # max memory per GPU, in MB
# FIXME! Should be able to get devices from pytorch
DEVICES = [0, 1, 2, 3] # GPU device numbers

# Mu2e Bmap coordinate definitions
# taken from Mau13 in CVMFS
PS_grid = {'X0':2.804, 'Y0':-1.200, 'Z0':-9.929,
           'nX':89, 'nY':97, 'nZ':281,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
# incorrect TSu grid!! X0 is wrong
#TSu_grid = {'X0':4.000, 'Y0':-1.200, 'Z0':-2.929,
#            'nX':201, 'nY':97, 'nZ':149,
#            'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSu_grid = {'X0':0.004, 'Y0':-1.200, 'Z0':-2.929,
           'nX':201, 'nY':97, 'nZ':149,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSd_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':-0.829,
            'nX':205, 'nY':97, 'nZ':157,
            'dX':0.025, 'dY':0.025, 'dZ':0.025}
DS_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':3.071,
           'nX':97, 'nY':97, 'nZ':521,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
# additions outside solenoid regions
# note that for now we do not do "flipy", so should include Y<0 in grid
PStoDumpArea_grid = {'X0':0.004, 'Y0':-5.500, 'Z0':-14.929,
                     'nX':73, 'nY':111, 'nZ':51,
                     'dX':0.100, 'dY':0.100, 'dZ':0.100}
# "flipy"
ProtonDumpArea_grid = {'X0':-0.796, 'Y0':-5.600, 'Z0':-20.929,
                       'nX':20, 'nY':57, 'nZ':31,
                       'dX':0.200, 'dY':0.200, 'dZ':0.200}

# dxyz for helicalc (nominal values for Mu2e DS coils)
# radius is hard coded for now.
# coarse integration grid (3x3 in cross section, 1/(5cm) in R*phi)
dxyz_dict_coarse = {1: np.array([3e-3,1e-3, 5e-2/1.05]),
                    2: np.array([2e-3,1e-3, 5e-2/1.05])}
# fine integration grid (3x3 in cross section, 1/(1cm) in R*phi)
dxyz_dict = {1: np.array([3e-3,1e-3, 1e-2/1.05]),
             2: np.array([2e-3,1e-3, 1e-2/1.05])}
