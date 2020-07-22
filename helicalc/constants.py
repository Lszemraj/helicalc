'''
Constants useful for Biot-Savart calculations
'''
import math
import numpy as np
from scipy.constants import mu_0

mu0 = mu_0 # 4*pi*1e-7    # permeability of free space

MAXMEM = 11019. # max memory per GPU, in MB
DEVICES = [0, 1, 2, 3] # GPU device numbers (should be able to get from pytorch) FIXME!
