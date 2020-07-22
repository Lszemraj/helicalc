import torch as tc
import numpy as np

from helicalc.coil import CoilIntegrator
from helicalc.tools import *
from helicalc.geometry import *
from helicalc.integrate import *

import sys

if len(sys.argv) == 1:
    i = 0
else:
    i = int(sys.argv[1])

param_dir = '../dev/params/'
geom = 'DS_V13'

# read in geometry files
geom_df = read_solenoid_geom_combined(param_dir, geom, sep=',', skiprows=1)

CoilIG = CoilIntegrator(geom_df.iloc[i], dxyz=np.array([1e-3, 1e-3, 1e-4]))

print(f'Estimated Init Mem: {CoilIG.est_mem_init_mb} MB, Actual Init Mem: {CoilIG.actual_mem_init_mb} MB')
print(f'Estimated / Actual = {CoilIG.est_mem_init_mb/CoilIG.actual_mem_init_mb:0.3f}')
print(f'Estimated Run Mem: {CoilIG.est_mem_run_mb} MB = {CoilIG.est_mem_run_mb*1e-3:0.3f} GB')
