#!/bin/bash
# Run SolCalc for the different magnet regions

source /home/ckampa/anaconda3/etc/profile.d/conda.sh
conda activate helicalc

# current setup is to only calculate coil 2

# run PS -- done
python calculate_PS_rot_single_region.py -r PS -c 1,2,3
# python calculate_PS_16mrad_single_region.py -r PS -c 1
# run TSu
python calculate_PS_rot_single_region.py -r TSu -c 1,2,3
# python calculate_PS_16mrad_single_region.py -r TSu -c 1
# run TSd
python calculate_PS_rot_single_region.py -r TSd -c 1,2,3
# python calculate_PS_16mrad_single_region.py -r TSd -c 1
# run DS
python calculate_PS_rot_single_region.py -r DS -c 1,2,3
# python calculate_PS_16mrad_single_region.py -r DS -c 1

# read -p "Press any key to resume ..."
