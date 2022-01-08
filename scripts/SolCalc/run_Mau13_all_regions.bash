#!/bin/bash
# Run SolCalc for the different magnet regions

source /home/ckampa/anaconda3/etc/profile.d/conda.sh
conda activate helicalc

# run PS -- done
# python calculate_Mau13_single_region.py -r PS -t n
# run PS
python calculate_Mau13_single_region.py -r TSu -t n
# run PS
python calculate_Mau13_single_region.py -r TSd -t n
# run PS
python calculate_Mau13_single_region.py -r DS -t n

# read -p "Press any key to resume ..."
