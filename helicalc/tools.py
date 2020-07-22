import subprocess
import matplotlib.pyplot as plt

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def config_plots():
    #must run twice for some reason (glitch in Jupyter)
    for i in range(2):
        plt.rcParams['figure.figsize'] = [10, 8] # larger figures
        plt.rcParams['axes.grid'] = True         # turn grid lines on
        plt.rcParams['axes.axisbelow'] = True    # put grid below points
        plt.rcParams['grid.linestyle'] = '--'    # dashed grid
        plt.rcParams.update({'font.size': 12.0})   # increase plot font size
