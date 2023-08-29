import os
import yaml
import numpy as np
import pandas as pd
import time
from subprocess import Popen, PIPE

def run_optimization(optimization_command):
    cwd_dir = os.getcwd()
    sim_dir = os.path.join(cwd_dir, 'eth_clients', 'hardhat')
    # restart simulation nodes
    os.chdir(sim_dir)
    print("killing sim nodes")
    pipe = Popen(["bash", "kill_hardhat.sh"], stdout=PIPE, stderr=PIPE, close_fds=True)
    (_output, err) = pipe.communicate()
    if 'not permitted' in err.decode('utf-8'):
        print("Simulation nodes not reachable! Exiting...")
        return
    print("launching sim nodes")
    pipe = Popen(["bash", "launch_hardhats.sh"], stdout=PIPE, stderr=PIPE, close_fds=True)
    time.sleep(5) # wait for simulation nodes to start up
    # ping simulation nodes
    print("pinging sim nodes")
    pipe = Popen(["bash", "ping_hardhats.sh"], stdout=PIPE, stderr=PIPE, close_fds=True)
    (_output, err) = pipe.communicate()
    if 'refused' in err.decode('utf-8'):
        print("Simulation nodes not reachable! Exiting...")
        return
    os.chdir(cwd_dir)
    # execute optimization command
    #print(command)
    print("running optimization")
    os.system(optimization_command)
    

TRANSACTION = 'examples/0xeth/aave2'
DOMAIN = 'examples/0xeth/aave2_domain'
DEXES = 'sushiswap aave uniswapv3'
command = f'python optimize.py -t {TRANSACTION} -d {DOMAIN} --dexes {DEXES} --reorder \
                    --n_iter 5 --num_samples 10 --parents_portion 0.0 --p_swap_max 0.8 --p_swap_min 0.1 \
                    --n_iter_gauss 50 --num_samples_gauss 40 --gauss_random_loguniform --u_random_portion_gauss 0.4 --local_portion 0.3 --cross_portion 0.3 --n_parallel_gauss 40 --capital 10000'
run_optimization(command)
