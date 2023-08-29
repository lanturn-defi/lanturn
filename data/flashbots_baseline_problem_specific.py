import json
import argparse
from collections import defaultdict
import glob
import sys
import requests
problemsdir = '../eth_token_tests_uniswapv3/*/*/amm_reduced'

parser = argparse.ArgumentParser(description='Run Optimization')
parser.add_argument('-d', '--data', help="Input File path containing raw data")

demanded_data = json.load(open('flashbots_data_on_demand.json', 'r'))

def calc_baseline(block, problem_transactions):
    bundle_rewards = defaultdict(lambda : 0)
    bundles = defaultdict(lambda: [])
    involved_bundles = set()

    for tx in block["transactions"]:
        bundles[tx['bundle_index']].append(tx)

    for idx in bundles:
        for tx in bundles[idx]:
            bundle_rewards[idx] += int(tx['total_miner_reward'])
    
    for idx in bundles:
        for tx in bundles[idx]:
            if tx['transaction_hash'] in problem_transactions:
                involved_bundles.add(idx)
    
    total_reward = 0
    for idx in involved_bundles:
        total_reward += bundle_rewards[idx]
    return total_reward

def get_baseline_data(block_number):
    if block_number < 11800000:
        return None
    if block_number <= 14986955:
        for block in data:
            if block['block_number'] == block_number:
                return block
        return None
    else:
        if str(block_number) in demanded_data:
            block = demanded_data[str(block_number)]
            if block == {}:
                return None
            else:
                return block 
        else:
            return None
        

def get_baseline_problem(problem_file):
    fp = open(problem_file, 'r')
    lines = fp.readlines()
    if len(lines) == 0:
        return -1
    block_number = int(lines[0].strip().split(',')[0])
    
    baseline_data = get_baseline_data(block_number)
    if baseline_data is None:
        return -1
    
    problem_transactions = set()
    for line in lines[1:]:
        if line.startswith('0,'):
            problem_transactions.add(line.strip().split(',')[-1])
    
    return calc_baseline(baseline_data, problem_transactions)

args = parser.parse_args()
f = open(args.data, 'r')
data = json.load(f)

print('blocknumber,fb_mev')
for filename in glob.glob(problemsdir):
    fb_value = get_baseline_problem(filename)
    if fb_value >= 0:
        print('{},{}'.format(filename, float(fb_value)/1e18))
        sys.stdout.flush()
