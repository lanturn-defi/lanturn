import yaml
import sys, os
sys.path.append('../../')
from util import uniswapv2_to_uniswapv3


def merge(uniswapv3_filename, uniswapv2_filename):
    a = open(uniswapv2_filename, 'r').readlines()
    b = open(uniswapv3_filename, 'r').readlines()
    transactions = []
    existing = set()
    for line in a:
        transaction = line.strip()
        transaction = transaction.replace('alpha1', 'alpha7').replace('alpha2', 'alpha8')
        if transaction not in existing:
            transactions.append(transaction)
            existing.add(transaction)
    for line in b:
        transaction = line.strip()
        if transaction not in existing:
            transactions.append(transaction)
            existing.add(transaction)
        else:
            print(transaction)
    return transactions

with open('uniswapv2_for_combo.yaml', 'r') as f:
    common_problems = yaml.safe_load(f)

for problem in common_problems:
    uniswapv2_address = problem.split('/')[-2]
    block_number = problem.split('/')[-1]
    uniswav3_address = uniswapv2_to_uniswapv3(uniswapv2_address)
    uniswapv3_filename = '../eth_token_tests_uniswapv3/{}/{}/amm_reduced'.format(uniswav3_address, block_number)

    transactions = merge(uniswapv3_filename, '{}/amm_reduced'.format(problem))
    outfile = '../eth_token_tests_uniswap_composition/{}/{}/amm_reduced'.format(uniswapv2_address, block_number)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print(outfile)
    with open(outfile, 'w') as f:
        for tx in transactions:
            f.write('{}\n'.format(tx))
