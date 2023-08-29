import os
from pathlib import Path
from collections import defaultdict
import json
import requests
import argparse
import logging
from subprocess import Popen, PIPE
import pandas as pd
import re
import random

def to_format(transaction):
    return '0,{},{}\n'.format(transaction['from'], transaction['hash'])


parser = argparse.ArgumentParser(description='Generate problems')

parser.add_argument(
    '-v', '--verbose',
    help="Be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
    default=logging.WARNING
)

parser.add_argument(
    '-f', '--file',
    help="Input File path",
    required=True
)

parser.add_argument(
    '-sb', '--start-block',
    help="StartBlock",
    required=True
)

parser.add_argument(
    '-eb', '--end-block',
    help="EndBlock",
    required=True
)

parser.add_argument(
    '-o', '--output-dir',
    help="output dir",
    required=True
)


ARCHIVE_NODE_URL = 'http://localhost:8545'
def query_block(block_number):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_getBlockByNumber'
    data['params'] = [block_number, True] # get full tx
    data['id'] = block_number + 1000000
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    return response

args = parser.parse_args()

output = ''

for block in range(int(args.start_block), int(args.end_block)):
    pipe = Popen("grep 'block " + str(block) + "' " + args.file, shell=True, stdout=PIPE, stderr=PIPE)
    output += str(pipe.stdout.read() + pipe.stderr.read(), "utf-8")

lines = output.strip().split('\n')
block_to_tx = defaultdict(lambda : set())
for line in lines:
    if line.startswith('//') and len(line) > 0:
        transaction_hash = line.split()[2]
        block_num = int(line.split()[4])
        block_to_tx[block_num].add(transaction_hash)



swap_template1 = '1,miner,SushiswapRouter,{},swapExactETHForTokens,\
0,[0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2-0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48],miner,1800000000'.format('alpha1')
swap_template2 = '1,miner,SushiswapRouter,0,swapExactTokensForETH,\
{},0,[0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48-0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2],miner,1800000000'.format('alpha2')


insertions = [swap_template1, swap_template2]

dir = args.output_dir

filename = '{}/{}_{}'.format(dir, args.start_block, args.end_block)
os.makedirs(os.path.dirname(filename), exist_ok=True)
agg_output = ''

for block in block_to_tx:
    print(block)
    necessary_transactions = block_to_tx[block]
    interacting_addresses = set()
    complete_block = query_block(block)
    all_transactions = complete_block['result']['transactions']
    
    prev_size = 0
    seed_transactions = set(necessary_transactions)
    while prev_size != len(interacting_addresses) or prev_size == 0:
        print(prev_size)
        prev_size = len(interacting_addresses)
        for tx in all_transactions:
            if tx['hash'] in seed_transactions:
                interacting_addresses.add(tx['from'])
                interacting_addresses.add(tx['to'])
        for tx in all_transactions:
            if tx['from'] in interacting_addresses or tx['to'] in interacting_addresses:
                seed_transactions.add(tx['hash'])

    for tx in all_transactions:
        if tx['from'] in interacting_addresses or tx['to'] in interacting_addresses:
            agg_output += to_format(tx)

f1 = open(filename.format(), 'w')
f1.write('{},usdc\n'.format(args.start_block))
f1.write(agg_output)
for tx in insertions:
    f1.write('{}\n'.format(tx))