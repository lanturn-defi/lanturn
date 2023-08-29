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
from web3 import Web3

def to_format(transaction):
    return '0,{},{}\n'.format(transaction['from'], transaction['hash'])


parser = argparse.ArgumentParser(description='Process uniswapv3 problems')

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
    '-t', '--token',
    help="token addr",
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
    # print(response)
    return response

args = parser.parse_args()

output = ''

for block in range(int(args.start_block), int(args.end_block)):
    # print(args.file)
    command = "grep '^" + str(block) + ",' " + args.file
    pipe = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    output += str(pipe.stdout.read() + pipe.stderr.read(), "utf-8")

lines = output.strip().split('\n')

block_to_tx = defaultdict(lambda : set())
for line in lines:
    if len(line) > 0:
        transaction_hash = line.split(',')[1]
        block_num = int(line.split(',')[0])
        block_to_tx[block_num].add(transaction_hash)


# token = hex(int(token))[2:].zfill(40) # remove else causes confusion
token_hex = Web3.toChecksumAddress(args.token)

# print(token_hex)
swap_template1 = '1,miner,UniswapV3Router,0,exactInputSingle,\
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},{},miner,1800000000,{},0,0'.format(token_hex,500,'alpha1')
swap_template2 = '1,miner,UniswapV3Router,0,exactInputSingle,\
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},{},miner,1800000000,{},0,0'.format(token_hex,3000,'alpha2')
swap_template3 = '1,miner,UniswapV3Router,0,exactInputSingle,\
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},{},miner,1800000000,{},0,0'.format(token_hex,10000,'alpha3')

swap_template4 = '1,miner,UniswapV3Router,0,exactInputSingle,\
{},0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},miner,1800000000,{},0,0'.format(token_hex,500,'alpha4')
swap_template5 = '1,miner,UniswapV3Router,0,exactInputSingle,\
{},0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},miner,1800000000,{},0,0'.format(token_hex,3000,'alpha5')
swap_template6 = '1,miner,UniswapV3Router,0,exactInputSingle,\
{},0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,{},miner,1800000000,{},0,0'.format(token_hex,10000,'alpha6')


insertions = [swap_template1,swap_template2,swap_template3,swap_template4,swap_template5,swap_template6]

dir = args.output_dir

for block in block_to_tx:
    print(block)
    filename1 = '{}/{}/amm'.format(dir, block)
    filename2 = '{}/{}/amm_reduced'.format(dir, block)
    os.makedirs(os.path.dirname(filename1), exist_ok=True)
    os.makedirs(os.path.dirname(filename2), exist_ok=True)
    f1 = open(filename1.format(), 'w')
    f2 = open(filename2.format(), 'w')
    necessary_transactions = block_to_tx[block]
    interacting_addresses = set()
    complete_block = query_block(block)
    if complete_block is None or complete_block['result'] is None:
        print(filename1)
        continue
    complete_output = ''
    reduced_output = ''
    all_transactions = complete_block['result']['transactions']
    for tx in all_transactions:
        if tx['hash'] in necessary_transactions:
            interacting_addresses.add(tx['from'])
            interacting_addresses.add(tx['to'])
    f1.write('{},{}\n'.format(block, token_hex))
    f2.write('{},{}\n'.format(block, token_hex))
    for tx in all_transactions:
        f1.write(to_format(tx))
        if tx['from'] in interacting_addresses or tx['to'] in interacting_addresses:
            f2.write(to_format(tx))
    for tx in insertions:
        f1.write('{}\n'.format(tx))
        f2.write('{}\n'.format(tx))