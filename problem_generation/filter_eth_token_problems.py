import glob 
import pandas as pd
import os
from collections import defaultdict

data_dir = "../mev/data-scripts/latest-data/sushiswap-processed/"
reserves = pd.read_csv("../mev/data-scripts/latest-data/sushiswap-reserves.csv")
eth = "1097077688018008265106216665536940668749033598146"

interesting_blocks = defaultdict(lambda : set())

for filename in os.listdir(data_dir):
    address = filename.rstrip(".csv")
    pair_info = reserves[reserves.Address == address]
    if len(pair_info) > 0:
        token0 = pair_info.iloc[0].Token0
        token1 = pair_info.iloc[0].Token1
        if token0 != eth and token1 != eth:
            continue # ignore token-token pairs, only consider eth-token
    curr_block = 0
    transaction_data = open(data_dir + filename).readlines()
    for line in transaction_data:
        if line.startswith('//'):
            curr_block = int(line.split()[-1])
        elif 'swaps' in line:
            chunks = line.split()
            amount0 = chunks[6]
            token0 = chunks[7]
            if token0 == eth and int(amount0) > 5e20:
                print(address, curr_block)
                interesting_blocks[address].add(curr_block)

f = open('eth_token_interesting_blocks.csv', 'w')
for address in interesting_blocks:
    for b in interesting_blocks[address]:
        f.write("{},{}\n".format(address, b))