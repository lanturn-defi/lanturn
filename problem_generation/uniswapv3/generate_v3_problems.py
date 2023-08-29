import os
import pandas as pd

filename = "../eth_token_interesting_blocks_v3.csv"
pair_info = '/data/latest-data/uniswapv3_pools.csv'
weth = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'

df = pd.read_csv(pair_info)

lines = open(filename, 'r').readlines()
generated = set()
for line in lines[:]:
    address = line.split(",")[0]
    if len(address) != 42:
        print('invalid addr')
        continue
    block = int(line.split(",")[1].strip(), 16)
    row = df[(df['pool'] == address)].iloc[0]
    if row.token0 == weth:
        token = row.token1
    elif row.token1 == weth:
        token = row.token0
    else:
        raise ValueError('A very specific bad thing happened.')
    if (block, token) in generated:
        continue
    cmd="python3 process_v3.py -f /data/latest-data/uniswapv3-separated/{}.csv  -t {} -sb {} -eb {} -o ../../eth_token_tests_uniswapv3/{}".format(token, token, block, int(block)+1, token)
    os.system(cmd)
    generated.add((block, token))
    # print(len(generated))