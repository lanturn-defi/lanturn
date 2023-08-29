import os
import pandas as pd

filename = "../eth_token_interesting_blocks.csv" #sushiswap's
sushiswap = '0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac'
uniswapv2 = '0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f'
pair_info = '/data/latest-data/uniswapv2_pairs.csv'
weth = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'

df = pd.read_csv(pair_info)

lines = open(filename, 'r').readlines()
for line in lines[::-1]:
    address = line.split(",")[0]
    if len(address) != 42:
        print('invalid addr')
        continue
    block = line.split(",")[1].strip()
    row = df[(df['pair'] == address) & (df['exchange'] == sushiswap)].iloc[0]
    if row.token0 == weth:
        token = row.token1
    elif row.token1 == weth:
        token = row.token0
    else:
        raise ValueError('A very specific bad thing happened.')
    sushiswap_address = address
    uniswapv2_df = df[(df['exchange'] == uniswapv2) & (df['token0'] == row.token0) & (df['token1'] == row.token1)]
    if len(uniswapv2_df) >= 1:
        uniswapv2_address = uniswapv2_df.iloc[0].pair
    else:
        print('not found in uniswapv2')
        continue
    cmd="python3 process_v2amm.py -sf /data/latest-data/sushiswap-processed/{}.csv -uf /data/latest-data/uniswapv2-processed/{}.csv -t {} -sb {} -eb {} -o ../../eth_token_tests_v2composition/{}".format(sushiswap_address, uniswapv2_address, token, block, int(block)+1, sushiswap_address)
    os.system(cmd)