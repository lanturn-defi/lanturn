import os

filename = "eth_token_interesting_blocks.csv"
# filename = "eth_token_interesting_blocks_from_logs.csv"
lines = open(filename, 'r').readlines()
for line in lines[::-1]:
    address = line.split(",")[0]
    block = line.split(",")[1].strip()
    cmd="python3 generate_client_problems.py -f /data/latest-data/sushiswap-processed/{}.csv -sb {} -eb {} -o ../eth_token_tests_v2composition/{}".format(address, block, int(block)+1, address)
    os.system(cmd)
    # print(cmd)