import csv, os
import pandas as pd
import logging
import json
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

uniswapv3_logs = '/data/latest-data/all_logs_uniswapv3.csv'

uniswapv3_pairs = pd.read_csv('/data/latest-data/uniswapv3_pools.csv').set_index('pool')
eth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
# eth = "1097077688018008265106216665536940668749033598146"

logsdict = csv.DictReader(open(uniswapv3_logs), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# logs sorted by block number and then transaction indices (all logs from same txhash are consecutive)

def topics_from_text(raw_text):
    return json.loads(raw_text.replace('\'', '\"'))

interesting_blocks = defaultdict(lambda : set())
fout = open('eth_token_interesting_blocks_v3.csv', 'w')

#Interested in only Mint, Burn and Swap events
interested_topics = ['0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67']

parsed = 0
for log in logsdict:
    topics = topics_from_text(log['topics'])
    topics = log['topics'][1:-1]
    topics = topics.replace("'","").replace(" ", "").split(',')
    if topics[0] not in interested_topics:
        continue
    
    address = log['address']
    token0 = uniswapv3_pairs.loc[address].token0
    token1 = uniswapv3_pairs.loc[address].token1
    if token0 != eth and token1 != eth:
        continue
    data = log['data']
    data = data[2:] # strip 0x from hex
    block_number = log['block_number']

    amount0_delta = int.from_bytes(bytes.fromhex(data[:64]) , byteorder='big', signed=True)
    amount1_delta = int.from_bytes(bytes.fromhex(data[64:128]) , byteorder='big', signed=True)
    # print(amount0_in, amount1_in, address, token0, token1)
    if token0 == eth:
        if abs(amount0_delta) > 1e21:
            fout.write("{},{},{}\n".format(address, block_number,abs(amount0_delta)))
    elif token1 == eth:
        if abs(amount1_delta) > 1e21:
            fout.write("{},{},{}\n".format(address, block_number,abs(amount1_delta)))
    parsed += 1
    if  (parsed % 10000 == 0):
        logger.info("Parsed %d" %(parsed))

    

logger.info("Done...")
