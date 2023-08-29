import csv, os
import pandas as pd
import logging
import json
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

exchange_name = sys.argv[1]

uniswapv2_logs = '/data/latest-data/all_logs_uniswapv2.csv'
sushiswap_logs = '/data/latest-data/all_logs_sushiswap.csv'

exchange_logs = {'uniswapv2' : uniswapv2_logs, 'sushiswap' : sushiswap_logs}

uniswapv2_pairs = pd.read_csv('/data/latest-data/uniswapv2_pairs.csv').set_index('pair')
eth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
# eth = "1097077688018008265106216665536940668749033598146"

logsdict = csv.DictReader(open(exchange_logs[exchange_name]), delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

# logs sorted by block number and then transaction indices (all logs from same txhash are consecutive)

def topics_from_text(raw_text):
    return json.loads(raw_text.replace('\'', '\"'))

interesting_blocks = defaultdict(lambda : set())
fout = open('eth_token_interesting_blocks_from_logs.csv', 'w')

#Interested in only Mint, Burn and Swap events
interested_topics = ['0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822']

parsed = 0
for log in logsdict:
    topics = topics_from_text(log['topics'])
    topics = log['topics'][1:-1]
    topics = topics.replace("'","").replace(" ", "").split(',')
    if topics[0] not in interested_topics:
        continue
    
    address = log['address']
    token0 = uniswapv2_pairs.loc[address].token0
    token1 = uniswapv2_pairs.loc[address].token1
    if token0 != eth and token1 != eth:
        continue
    data = log['data']
    data = data[2:] # strip 0x from hex
    block_number = log['block_number']

    amount0_in = int(str(data[:64]), 16)
    amount1_in = int(str(data[64:128]), 16)
    amount0_out = int(str(data[128:192]), 16)
    amount1_out = int(str(data[192:256]), 16)
    # print(amount0_in, amount1_in, address, token0, token1)
    if token0 == eth:
        if amount0_in > 5e20:
            fout.write("{},{},{}\n".format(address, block_number,amount0_in))
        elif amount0_out > 5e20:
            fout.write("{},{},{}\n".format(address, block_number,amount0_out))
    elif token1 == eth:
        if amount1_in > 5e20:
            fout.write("{},{},{}\n".format(address, block_number,amount1_in))
        elif amount1_out > 5e20:
            fout.write("{},{},{}\n".format(address, block_number,amount1_out))
    parsed += 1
    if  (parsed % 10000 == 0):
        logger.info("Parsed %d" %(parsed))

    

logger.info("Done...")
