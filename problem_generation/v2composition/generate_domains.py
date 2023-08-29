import glob
from web3 import Web3
import json
import os


dir = '../../eth_token_tests_v2composition'
NODE_URL = 'http://localhost:8545'
w3 = Web3(Web3.HTTPProvider(NODE_URL))

pair_abi = json.loads(open('../abi/uniswapv2pair_abi.json','r').read())
erc20_abi = json.loads(open('../abi/erc20_abi.json','r').read())
weth = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'

for filename in glob.glob(dir+'/*'):
    address = filename.split('/')[-1]
    pair_contract = w3.eth.contract(abi=pair_abi, address=w3.toChecksumAddress(address))
    token0 = pair_contract.functions.token0().call()
    token1 = pair_contract.functions.token1().call()
    if token0 == weth:
        token = token1
    else:
        token = token0
    token_contract = w3.eth.contract(abi=erc20_abi, address=w3.toChecksumAddress(token))
    token_decimals = token_contract.functions.decimals().call()
    lines = ["alpha1,0,1000", "alpha2,0,{}".format(int(10**(9+token_decimals))),"alpha3,0,1000", "alpha4,0,{}".format(int(10**(9+token_decimals))) ]
    out_filename = os.path.join(filename, 'domain')
    f = open(out_filename, 'w')
    f.write('\n'.join(lines))
    f.close()