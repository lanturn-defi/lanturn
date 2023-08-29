import glob
from web3 import Web3
import json
import os


dir = '../../eth_token_tests_uniswapv3'

NODE_URL = 'http://localhost:8545'
w3 = Web3(Web3.HTTPProvider(NODE_URL))
erc20_abi = json.loads(open('../abi/erc20_abi.json','r').read())

for filename in glob.glob(dir+'/*'):
    token_address = filename.split('/')[-1]
    token_contract = w3.eth.contract(abi=erc20_abi, address=w3.toChecksumAddress(token_address))
    token_decimals = token_contract.functions.decimals().call()
    lines = ["alpha1,0,{}".format(int(10**(3+18))), "alpha2,0,{}".format(int(10**(3+18))), "alpha3,0,{}".format(int(10**(3+18))), "alpha4,0,{}".format(int(10**(9+token_decimals))),"alpha5,0,{}".format(int(10**(9+token_decimals))), "alpha6,0,{}".format(int(10**(9+token_decimals))) ]
    out_filename = os.path.join(filename, 'domain')
    f = open(out_filename, 'w')
    f.write('\n'.join(lines))
    f.close()