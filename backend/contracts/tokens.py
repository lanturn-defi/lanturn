from web3 import Web3
import json
import eth_abi
from pathlib import Path

w3 = Web3()
path = Path(__file__).parent / "erc20_abi.json"
erc20_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty

path = Path(__file__).parent / "weth_abi.json"
weth_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty

token_contracts = {}