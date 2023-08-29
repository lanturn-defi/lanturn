from web3 import Web3
import json
from pathlib import Path

w3 = Web3()
path = Path(__file__).parent / "aave_abi.json"
aave_abi = json.loads(path.open('r').read()) 
aave_contract = w3.eth.contract(abi=aave_abi, address='0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9')
