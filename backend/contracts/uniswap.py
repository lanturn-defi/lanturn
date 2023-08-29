from web3 import Web3
import json
import eth_abi
from pathlib import Path

w3 = Web3()
path = Path(__file__).parent / "uniswap_router_abi.json"
uniswap_router_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty
uniswap_router_contract = w3.eth.contract(abi=uniswap_router_abi, address='0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D')
sushiswap_router_contract = w3.eth.contract(abi=uniswap_router_abi, address='0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F')

path = Path(__file__).parent / "uniswapv3_router_abi.json"
uniswapv3_router_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty
uniswapv3_router_contract = w3.eth.contract(abi=uniswapv3_router_abi, address='0xE592427A0AEce92De3Edee1F18E0157C05861564')

path = Path(__file__).parent / "uniswapv3_quoter_abi.json"
uniswapv3_quoter_abi = json.loads(path.open('r').read()) #TODO: weird abs path, make pretty

position_manager_path = Path(__file__).parent / "PositionManager.json"
position_manager_abi = json.load(position_manager_path.open('r'))["abi"]

temp_abi = json.load((Path(__file__).parent / "PositionManager.json").open('r'))["abi"]

def get_reserves(exchange_addr):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    function_selector = "0x0902f1ac000000000000000000000000"
    calldata = function_selector
    data["params"] = [{"to": exchange_addr, "data":calldata}, "latest"]
    # now = datetime.now()
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    result = response["result"]
    reserve0 = result[2:66]
    reserve1 = result[66:130]
    ts = result[130:]
    # print("Reserve TS", int(ts, 16))
    return reserve0, reserve1

def getAmountOutv1(token_addr, exchange_addr, router_contract, in_amount):
    reserve0, reserve1 = get_reserves(exchange_addr)
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'eth_call'
    if int(token_addr, 16) < int(weth, 16):
        calldata = utils.encode_function_call1(router_contract, 'getAmountOut', [in_amount, int(reserve0, 16), int(reserve1, 16)])
    else:
        calldata = utils.encode_function_call1(router_contract, 'getAmountOut', [in_amount, int(reserve1, 16), int(reserve0, 16)])
    data["params"] = [{"to": router_contract.address, "data":calldata}, "latest"]
    data['id'] = 1
    r = requests.post(FORK_URL, json=data)
    response = json.loads(r.content)
    if 'result' in response:
        return int(response['result'], 16)
    else:
        # TODO log response
        return 0
