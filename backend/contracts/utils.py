from web3 import Web3
import json
import eth_abi

def encode_parameters(contract, fn_name, params):
    #TODO: broken for arrays, use encode_funcion_call1 instead
    fn = contract.find_functions_by_name(fn_name)[0]
    ret = b''
    for i in range(len(params)):
        arg_type = fn.abi['inputs'][i]['type']
        if 'int' in arg_type:
            param = int(float(params[i])) #only handles int with base 10, handle hex strings in future
        elif '[]' in arg_type: #make more robust
            param = (params[i][1:-1]).split("-")
        else:
            # handle other types in future
            param = params[i]
        ret += eth_abi.encode_single(arg_type, param)
    return ret

def encode_function_signature(contract, fn_name):
    fn = contract.find_functions_by_name(fn_name)[0]
    types = [arg['type'] for arg in fn.abi['inputs']]
    selector =  Web3.sha3(text='{}({})'.format(fn_name,','.join(types)))[0:4]
    return selector

def encode_function_call(contract, fn_name, params):
    return encode_function_signature(contract, fn_name).hex() + encode_parameters(contract, fn_name, params).hex()

def typify(params, fn):
    typed_params = []
    for i in range(len(params)):
        arg_type = fn.abi['inputs'][i]['type']
        if 'int' in arg_type:
            param = int(float(params[i])) #only handles int with base 10, handle hex strings in future
            typed_params.append(param)
        elif 'bool' in arg_type:
            if params[i] == 'false':
                typed_params.append(False)
            elif params[i] == 'true':
                typed_params.append(True)
            else:
                # should not reach here
                param = params[i]
                typed_params.append(param)
        elif '[]' in arg_type: #make more robust
            param = (params[i][1:-1]).split("-")
            typed_params.append(param) 
        else:
            # handle other types in future
            param = params[i]
            typed_params.append(param)
    return typed_params

def encode_function_call1(contract, fn_name, params):
    # ugly uniswapv3, figure out encoding of tuples
    if fn_name == 'exactInputSingle':
        return "0x414bf389" + params[0][2:].lower().zfill(64) + params[1][2:].lower().zfill(64) + hex(int((params[2])))[2:].zfill(64) + params[3][2:].lower().zfill(64)+ hex(int((params[4])))[2:].zfill(64) + hex(int((params[5])))[2:].zfill(64) + hex(int((params[6])))[2:].zfill(64) + hex(int((params[7])))[2:].zfill(64)
    fn = contract.find_functions_by_name(fn_name)[0]
    return contract.encodeABI(fn_name = fn_name, args=typify(params, fn))