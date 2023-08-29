import requests
import json


outfile = 'block_times.csv'
startblock = 9000000
endblock = 15500000

ARCHIVE_NODE_URL = 'http://localhost:8545'

def get_block_unix_time(block_num):
    data = {}
    data['jsonrpc'] = '2.0'
    data['method'] = 'erigon_getHeaderByNumber'
    data['params'] = [block_num]
    data['id'] = block_num
    r = requests.post(ARCHIVE_NODE_URL, json=data)
    response = json.loads(r.content)
    # print(response)
    return int(response['result']['timestamp'],16)

block_times = {}

with open(outfile, 'w', newline='') as fout:
    fout.write('block,unixtime\n')

with open(outfile, 'a', newline='') as fout:
    for block in range(startblock, endblock+1):
        if block % 1000 == 0:
            print(block)
        # block_times[block] = get_block_unix_time(block)
        # fout.write('{},{}\n'.format(block,block_times[block]))    
        fout.write('{},{}\n'.format(block,get_block_unix_time(block)))    
