import json
import argparse

parser = argparse.ArgumentParser(description='Process fb data')
parser.add_argument('-d', '--data', help="Input File path containing raw data")
parser.add_argument('-o', '--outfile', help="output csv")

args = parser.parse_args()  

f = open(args.data, 'r')
data = json.load(f)

print("read data")

fout = open(args.outfile, 'w')
fout.write("blocknumber,fb_mev\n")
for block in data:
    fout.write("{},{}\n".format(block["block_number"], float(block["miner_reward"])/1e18))