import csv
import subprocess
from subprocess import Popen, PIPE

block_times = './data/block_times.csv'
data_path = './data'
binance_prices_path = f'{data_path}/prices-binance/'
token_names = f'{data_path}/token_names.csv'


def get_price(block, token_addr,source='binance'):
    token_addr = token_addr.lower()
    if source=='binance':
        if token_addr == 'eth' or token_addr == '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2':
            market = 'ETHUSDT'
        elif token_addr == '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599':
            market = 'BTCUSDT'
        else:
            pipe = Popen(["grep", token_addr.lower(), token_names], stdout=PIPE, stderr=PIPE, close_fds=True)
            matched = pipe.stdout.read()
            token_name = str(matched, "utf-8").strip().split(",")[1]
            market = f"{token_name}USDT"
        if market == "USDTUSDT":
            return 1
        matched_ts = subprocess.check_output(f'grep {block}, {block_times}', shell=True)
        ts = int(str(matched_ts, "utf-8").strip().split(",")[1])
        minute_ts = ts//60 * 60
        matched_price = subprocess.check_output(f'grep {minute_ts*1000} {binance_prices_path}{market}.csv', shell=True)
        price = float(str(matched_price, "utf-8").strip().split(",")[1])
        return price
