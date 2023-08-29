import glob
import os 
import zipfile

data_dir = 'prices/data/spot/monthly/klines'
out_dir = 'prices-binance'

for pair in os.scandir(data_dir):
    if not pair.is_dir():
        continue
    market = pair.name
    archives = f'{data_dir}/{market}/1m/*/*zip'
    outfilename = f'{out_dir}/{market}.csv'
    print(market)
    fout = open(outfilename, 'w')
    # print(archives)
    for archive in glob.glob(archives):
        contents = zipfile.ZipFile(archive, 'r')
        for filename in contents.namelist():
            data = contents.read(filename)
            for line in data.splitlines():
                vals = line.decode('utf-8').split(',')
                ts = vals[0]
                openprice = vals[1]
                closeprice = vals[4]
                price = (float(openprice) + float(closeprice))/2
                fout.write('{},{}\n'.format(ts,price))
    fout.close()