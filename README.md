# Pre-requisites
Run archive node, serving Ethereum HTTP JSON-RPC on port 8545

Install common software dependencies: `python`, `npm`

Python Modules: `pip3 install GitPython pandas tqdm scikit-learn`

# Simulation Clients
The following scripts by default launch 44 simulation networks.

`cd eth_clients/hardhat`

`bash setup_hardhat.sh`

`bash launch_hardhats.sh`

`eth_clients/hardhat` directory has detailed instructions on managing sim clients

# Download data:
Download and uncompress the zip file from the URL: [https://lanturn.s3.us-east-2.amazonaws.com/dataset.zip](https://lanturn.s3.us-east-2.amazonaws.com/dataset.zip)

Copy the files from `dataset/data/` to `data/` directory of this repository. This data is used for deriving the prices for Binance (CEX settlement of non-ETH currencies), although optimizations and simulations can run without this data, in which case the settlement will occur only on dexes.

`problem-dataset` dir contains the input transactions (user transactions and templates) and domain for optimization.

`baseline-dataset` dir contains the flashbots baseline for the above problems.


# Sample optimization command:
`python3 optimize.py -t <example_transactions> -d <example_domain> --dexes <contract_list> --n_parallel_gauss 44 --capital 1000 --reorder`

where `contract_list` can be a sublist of `['sushiswap', 'aave', 'uniswapv3', 'uniswapv2', 'uniswapv3-jit']` depending on the contracts and experiments involved.


`python3 runme.py` script automatically launches simulation clients (setup has to be called manually first) and runs an example optimization.

If you want to run simulation standalone, sample command:
`python3 simulate_client.py -f <concrete_transactions> -p <PORT_ID>` where `PORT_ID = <PORT of sim client> - 8601`



# Example files
concrete_transactions : `examples/optimised`

example_transactions : `examples/0xeth/multiple_tokens`

example_domain : `examples/0xeth/multiple_tokens_domain`
