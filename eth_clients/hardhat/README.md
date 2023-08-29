44 Clients run on port 8601 - 8644

## setup all clients:
`bash setup_hardhat.sh`


## start all clients:
`bash launch_hardhats.sh`

## Increase linux priority for performance:
`sudo bash renice_hardhat.sh`

## Check if clients are running:
`bash ping_hardhats.sh` -> Ping clients by issuing a request for getting the current block number

If the client is running, it will return a json response with the "result"
If the client is stuck, the command will get stuck.
If the client is not running, output will say "refused to connect"


## kill all clients:
`bash kill_hardhat.sh` -> Stop all clients

## Check logs of clients:
eg. `tail -f logs/hardhat.8601`

## Check all clients are being used via the logging time stamps:
ls -al logs/
