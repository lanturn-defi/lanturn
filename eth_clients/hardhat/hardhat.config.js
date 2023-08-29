// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more

/**
 * @type import('hardhat/config').HardhatUserConfig
 */
module.exports = {
  solidity: "0.8.4",
  networks: {
  hardhat: {
    hardfork: "london",
    chainId: 1,
    mining: {
      auto: false,
      interval: 0,
      mempool: {
        order: "fifo"
      }
    },
    forking: {
      url: "http://localhost:8545"
    },
    loggingEnabled: false
  }
  },
};


