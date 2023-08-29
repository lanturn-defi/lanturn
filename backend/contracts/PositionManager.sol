// SPDX-License-Identifier: GPL-2.0-or-later
pragma solidity =0.7.6;
pragma abicoder v2;

import '@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol';
import './libraries/PoolAddress.sol';
import '@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3MintCallback.sol';
import './base/PeripheryPayments.sol';


contract PositionManager is IUniswapV3MintCallback, PeripheryPayments {
    
    struct MintCallbackData {
        PoolAddress.PoolKey poolKey;
        address payer;
    }

    constructor(
        address _factory,
        address _WETH9
    )  PeripheryImmutableState(_factory, _WETH9) {
        
    }

    function uniswapV3MintCallback(
        uint256 amount0Owed,
        uint256 amount1Owed,
        bytes calldata data
    ) external override {
        MintCallbackData memory decoded = abi.decode(data, (MintCallbackData));
        // CallbackValidation.verifyCallback(factory, decoded.poolKey);

        if (amount0Owed > 0) pay(decoded.poolKey.token0, decoded.payer, msg.sender, amount0Owed);
        if (amount1Owed > 0) pay(decoded.poolKey.token1, decoded.payer, msg.sender, amount1Owed);
    }

    // all new positions are owned by this contract, not the payer/sender of the mint transaction
    function mint(address token0, address token1, uint24 fee, uint128 liquidity, int24 tickLower, int24 tickDiff) external payable {
        PoolAddress.PoolKey memory poolKey = PoolAddress.getPoolKey(token0, token1, fee);
        IUniswapV3Pool pool = IUniswapV3Pool(PoolAddress.computeAddress(factory, poolKey));
        int24 tickUpper = tickLower + tickDiff;
        int24 tickSpacing = pool.tickSpacing();
        pool.mint(
            address(this),
            tickLower * tickSpacing,
            tickUpper * tickSpacing,
            liquidity,
            abi.encode(MintCallbackData({poolKey: poolKey, payer: msg.sender}))
        );
    }

    // anyone can burn this contract's positions and collect the payment
    function burnAndCollect(address token0, address token1, uint24 fee, uint128 liquidity, int24 tickLower, int24 tickDiff) external {
        PoolAddress.PoolKey memory poolKey = PoolAddress.getPoolKey(token0, token1, fee);
        IUniswapV3Pool pool = IUniswapV3Pool(PoolAddress.computeAddress(factory, poolKey));
        int24 tickUpper = tickLower + tickDiff;
        int24 tickSpacing = pool.tickSpacing();
        pool.burn(tickLower * tickSpacing, tickUpper * tickSpacing, liquidity);
        pool.collect(msg.sender, tickLower * tickSpacing, tickUpper * tickSpacing, uint128(-1), uint128(-1));
    }
}