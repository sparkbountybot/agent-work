"""
Grid/mean-reversion trading strategy for backtesting.
Designed for crypto/S-coin volatility with proper position lifecycle.
"""

import pandas as pd
import numpy as np
from typing import Dict


def grid_strategy(
    symbol: str,
    price: float,
    df: pd.DataFrame,
    date,
    portfolio_value: float,
    positions: Dict,
    **kwargs,
) -> Dict:
    """
    Mean-reversion strategy: buy at RSI oversold/below MA, sell at RSI overbought/above MA.
    
    Key design:
    - ONE position per symbol at a time
    - Sell before re-entry
    - Take-profit + stop-loss
    - Fee-aware sizing (size decreases as portfolio depletes from fees)
    """
    if len(df) < 30:
        return {"action": "hold"}
    
    # Calculate indicators on this slice
    ma_20 = df["ma_20"].iloc[-1] if "ma_20" in df.columns else 0
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    bb_lower = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else 0
    bb_upper = df["bb_upper"].iloc[-1] if "bb_upper" in df.columns else 0
    
    # MACD signal
    macd_hist = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0
    
    pos = positions.get(symbol, {})
    qty = pos.get("qty", 0.0)
    avg_price = pos.get("avg_price", 0.0)
    
    # ===== SELL CONDITIONS (must happen before any buy) =====
    if qty > 0:
        current_pnl_pct = (price - avg_price) / avg_price if avg_price > 0 else 0
        
        # 1. Take profit: +10% gain
        if current_pnl_pct >= 0.10:
            return {"action": "sell", "size": 1.0}
        
        # 2. Stop loss: -5% loss
        if current_pnl_pct <= -0.05:
            return {"action": "sell", "size": 1.0}
        
        # 3. RSI overbought exit
        if rsi > 75:
            return {"action": "sell", "size": 1.0}
        
        # 4. Price above BB upper (reversion play complete)
        if price > bb_upper * 0.98 and rsi > 55:
            return {"action": "sell", "size": 0.5}  # Partial exit
        
        # 5. MACD turns negative while holding
        if macd_hist < 0 and current_pnl_pct > 0.02:
            return {"action": "sell", "size": 0.5}
    
    # ===== BUY CONDITIONS =====
    if qty == 0 and portfolio_value > 1000:
        # 1. RSI oversold + price below MA
        if rsi < 25 and ma_20 > 0:
            return {"action": "buy", "size": 0.3}
        
        # 2. MACD turning positive + price near BB lower
        if macd_hist > 0 and price < bb_lower * 1.02:
            return {"action": "buy", "size": 0.2}
        
        # 3. Extreme RSI dip (panic buy)
        if rsi < 20:
            return {"action": "buy", "size": 0.25}
    
    return {"action": "hold"}


def grid_strategy_aggressive(
    symbol: str,
    price: float,
    df: pd.DataFrame,
    date,
    portfolio_value: float,
    positions: Dict,
    **kwargs,
) -> Dict:
    """
    Aggressive grid strategy with more frequent trading.
    Uses tighter RSI bands and faster entry/exit.
    """
    if len(df) < 20:
        return {"action": "hold"}
    
    ma_7 = df["ma_7"].iloc[-1] if "ma_7" in df.columns else 0
    ma_25 = df["ma_25"].iloc[-1] if "ma_25" in df.columns else 0
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    bb_lower = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else 0
    bb_upper = df["bb_upper"].iloc[-1] if "bb_upper" in df.columns else 0
    macd_hist = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0
    
    pos = positions.get(symbol, {})
    qty = pos.get("qty", 0.0)
    avg_price = pos.get("avg_price", 0.0)
    
    # SELL
    if qty > 0:
        current_pnl_pct = (price - avg_price) / avg_price if avg_price > 0 else 0
        
        # Quick take profit: +5%
        if current_pnl_pct >= 0.05:
            return {"action": "sell", "size": 1.0}
        
        # Quick stop: -3%
        if current_pnl_pct <= -0.03:
            return {"action": "sell", "size": 1.0}
        
        # RSI extreme exit
        if rsi > 80 or rsi < 15:
            return {"action": "sell", "size": 1.0}
    
    # BUY
    if qty == 0:
        # RSI oversold
        if rsi < 30 and ma_25 > 0:
            return {"action": "buy", "size": 0.25}
        
        # MACD crossover + price pullback
        if macd_hist > 0 and price < ma_7 * 0.99:
            return {"action": "buy", "size": 0.2}
        
        # Touching BB lower
        if price <= bb_lower:
            return {"action": "buy", "size": 0.2}
    
    return {"action": "hold"}


def trend_strategy(
    symbol: str,
    price: float,
    df: pd.DataFrame,
    date,
    portfolio_value: float,
    positions: Dict,
    **kwargs,
) -> Dict:
    """
    Trend-following strategy: buy breakouts, sell reversals.
    Uses EMA crossover (7/25) with RSI confirmation.
    """
    if len(df) < 30:
        return {"action": "hold"}
    
    ema_7 = df["ma_7"].iloc[-1] if "ma_7" in df.columns else 0
    ema_25 = df["ma_25"].iloc[-1] if "ma_25" in df.columns else 0
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    
    pos = positions.get(symbol, {})
    qty = pos.get("qty", 0.0)
    avg_price = pos.get("avg_price", 0.0)
    
    if qty > 0:
        current_pnl_pct = (price - avg_price) / avg_price if avg_price > 0 else 0
        
        # Exit: EMA7 crosses below EMA25
        if ema_7 < ema_25:
            return {"action": "sell", "size": 1.0}
        
        # Stop loss
        if current_pnl_pct <= -0.05:
            return {"action": "sell", "size": 1.0}
        
        # Take profit: +20%
        if current_pnl_pct >= 0.20:
            return {"action": "sell", "size": 1.0}
    
    if qty == 0:
        # Enter: EMA7 crosses above EMA25 + RSI not overbought
        if ema_7 > ema_25 and rsi > 40 and rsi < 70:
            return {"action": "buy", "size": 0.3}
    
    return {"action": "hold"}


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import yfinance as yf
    import pandas as pd
    from backtest.engine import BacktestEngine
    
    raw = yf.download('SOL-USD', period='6mo', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'].copy()
    else:
        df = raw.copy()
    df.columns = pd.Index(['close'])
    df['return'] = df['close'].pct_change()
    
    # Calculate indicators
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_25'] = df['close'].rolling(window=25).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    bb_std = df['close'].rolling(window=20).std()
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df = df.dropna()
    
    # Test all 3 strategies
    strategies = {
        'grid': grid_strategy,
        'grid_aggressive': grid_strategy_aggressive,
        'trend': trend_strategy,
    }
    
    for name, strategy_func in strategies.items():
        engine = BacktestEngine(initial_capital=100000, fee_rate=0.001, slippage_pct=0.0005)
        result = engine.run_backtest(
            strategy_func=strategy_func,
            symbols=['SOL-USD'],
            start_date='2025-06-01',
            end_date='2026-05-31',
        )
        m = result.metrics
        print(f'\n=== {name.upper()} ===')
        print(f'  Equity: ${m.get("final_equity", 0):,.2f}')
        print(f'  Trades: {m.get("total_trades", 0)}')
        print(f'  Return: {m.get("total_return", 0):.2%}')
        print(f'  Sharpe: {m.get("sharpe_ratio", 0):.2f}')
        print(f'  Max DD: {m.get("max_drawdown", 0):.2%}')
        print(f'  Win Rate: {m.get("win_rate", 0):.2%}')
