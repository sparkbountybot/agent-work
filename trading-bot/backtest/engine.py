"""
Backtesting Engine — event-driven with realistic execution.
Supports: grid trading (crypto/S-coins), options strategies, multi-asset portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
import yfinance as yf


class BacktestResult:
    """Aggregates backtest results with performance metrics."""
    
    def __init__(self):
        self.equity: List[float] = []
        self.returns: List[float] = []
        self.trades: List[Dict] = []
        self.metrics: Dict[str, float] = {}
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.equity or len(self.equity) < 2:
            return {}
        
        equity_series = pd.Series(self.equity)
        returns_series = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        daily_vol = returns_series.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        risk_free_rate = 0.05
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        downside = returns_series[returns_series < 0]
        downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = (annualized_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        cummax = equity_series.cummax()
        drawdowns = (equity_series - cummax) / cummax
        max_drawdown = float(drawdowns.min())
        
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0
        
        gross_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
        
        var_95 = float(returns_series.quantile(0.05))
        es_95 = float(returns_series[returns_series <= var_95].mean()) if var_95 != 0 else 0
        
        self.metrics = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_drawdown,
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "var_95": var_95,
            "expected_shortfall_95": es_95,
            "total_trades": len(self.trades),
            "total_days": len(returns_series),
            "final_equity": float(self.equity[-1]),
        }
        
        return self.metrics


class DataEngine:
    """Fetches and processes market data for backtesting."""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_crypto(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Fetch cryptocurrency OHLCV data."""
        if symbol in self.cache:
            return self.cache[symbol]
        
        try:
            raw = yf.download(symbol, period=period, progress=False)
            
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw["Close"].copy()
            else:
                df = raw.copy()
            
            df.columns = pd.Index(["close"])
            df["return"] = df["close"].pct_change()
            
            self.cache[symbol] = df
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = df.copy()
        
        df["ma_7"] = df["close"].rolling(window=7).mean()
        df["ma_25"] = df["close"].rolling(window=25).mean()
        df["ma_50"] = df["close"].rolling(window=50).mean()
        df["ma_200"] = df["close"].rolling(window=200).mean()
        
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        bb_std = df["close"].rolling(window=20).std()
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        high_low = df["close"].rolling(window=14).max() - df["close"].rolling(window=14).min()
        df["atr"] = high_low / 14
        
        df["volatility_20d"] = df["return"].rolling(window=20).std() * np.sqrt(252)
        
        df = df.dropna()
        return df


class BacktestEngine:
    """
    Main backtesting engine.
    
    Features:
    - Event-driven simulation
    - Multiple strategy support
    - Realistic fees and slippage
    - Portfolio risk management
    - Walk-forward analysis
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.05,
        fee_rate: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.result = BacktestResult()
        self.data_engine = DataEngine()
    
    def run_backtest(
        self,
        strategy_func: Callable,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        **kwargs,
    ) -> BacktestResult:
        """Run backtest with the given strategy."""
        self.result = BacktestResult()
        
        equity_history = [self.initial_capital]
        all_trades = []
        
        for symbol in symbols:
            df = self.data_engine.fetch_crypto(symbol, period="5y")
            if df.empty or len(df) < 100:
                print(f"Skipping {symbol}: insufficient data")
                continue
            
            df = self.data_engine.calculate_indicators(df)
            
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            if len(df) < 10:
                print(f"Skipping {symbol}: insufficient filtered data")
                continue
            
            # Recalculate indicators on filtered data so rolling windows start fresh
            df = self.data_engine.calculate_indicators(df)
            if len(df) < 30:
                print(f"Skipping {symbol}: insufficient filtered data after indicators")
                continue
            
            # Track portfolio state
            portfolio_cash = self.initial_capital
            positions: Dict[str, Dict[str, float]] = {}
            
            for i, (date, row) in enumerate(df.iterrows()):
                price = row["close"]
                
                # Generate signal
                try:
                    signal = strategy_func(
                        symbol=symbol,
                        price=price,
                        df=df[:i+1],
                        date=date,
                        portfolio_value=portfolio_cash,
                        positions=positions,
                        **kwargs,
                    )
                except Exception:
                    signal = {"action": "hold"}
                
                if not signal:
                    signal = {"action": "hold"}
                
                action = signal.get("action", "hold")
                
                if action == "buy":
                    size = signal.get("size", 0.1)
                    entry_price = price * (1 + self.slippage_pct)
                    amount = portfolio_cash * size
                    
                    if amount > 0:
                        cost = amount * (1 + self.fee_rate)
                        qty = amount / entry_price
                        
                        if cost <= portfolio_cash:
                            portfolio_cash -= cost
                            
                            if symbol not in positions:
                                positions[symbol] = {"qty": 0.0, "avg_price": 0.0}
                            
                            old_qty = positions[symbol]["qty"]
                            old_avg = positions[symbol]["avg_price"]
                            
                            if old_qty > 0:
                                total_cost = old_avg * old_qty
                                positions[symbol]["avg_price"] = (total_cost + entry_price * qty) / (old_qty + qty)
                            else:
                                positions[symbol]["avg_price"] = entry_price
                            
                            positions[symbol]["qty"] += qty
                            
                            all_trades.append({
                                "symbol": symbol,
                                "entry_date": str(date.date()),
                                "entry_price": entry_price,
                                "qty": qty,
                                "action": "buy",
                                "pnl": 0,
                            })
                
                elif action == "sell":
                    if symbol in positions and positions[symbol]["qty"] > 0:
                        exit_price = price * (1 - self.slippage_pct)
                        qty = positions[symbol]["qty"]
                        
                        pnl = (exit_price - positions[symbol]["avg_price"]) * qty
                        proceeds = qty * exit_price * (1 - self.fee_rate)
                        portfolio_cash += proceeds
                        
                        entry_p = positions[symbol]["avg_price"]
                        positions[symbol]["qty"] = 0.0
                        
                        all_trades.append({
                            "symbol": symbol,
                            "exit_date": str(date.date()),
                            "exit_price": exit_price,
                            "entry_price": entry_p,
                            "action": "sell",
                            "pnl": pnl,
                            "qty": qty,
                        })
                
                # Track equity on EVERY day
                unrealized = 0.0
                for sym, pos in positions.items():
                    if sym == symbol and pos["qty"] > 0:
                        unrealized = (price - pos["avg_price"]) * pos["qty"]
                
                daily_equity = portfolio_cash + unrealized
                equity_history.append(daily_equity)
        
        # Store results
        self.result.trades = all_trades
        self.result.equity = equity_history
        
        return self.result
    
    def walk_forward(
        self,
        strategy_func: Callable,
        symbols: List[str],
        train_years: int = 2,
        test_years: int = 1,
        **kwargs,
    ) -> List[Dict]:
        """Walk-forward analysis to test strategy robustness."""
        results = []
        
        if not symbols:
            return results
        
        df = self.data_engine.fetch_crypto(symbols[0], period="max")
        if df.empty:
            return results
        
        start = df.index[0]
        end = df.index[-1]
        total_years = (end - start).days / 365.25
        num_windows = int(total_years / (train_years + test_years))
        
        for i in range(num_windows):
            train_start = start + pd.Timedelta(years=train_years * i)
            train_end = train_start + pd.Timedelta(years=train_years)
            test_start = train_end
            test_end = test_start + pd.Timedelta(years=test_years)
            
            if test_end > end:
                break
            
            print(f"Window {i+1}: Train [{train_start.date()} - {train_end.date()}], "
                  f"Test [{test_start.date()} - {test_end.date()}]")
            
            result = self.run_backtest(
                strategy_func=strategy_func,
                symbols=symbols,
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
                **kwargs,
            )
            
            results.append({
                "window": i + 1,
                "metrics": result.metrics,
            })
        
        return results


def grid_strategy(
    symbol: str,
    price: float,
    df: pd.DataFrame,
    date,
    portfolio_value: float,
    positions: Dict,
    **kwargs,
) -> Dict:
    """Simple grid/mean-reversion strategy for backtesting."""
    if len(df) < 30:
        return {"action": "hold"}
    
    ma_20 = df["ma_20"].iloc[-1] if "ma_20" in df.columns else 0
    bb_lower = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else 0
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    
    pos = positions.get(symbol, {})
    qty = pos.get("qty", 0)
    
    # Buy signal: price below MA20 or near BB lower, RSI oversold
    if rsi < 30 and price < ma_20 * 0.98:
        return {"action": "buy", "size": 0.2}
    
    # Sell signal: price above MA20, RSI overbought, or near BB upper
    if rsi > 70 and price > ma_20 * 1.02 and qty > 0:
        return {"action": "sell", "size": 1.0}
    
    # Trend signal: price breaking above MA20 with momentum
    if price > ma_20 * 1.05 and rsi > 50 and qty == 0:
        return {"action": "buy", "size": 0.15}
    
    # Exit if holding and price drops below BB lower
    if qty > 0 and price < bb_lower * 0.98:
        return {"action": "sell", "size": 1.0}
    
    return {"action": "hold"}


if __name__ == "__main__":
    print("=== Backtesting Engine Demo ===")
    
    engine = BacktestEngine(initial_capital=100000)
    
    result = engine.run_backtest(
        strategy_func=grid_strategy,
        symbols=["SOL-USD"],
        start_date="2024-01-01",
        end_date="2026-11-24",
    )
    
    print("\n=== Results ===")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            if key in ["total_return", "annualized_return", "max_drawdown",
                       "win_rate", "var_95", "expected_shortfall_95",
                       "annualized_volatility"]:
                print(f"  {key}: {value:.2%}")
            elif key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                         "profit_factor", "annualized_volatility"]:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
