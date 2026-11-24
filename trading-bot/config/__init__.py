"""
Config: trading bot parameters, asset universe, risk limits.
"""
# S-coin universe (high-volatility small/mid-cap)
S_COINS = [
    "SOL", "AVAX", "LINK", "ATOM", "NEAR", "FTM", "APT",
    "ARB", "OP", "INJ", "SEI", "TIA", "STRK", "WLD",
]

# Major pairs with options availability on Alpaca (equity index/ETF options)
OPTIONS_UNDERLYING = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META",
]

# Crypto for backtesting (no options, but good for grid/trading strategies)
CRYPTO_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD",
    "LINK-USD", "ATOM-USD", "NEAR-USD", "APT-USD",
    "ARB-USD", "OP-USD",
]

# Strategy configurations
STRATEGY_CONFIG = {
    "grid": {
        "grid_levels": 8,
        "grid_pct": 2.0,  # 2% between grid lines
        "trade_pct": 10,  # trade 10% of portfolio per grid
        "assets": CRYPTO_ASSETS,
    },
    "trend": {
        "fast_ma": 7,
        "slow_ma": 25,
        "atr_mult": 2.0,
        "assets": CRYPTO_ASSETS,
    },
    "options_put_backspread": {
        "strike_dist": 0.05,  # 5% OTM
        "long_strike_dist": 0.03,  # 3% OTM long
        "expiry_days": 30,
        "assets": OPTIONS_UNDERLYING,
    },
    "options_iron_condor": {
        "call_dist": 0.02,
        "put_dist": 0.02,
        "call_width": 0.015,
        "put_width": 0.015,
        "expiry_days": 30,
        "assets": OPTIONS_UNDERLYING,
    },
}

# Risk management
RISK = {
    "max_portfolio_risk_pct": 0.02,   # max 2% portfolio risk per trade
    "max_drawdown_pct": 0.15,         # 15% max drawdown
    "max_position_pct": 0.20,         # 20% max per position
    "max_correlation": 0.7,           # max correlation between positions
    "trailing_stop_pct": 0.10,        # 10% trailing stop
    "take_profit_pct": 0.20,          # 20% take profit
}

# GEV parameters for tail-risk modeling
GEV_PARAMS = {
    "shape_alpha": 3.0,  # tail thickness (smaller = fatter tails)
    "location_mu": 0.0,  # central location
    "scale_sigma": 1.0,  # scale parameter
    "threshold_p": 0.95,  # VaR threshold percentile
    "lookback": 252,      # 1 year of daily returns
}

# Paper trading defaults
PAPER_TRADING = True
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
