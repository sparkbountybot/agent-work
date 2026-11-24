"""
Options Strategies for Alpaca Trading API.
Implements: Bull Call Spread, Iron Condor, Iron Butterfly, Covered Call, Protective Put.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from config import OPTIONS_UNDERLYING, STRATEGY_CONFIG


class OptionLeg:
    """A single option contract leg."""
    
    def __init__(self, symbol: str, side: str, strike: float, premium: float, expiry: str):
        self.symbol = symbol
        self.side = side  # "buy" or "sell"
        self.strike = strike
        self.premium = premium  # credit if sell, debit if buy
        self.expiry = expiry
        self.pnl = 0.0
        self.open = True
        self.greeks: Dict[str, float] = {}
    
    def mark_to_market(self, underlying_price: float, volatility: float, days_to_expiry: int) -> float:
        """Mark leg to current market price."""
        from .options_pricing import black_scholes_price
        
        if self.side == "sell":
            # Short position: premium is credit received
            current_price = black_scholes_price(
                underlying_price, self.strike, days_to_expiry / 365,
                volatility, "call" if "C" in self.symbol else "put"
            )
            self.pnl = (self.premium - current_price) / self.premium if self.premium > 0 else 0
        else:
            # Long position: premium is debit paid
            current_price = black_scholes_price(
                underlying_price, self.strike, days_to_expiry / 365,
                volatility, "call" if "C" in self.symbol else "put"
            )
            self.pnl = (current_price - self.premium) / self.premium if self.premium > 0 else 0
        
        return self.pnl


class OptionPosition:
    """A multi-leg options position."""
    
    def __init__(self, legs: List[OptionLeg]):
        self.legs = legs
        self.net_premium = sum(l.premium if l.side == "sell" else -l.premium for l in legs)
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.pnl = 0.0
        self.open = True
        self.type: str = "unknown"
    
    def identify_strategy(self, underlying_price: float) -> str:
        """Identify the options strategy from the legs."""
        calls = [l for l in self.legs if "C" in l.symbol]
        puts = [l for l in self.legs if "P" in l.symbol]
        
        long_calls = [l for l in calls if l.side == "buy"]
        short_calls = [l for l in calls if l.side == "sell"]
        long_puts = [l for l in puts if l.side == "buy"]
        short_puts = [l for l in puts if l.side == "sell"]
        
        if len(long_calls) == 1 and len(short_calls) == 1:
            self.type = "bull_call_spread"
            self.max_profit = (
                short_calls[0].strike - long_calls[0].strike - self.net_premium
            )
            self.max_loss = self.net_premium
        elif len(short_calls) == 1 and len(short_puts) == 1 and len(long_calls) <= 1 and len(long_puts) <= 1:
            if len(long_calls) == 1 and len(long_puts) == 1:
                self.type = "iron_condor"
                call_width = short_calls[0].strike - long_calls[0].strike
                put_width = long_puts[0].strike - short_puts[0].strike
                self.max_profit = self.net_premium
                self.max_loss = max(call_width, put_width) - self.net_premium
            elif len(long_calls) == 0 and len(long_puts) == 0:
                self.type = "short_straddle"
                self.max_profit = self.net_premium
                self.max_loss = float("inf")  # Theoretical unlimited loss
            else:
                self.type = "iron_butterfly"
                self.max_profit = self.net_premium
                self.max_loss = float("inf")
        elif len(short_calls) == 1 and len(long_puts) == 1:
            self.type = "protective_put"
            self.max_profit = float("inf")  # Unlimited upside
            self.max_loss = short_calls[0].strike - long_puts[0].strike + self.net_premium
        elif len(short_puts) == 1 and len(long_calls) == 1:
            self.type = "protective_call"
            self.max_profit = long_calls[0].strike - self.net_premium
            self.max_loss = float("inf")
        else:
            self.type = "unknown"
        
        return self.type
    
    def mark_to_market(self, underlying_price: float, volatility: float, days_to_expiry: int) -> float:
        """Mark entire position to market."""
        leg_pnls = [l.mark_to_market(underlying_price, volatility, days_to_expiry) for l in self.legs]
        self.pnl = sum(leg_pnls) / len(leg_pnls)
        return self.pnl
    
    def check_exits(self, underlying_price: float, days_to_expiry: int) -> Tuple[bool, str]:
        """Check if position should be closed."""
        if days_to_expiry < 0:
            return True, "expired"
        
        # Close at 80% of max profit
        if self.pnl > 0.8:
            return True, "profit_target"
        
        # Stop loss at -50% of max loss
        if self.max_loss > 0 and self.pnl < -0.5 * abs(self.max_loss):
            return True, "stop_loss"
        
        # Time decay: close 1 day before expiry
        if days_to_expiry <= 1:
            return True, "time_decay"
        
        return False, ""


class OptionsStrategyEngine:
    """
    Manages options positions and generates strategy signals.
    
    Supports:
    - Bull Call Spread: directional bullish
    - Iron Condor: neutral, profits from time decay
    - Iron Butterfly: neutral, higher risk/reward
    - Covered Call: income generation
    - Protective Put: portfolio insurance
    """
    
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.positions: List[OptionPosition] = []
        self.equity_curve: List[float] = []
        self.total_premium_collected = 0.0
        self.total_premium_paid = 0.0
        self.win_rate = 0.0
        self.total_exits = 0
        self.winning_exits = 0
    
    def find_bull_call_spread(
        self,
        underlying: str,
        current_price: float,
        option_chains: List[Dict],
        expiry_days: int = 30,
    ) -> Optional[OptionPosition]:
        """
        Find optimal bull call spread.
        
        Rules:
        - Buy call: ATM or slightly ITM (95-100% of spot)
        - Sell call: OTM (105-110% of spot)
        - Target: credit ≤ 50% of spread width
        """
        config = STRATEGY_CONFIG["options_put_backspread"]
        
        call_options = [
            o for o in option_chains
            if "C" in o.get("contractType", "") and o.get("expirationDate", "").startswith("20")
        ]
        
        # Filter by expiry
        valid_expiries = [
            o for o in call_options
            if expiry_days - 5 <= _days_to_expiry(o["expirationDate"]) <= expiry_days + 5
        ]
        
        if not valid_expiries:
            return None
        
        # Find strikes
        buy_strike_idx = int(len(valid_expiries) * 0.5)  # Near ATM
        sell_strike_idx = int(len(valid_expiries) * 0.75)  # OTM
        
        if buy_strike_idx >= len(valid_expiries) or sell_strike_idx >= len(valid_expiries):
            return None
        
        buy_strike = valid_expiries[buy_strike_idx].get("strikePrice", 0)
        sell_strike = valid_expiries[sell_strike_idx].get("strikePrice", 0)
        
        if sell_strike <= buy_strike:
            return None
        
        spread_width = sell_strike - buy_strike
        
        # Estimate premiums from bid/ask
        buy_premium = (
            valid_expiries[buy_strike_idx].get("bid", 0) +
            valid_expiries[buy_strike_idx].get("ask", 0)
        ) / 2
        
        sell_premium = (
            valid_expiries[sell_strike_idx].get("bid", 0) +
            valid_expiries[sell_strike_idx].get("ask", 0)
        ) / 2
        
        net_debit = buy_premium - sell_premium
        
        # Only enter if debit is reasonable (< 50% of spread width)
        if net_debit > spread_width * 0.5:
            return None
        
        legs = [
            OptionLeg(
                symbol=valid_expiries[buy_strike_idx].get("symbol", ""),
                side="buy",
                strike=buy_strike,
                premium=buy_premium,
                expiry=valid_expiries[buy_strike_idx]["expirationDate"],
            ),
            OptionLeg(
                symbol=valid_expiries[sell_strike_idx].get("symbol", ""),
                side="sell",
                strike=sell_strike,
                premium=sell_premium,
                expiry=valid_expiries[sell_strike_idx]["expirationDate"],
            ),
        ]
        
        position = OptionPosition(legs)
        position.type = "bull_call_spread"
        position.max_profit = spread_width - net_debit
        position.max_loss = net_debit
        
        self.positions.append(position)
        return position
    
    def find_iron_condor(
        self,
        underlying: str,
        current_price: float,
        option_chains: List[Dict],
        expiry_days: int = 30,
    ) -> Optional[OptionPosition]:
        """
        Find optimal iron condor.
        
        Structure:
        - Sell call at +2% OTM
        - Buy call at +3.5% OTM (protect)
        - Sell put at -2% OTM
        - Buy put at -3.5% OTM (protect)
        
        Target: credit > 25% of total width
        """
        calls = [o for o in option_chains if "C" in o.get("contractType", "")]
        puts = [o for o in option_chains if "P" in o.get("contractType", "")]
        
        # Filter by expiry
        valid_calls = [o for o in calls if _days_to_expiry(o["expirationDate"]) <= expiry_days]
        valid_puts = [o for o in puts if _days_to_expiry(o["expirationDate"]) <= expiry_days]
        
        if not valid_calls or not valid_puts:
            return None
        
        # Find strikes
        sell_call = _find_nearest_strike(valid_calls, current_price * 1.02)
        buy_call = _find_nearest_strike(valid_calls, current_price * 1.035)
        sell_put = _find_nearest_strike(valid_puts, current_price * 0.98)
        buy_put = _find_nearest_strike(valid_puts, current_price * 0.965)
        
        if not all([sell_call, buy_call, sell_put, buy_put]):
            return None
        
        # Build legs
        legs = [
            OptionLeg(sell_call["symbol"], "sell", float(sell_call["strikePrice"]),
                     _get_mid_price(sell_call), sell_call["expirationDate"]),
            OptionLeg(buy_call["symbol"], "buy", float(buy_call["strikePrice"]),
                     _get_mid_price(buy_call), buy_call["expirationDate"]),
            OptionLeg(sell_put["symbol"], "sell", float(sell_put["strikePrice"]),
                     _get_mid_price(sell_put), sell_put["expirationDate"]),
            OptionLeg(buy_put["symbol"], "buy", float(buy_put["strikePrice"]),
                     _get_mid_price(buy_put), buy_put["expirationDate"]),
        ]
        
        position = OptionPosition(legs)
        position.identify_strategy(current_price)
        
        self.positions.append(position)
        return position
    
    def simulate_options(self, underlying_price: float, days_to_expiry: int) -> List[Dict]:
        """
        Simulate options PnL across strike grid.
        Returns PnL data for each strategy type.
        """
        from .options_pricing import black_scholes_price, implied_volatility
        
        results = {
            "bull_call_spread": [],
            "iron_condor": [],
            "iron_butterfly": [],
        }
        
        # Simulate price paths
        np.random.seed(42)
        daily_vol = 0.02  # 2% daily vol for simulation
        returns = np.random.normal(0, daily_vol, days_to_expiry)
        prices = [underlying_price * np.exp(sum(returns[:i])) for i in range(days_to_expiry)]
        
        for price in prices:
            # Bull Call Spread PnL
            bcs_pnl = self._calc_bull_call_spread_pnl(price, underlying_price, days_to_expiry)
            results["bull_call_spread"].append(bcs_pnl)
            
            # Iron Condor PnL
            ic_pnl = self._calc_iron_condor_pnl(price, underlying_price, days_to_expiry)
            results["iron_condor"].append(ic_pnl)
            
            # Iron Butterfly PnL
            ib_pnl = self._calc_iron_butterfly_pnl(price, underlying_price, days_to_expiry)
            results["iron_butterfly"].append(ib_pnl)
        
        return results
    
    def _calc_bull_call_spread_pnl(self, price: float, entry_price: float, dte: int) -> float:
        """Calculate bull call spread PnL at price level."""
        # Simulate typical spread: buy at 98%, sell at 104%
        buy_strike = entry_price * 0.98
        sell_strike = entry_price * 1.04
        
        max_profit = (sell_strike - buy_strike) * 100  # Per contract
        max_loss = 1000  # Fixed debit for simulation
        
        if price <= buy_strike:
            return -max_loss
        elif price >= sell_strike:
            return max_profit
        else:
            return (price - buy_strike) / (sell_strike - buy_strike) * max_profit - max_loss
    
    def _calc_iron_condor_pnl(self, price: float, entry_price: float, dte: int) -> float:
        """Calculate iron condor PnL at price level."""
        # Simulate typical IC: 2% OTM each side
        short_call_strike = entry_price * 1.02
        buy_call_strike = entry_price * 1.035
        short_put_strike = entry_price * 0.98
        buy_put_strike = entry_price * 0.965
        
        call_width = buy_call_strike - short_call_strike
        put_width = short_put_strike - buy_put_strike
        max_width = max(call_width, put_width)
        
        credit_pct = 0.30  # 30% of width as credit
        max_profit = max_width * 100 * credit_pct
        max_loss = (max_width - max_width * credit_pct) * 100
        
        if buy_put_strike <= price <= short_call_strike:
            return max_profit
        elif price < buy_put_strike:
            return -max_loss + (buy_put_strike - price) / put_width * max_profit
        elif price > short_call_strike:
            return -max_loss + (price - short_call_strike) / call_width * max_profit
        else:
            return 0  # In credit zone but not max yet
    
    def _calc_iron_butterfly_pnl(self, price: float, entry_price: float, dte: int) -> float:
        """Calculate iron butterfly PnL at price level."""
        # Simulate typical IB: sell ATM, buy 3% OTM
        short_strike = entry_price
        long_call_strike = entry_price * 1.03
        long_put_strike = entry_price * 0.97
        
        max_width = (short_strike - long_put_strike)
        credit_pct = 0.40  # 40% of width as credit
        max_profit = max_width * 100 * credit_pct
        max_loss = (max_width - max_width * credit_pct) * 100
        
        if price == short_strike:
            return max_profit
        elif price < long_put_strike or price > long_call_strike:
            return -max_loss
        elif price < short_strike:
            return max_profit - (short_strike - price) / (short_strike - long_put_strike) * (max_profit + max_loss)
        else:
            return max_profit - (price - short_strike) / (long_call_strike - short_strike) * (max_profit + max_loss)
    
    def get_position_summary(self) -> Dict:
        """Get summary of all open positions."""
        open_positions = [p for p in self.positions if p.open]
        
        return {
            "open_positions": len(open_positions),
            "total_exits": self.total_exits,
            "winning_exits": self.winning_exits,
            "win_rate": self.winning_exits / self.total_exits if self.total_exits > 0 else 0,
            "positions": [
                {
                    "type": p.type,
                    "legs": len(p.legs),
                    "pnl": p.pnl,
                    "max_profit": p.max_profit,
                    "max_loss": p.max_loss,
                }
                for p in self.positions
            ],
        }


def _days_to_expiry(expiration_date: str) -> int:
    """Calculate days to expiry from expiration date string."""
    try:
        from datetime import datetime, date
        # Handle various date formats
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
            try:
                expiry = datetime.strptime(expiration_date[:10], "%Y-%m-%d")
                return max(0, (expiry - date.today()).days)
            except ValueError:
                continue
        return 30  # Default fallback
    except Exception:
        return 30


def _find_nearest_strike(options: List[Dict], target_strike: float) -> Optional[Dict]:
    """Find the option contract nearest to the target strike."""
    if not options:
        return None
    
    strikes = [float(o.get("strikePrice", float("inf"))) for o in options]
    nearest_idx = np.argmin(np.abs(np.array(strikes) - target_strike))
    return options[nearest_idx]


def _get_mid_price(option: Dict) -> float:
    """Get mid price from bid/ask."""
    bid = option.get("bid", 0) or 0
    ask = option.get("ask", 0) or 0
    return (bid + ask) / 2
