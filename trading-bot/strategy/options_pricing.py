"""
Black-Scholes pricing engine with implied volatility solver.
For options Greeks, PnL, and backtesting.
"""

import numpy as np
from scipy import stats
from typing import Tuple


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return stats.norm.cdf(x)


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return stats.norm.pdf(x)


def d1(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """BS d1 factor."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS d2 factor."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        # At expiry
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if option_type == "call":
        price = S * norm_cdf(d1_val) - K * np.exp(-r * T) * norm_cdf(d2_val)
    else:
        price = K * np.exp(-r * T) * norm_cdf(-d2_val) - S * norm_cdf(-d1_val)
    
    return max(price, 0)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for implied volatility using Newton-Raphson.
    
    Args:
        market_price: Observed market option price
        S, K, T, r: Same as black_scholes_price
        tol: Convergence tolerance
        max_iter: Maximum iterations
    """
    vol = 0.2  # Initial guess: 20%
    
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, vol, option_type)
        vega = vega_val(S, K, T, r, vol)
        
        diff = price - market_price
        
        if abs(diff) < tol:
            return vol
        
        if vega < 1e-10:
            break
        
        vol = vol - diff / vega
        vol = max(vol, 0.01)  # Floor at 1%
        vol = min(vol, 5.0)  # Cap at 500%
    
    return vol


def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Option delta."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    
    if option_type == "call":
        return norm_cdf(d1_val)
    else:
        return norm_cdf(d1_val) - 1


def gamma(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Option gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    return norm_pdf(d1_val) / (S * sigma * np.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Option theta (daily)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    term1 = -(S * norm_pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm_cdf(d2_val) if option_type == "call" else r * K * np.exp(-r * T) * norm_cdf(-d2_val)
    
    if option_type == "call":
        return (term1 - term2) / 365
    else:
        term2_neg = r * K * np.exp(-r * T) * norm_cdf(-d2_val)
        return (term1 + term2_neg) / 365


def vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Option vega (change per 1% vol change)."""
    return vega_val(S, K, T, r, sigma) * 0.01


def vega_val(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Raw vega (change per 1.0 vol change)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    return S * norm_pdf(d1_val) * np.sqrt(T)


def rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Option rho (change per 1% rate change)."""
    if T <= 0:
        return 0.0
    
    d2_val = d2(S, K, T, r, sigma)
    
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm_cdf(d2_val) / 100
    else:
        return -K * T * np.exp(-r * T) * norm_cdf(-d2_val) / 100


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """Compute all Greeks for an option."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "vega": vega(S, K, T, r, sigma),
        "rho": rho(S, K, T, r, sigma, option_type),
    }


def binomial_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
    steps: int = 100,
) -> float:
    """
    Binomial tree option pricing (Cox-Ross-Rubinstein).
    More accurate for American options.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    prices = []
    for i in range(steps + 1):
        price = S * (u ** i) * (d ** (steps - i))
        prices.append(price)
    
    # Terminal values
    if option_type == "call":
        values = [max(p - K, 0) for p in prices]
    else:
        values = [max(K - p, 0) for p in prices]
    
    # Backward induction
    for j in range(steps):
        new_values = []
        for i in range(len(values) - 1):
            val = np.exp(-r * dt) * (p * values[i] + (1 - p) * values[i + 1])
            if option_type == "call":
                val = max(val, max(prices[i] - K, 0))  # American exercise
            else:
                val = max(val, max(K - prices[i], 0))
            new_values.append(val)
        values = new_values
    
    return values[0]


def black_76_price(
    F: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Black-76 model for futures/forward options.
    Used for index options and commodity options.
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(F - K, 0)
        else:
            return max(K - F, 0)
    
    d1_val = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2_val = d1_val - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = np.exp(-r * T) * (F * norm_cdf(d1_val) - K * norm_cdf(d2_val))
    else:
        price = np.exp(-r * T) * (K * norm_cdf(-d2_val) - F * norm_cdf(-d1_val))
    
    return max(price, 0)


def options_pnl_simulation(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    n_scenarios: int = 10000,
) -> dict:
    """
    Simulate PnL distribution for an option position.
    Useful for VaR estimation and risk analysis.
    """
    np.random.seed(42)
    
    # Simulate spot prices at expiry
    daily_vol = sigma / np.sqrt(252)
    spot_at_expiry = spot * np.exp((r - 0.5 * sigma ** 2) * T + daily_vol * np.sqrt(252 * T) * np.random.standard_normal(n_scenarios))
    
    # Long call PnL
    call_payoff = np.maximum(spot_at_expiry - strike, 0)
    call_premium = black_scholes_price(spot, strike, T, r, sigma, "call")
    call_pnl = (call_payoff - call_premium) / call_premium
    
    # Long put PnL
    put_payoff = np.maximum(strike - spot_at_expiry, 0)
    put_premium = black_scholes_price(spot, strike, T, r, sigma, "put")
    put_pnl = (put_payoff - put_premium) / put_premium
    
    return {
        "call_pnl": {
            "mean": float(np.mean(call_pnl)),
            "median": float(np.median(call_pnl)),
            "p5": float(np.percentile(call_pnl, 5)),
            "p95": float(np.percentile(call_pnl, 95)),
            "var_1p": float(np.percentile(call_pnl, 1)),
            "es_1p": float(-np.mean(call_pnl[call_pnl <= np.percentile(call_pnl, 1)])),
        },
        "put_pnl": {
            "mean": float(np.mean(put_pnl)),
            "median": float(np.median(put_pnl)),
            "p5": float(np.percentile(put_pnl, 5)),
            "p95": float(np.percentile(put_pnl, 95)),
            "var_1p": float(np.percentile(put_pnl, 1)),
            "es_1p": float(-np.mean(put_pnl[put_pnl <= np.percentile(put_pnl, 1)])),
        },
    }


if __name__ == "__main__":
    # Demo: Black-Scholes pricing and Greeks
    S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.05, 0.20  # $100 spot, $105 strike, 3mo, 5% rate, 20% vol
    
    print("=== Black-Scholes Pricing ===")
    print(f"Call Price: ${black_scholes_price(S, K, T, r, sigma, 'call'):.2f}")
    print(f"Put Price:  ${black_scholes_price(S, K, T, r, sigma, 'put'):.2f}")
    
    greeks = compute_greeks(S, K, T, r, sigma, "call")
    print(f"\nGreeks (Call):")
    for k, v in greeks.items():
        print(f"  {k}: {v:.4f}")
    
    iv = implied_volatility(5.5, S, K, T, r, "call")
    print(f"\nImplied Vol (from $5.50): {iv:.2%}")
    
    print("\n=== Options PnL Simulation ===")
    pnl_dist = options_pnl_simulation(S, K, T, r, sigma)
    print(f"Long Call PnL - Mean: {pnl_dist['call_pnl']['mean']:.2%}, VaR(1%): {pnl_dist['call_pnl']['var_1p']:.2%}")
    print(f"Long Put PnL - Mean: {pnl_dist['put_pnl']['mean']:.2%}, VaR(1%): {pnl_dist['put_pnl']['var_1p']:.2%}")
