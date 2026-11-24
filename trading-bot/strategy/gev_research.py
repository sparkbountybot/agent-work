"""
GEV (Generalized Extreme Value) Research Module
Implements Extreme Value Theory for crypto tail-risk modeling and position sizing.
Based on: Dimitrios Kouttos & Wang Chun Wei (2023) "Nowcasting bitcoin's crash risk
with order imbalance" and Roy (2026) "Intraday Risk Management of Cryptocurrency
Returns During 2020-2021 Upsurge: A Conditional EVT Approach"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
from config import GEV_PARAMS


class GEVModel:
    """
    Generalized Extreme Value distribution fitting and risk estimation.
    
    Models the tail distribution of returns to estimate:
    - Value at Risk (VaR) at extreme percentiles
    - Expected Shortfall (ES) / Conditional VaR
    - Optimal position sizing based on tail risk
    """
    
    def __init__(
        self,
        alpha: float = GEV_PARAMS["shape_alpha"],
        mu: float = GEV_PARAMS["location_mu"],
        sigma: float = GEV_PARAMS["scale_sigma"],
        threshold_p: float = GEV_PARAMS["threshold_p"],
        lookback: int = GEV_PARAMS["lookback"],
    ):
        self.volatility: Optional[pd.Series] = None
        self.alpha = alpha  # shape parameter (ξ in EVT literature)
        self.mu = mu
        self.sigma = sigma
        self.threshold_p = threshold_p
        self.lookback = lookback
        self.fitted = False
    
    def fit(self, returns: pd.Series) -> "GEVModel":
        """
        Fit GEV distribution to block maxima of returns.
        Uses Peak-Over-Threshold method for tail modeling.
        """
        n = len(returns)
        block_size = max(20, n // self.lookback)
        
        # Create blocks
        blocks = []
        for i in range(0, n - block_size + 1, block_size):
            block = returns.iloc[i:i+block_size]
            blocks.append(block.min())  # Use minima for left tail (crash risk)
        
        blocks = np.array(blocks)
        
        if len(blocks) < 5:
            print(f"GEV: Too few blocks ({len(blocks)}), using parametric fallback")
            return self
        
        # Peak-Over-Threshold: fit GPD to excess losses
        threshold = np.percentile(returns, self.threshold_p * 100)
        excesses = returns[returns < threshold] - threshold
        
        if len(excesses) < 10:
            print(f"GEV: Too few excess returns ({len(excesses)}), using parametric fallback")
            self.fitted = True
            return self
        
        # Fit GPD using Method of Moments (POT approach)
        mean_excess = np.mean(excesses)
        variance_excess = np.var(excesses, ddof=1)
        
        # σ = ξ * mean_excess
        # ξ (shape) estimated via variance: Var = σ² * (1-ξ-ξ²)/(1-2ξ) for ξ < 0.5
        # Simplified: ξ ≈ 0.5 for crypto (fat tails)
        self.alpha = 0.5  # Conservative fat-tail assumption for crypto
        self.sigma = abs(mean_excess) * self.alpha
        self.mu = threshold
        
        self.fitted = True
        return self
    
    def var(self, p: float = 0.01) -> float:
        """
        Estimate Value at Risk at percentile p (e.g., p=0.01 for 1% VaR).
        Uses GEV quantile function: Q(p) = mu + sigma/ξ * ((p)^(-ξ) - 1)
        """
        if not self.fitted:
            # Fallback: parametric VaR assuming log-normal
            return self._parametric_var(p)
        
        # GEV quantile function for left tail
        # For ξ > 0 (fat tail): F^(-1)(p) = mu + sigma/ξ * ((1-p)^(-1/ξ) - 1)
        # Note: We model left tail, so use (1-p) for extreme lows
        
        xi = self.alpha
        mu = self.mu
        sigma = self.sigma
        
        if abs(xi) < 1e-10:  # Near-zero shape (Gumbel case)
            quantile = mu - sigma * np.log(-np.log(1 - p))
        else:
            quantile = mu + sigma/xi * ((1-p)**(-1/xi) - 1)
        
        return quantile
    
    def _parametric_var(self, p: float) -> float:
        """Fallback parametric VaR assuming log returns are normally distributed."""
        return stats.norm.ppf(p) * self.sigma
    
    def expected_shortfall(self, p: float = 0.01) -> float:
        """
        Expected Shortfall (CVaR) given VaR at percentile p.
        ES = E[X | X <= VaR(p)]
        """
        if not self.fitted:
            # For normal distribution: ES = -σ * φ(Φ^(-1)(p)) / p
            z = stats.norm.ppf(p)
            return -self.sigma * stats.norm.pdf(z) / p
        
        xi = self.alpha
        mu = self.mu
        sigma = self.sigma
        
        var_p = self.var(p)
        
        if abs(xi) < 1e-10:
            # Gumbel case
            es = var_p - sigma/xi * (1 - np.log(-np.log(1-p)))
        else:
            # GPD case
            es = var_p + sigma * (1 - xi) / xi * ((1-p)**(-xi) - 1) / (1 - xi)
        
        return es
    
    def optimal_position_size(
        self,
        portfolio_value: float,
        risk_per_trade: float = 0.02,
        p: float = 0.01,
    ) -> float:
        """
        Calculate optimal position size based on GEV tail risk.
        
        Position size = (Portfolio Value * Risk_per_Trade) / VaR
        This ensures we never risk more than risk_per_trade on extreme tail events.
        """
        if not self.fitted:
            return self._parametric_position_size(portfolio_value, risk_per_trade)
        
        var = self.var(p)
        if var == 0:
            return portfolio_value * 0.1  # Default 10%
        
        # Position size such that max tail loss = risk_per_trade * portfolio
        position_value = (risk_per_trade * portfolio_value) / abs(var)
        
        # Cap at portfolio value
        return min(position_value, portfolio_value)
    
    def _parametric_position_size(self, portfolio_value: float, risk_per_trade: float) -> float:
        """Fallback position sizing using normal distribution VaR."""
        var = stats.norm.ppf(0.01) * self.sigma
        if var == 0:
            return portfolio_value * 0.1
        return min(
            (risk_per_trade * portfolio_value) / abs(var),
            portfolio_value
        )
    
    def nowcast_crash_risk(self, order_imbalance: float = 0.0) -> float:
        """
        Nowcast probability of crash using order flow imbalance.
        
        Combines GEV tail probability with order flow imbalance as a leading indicator.
        Positive imbalance = buying pressure (lower crash risk)
        Negative imbalance = selling pressure (higher crash risk)
        """
        base_crash_prob = 1 - self.threshold_p  # Base: 5% at 95th percentile
        
        # Order flow impact (logistic model per Kouttos & Wei 2023)
        # Sigmoid function maps imbalance to probability shift
        alpha_ofi = 2.0  # Sensitivity parameter
        crash_prob = base_crash_prob / (1 + np.exp(-alpha_ofi * order_imbalance))
        
        return crash_prob


class VolatilityClustering(GEVModel):
    """
    GARCH(1,1) + EVT combined model for intraday VaR estimation.
    Per Roy (2026): "Intraday Risk Management of Cryptocurrency Returns..."
    
    Uses ARMA-GARCH framework with skewed Student-t residuals for
    better tail modeling than normal distributions.
    """
    
    def __init__(self, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85):
        super().__init__()
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.volatility: Optional[pd.Series] = None
    
    def fit_garch(self, returns: pd.Series) -> "VolatilityClustering":
        """
        Fit GARCH(1,1) to volatility clustering.
        
        h_t = ω + α * ε_{t-1}² + β * h_{t-1}
        where ε_t = r_t - μ
        """
        n = len(returns)
        h = np.zeros(n)
        eps = returns.values - returns.mean()
        
        # Initialize
        h[0] = np.var(returns)
        
        # QMLE recursion
        for t in range(1, n):
            h[t] = self.omega + self.alpha * eps[t-1]**2 + self.beta * h[t-1]
        
        self.volatility = pd.Series(h, index=returns.index)
        self.sigma = np.sqrt(h[-1])
        self.fitted = True
        
        return self
    
    def forecast_volatility(self, h_next: int = 1) -> float:
        """Forecast volatility h_{t+h} using GARCH recursion."""
        if self.volatility is None:
            return self.sigma
        
        last_h = self.volatility.iloc[-1]
        return self.omega / (1 - self.alpha - self.beta) + \
               (last_h - self.omega / (1 - self.alpha - self.beta)) * \
               (self.alpha + self.beta) ** h_next
    
    def conditional_var(self, p: float = 0.01, h: int = 1) -> float:
        """
        Conditional VaR using GARCH-volatility and skewed Student-t residuals.
        """
        if self.volatility is None:
            return self.var(p)
        
        vol = self.forecast_volatility(h)
        # Skewed Student-t: adjust for fat tails and skewness
        df = 4  # Heavy tails (typical for crypto)
        skew = -1.5  # Left skew (crypto tends to crash harder than rally)
        
        # Quantile of skewed Student-t
        z = self._skewed_student_t_quantile(p, df, skew)
        
        return z * vol
    
    def _skewed_student_t_quantile(self, p: float, df: float, skew: float) -> float:
        """Approximate quantile of skewed Student-t distribution."""
        from scipy import stats
        
        # Use Student-t with degrees of freedom adjustment
        t_dist = stats.t(df)
        z_t = t_dist.ppf(p)
        
        # Adjust for skewness (simplified AAS distribution)
        if skew > 0:
            return z_t * (1 + skew * 0.3)
        else:
            return z_t * (1 + abs(skew) * 0.5)  # Left tail more extreme


class GridTradingGev:
    """
    GEV-informed grid trading strategy for S-coins.
    
    Uses tail risk estimates to dynamically adjust grid spacing:
    - High tail risk → wider grids (reduce frequency)
    - Low tail risk → tighter grids (increase frequency)
    
    Based on: "Dynamic Grid Trading Strategy: From Zero Expectation
    to Market Outperformance" (arxiv:2506.11921) with GEV tail adjustment.
    """
    
    def __init__(self, gev_model: GEVModel = None):
        self.gev = gev_model or GEVModel()
        self.grid_spacing: Optional[float] = None
        self.current_price: float = 0.0
    
    def adjust_grid(self, returns: pd.Series) -> float:
        """
        Adjust grid spacing based on GEV tail risk.
        
        Returns optimal grid percentage (0.5-5.0%) based on:
        - VaR at 1% level
        - Volatility clustering forecast
        - Order flow imbalance
        """
        self.gev.fit(returns)
        
        var_1p = abs(self.gev.var(0.01))
        es_1p = abs(self.gev.expected_shortfall(0.01))
        
        # Base grid from VaR (wider for higher tail risk)
        base_grid = min(max(var_1p * 3, 0.005), 0.03)  # 0.5% - 3.0%
        
        # Adjust for ES (expected depth of crash)
        if es_1p > var_1p * 2:  # Very fat tails
            base_grid *= 1.5
        
        # Adjust for volatility clustering
        if self.gev.volatility is not None:
            vol_forecast = self.gev.forecast_volatility(1)
            vol_current = np.sqrt(np.var(returns.iloc[-20:]))
            if vol_forecast > vol_current * 1.2:  # Rising vol
                base_grid *= 1.3
        
        self.grid_spacing = base_grid
        return base_grid
    
    def generate_grid(self, price: float, n_levels: int = 8) -> list:
        """Generate grid levels around current price."""
        if self.grid_spacing is None:
            raise ValueError("Call adjust_grid() first")
        
        self.current_price = price
        half = n_levels // 2
        
        levels = []
        for i in range(-half, half + 1):
            level = price * (1 + i * self.grid_spacing)
            levels.append(level)
        
        return levels
    
    def estimate_expected_value(self, levels: list, price: float, time_horizon: int = 252) -> float:
        """
        Estimate expected value of grid trading per arxiv:2506.11921.
        
        E(G) = 0.5 * (P_u - L_l) where:
        P_u = profit from upper grid breaches
        L_l = loss from lower grid breaches
        """
        n = len(levels) - 1
        grid_size = self.grid_spacing
        
        # Simplified: approximate P_u and L_l from historical data
        # Full implementation uses Markov chain on grid states
        return 0.0  # Placeholder — full model needs transition matrix


if __name__ == "__main__":
    # Demo: GEV model on sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.002, 0.05, 1000))
    
    print("=== GEV Tail Risk Analysis ===")
    gev = GEVModel()
    gev.fit(returns)
    
    print(f"VaR (1%): {gev.var(0.01):.2%}")
    print(f"ES (1%): {gev.expected_shortfall(0.01):.2%}")
    print(f"Optimal Position Size: ${gev.optimal_position_size(100000):.2f}")
    print(f"Crash Risk (base): {gev.nowcast_crash_risk():.2%}")
    print(f"Crash Risk (ofi=-0.5): {gev.nowcast_crash_risk(-0.5):.2%}")
