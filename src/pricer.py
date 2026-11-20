"""
Monte Carlo option pricing engine.

Model: risk-neutral Geometric Brownian Motion, zero dividends
    dS_t = r S_t dt + sigma S_t dW_t

Conventions (write these in the notebook's "Assumptions" section too):
    - 252 trading days per year.
    - expiry_days < 30  -> 0.5-day time steps; otherwise 1-day steps.
    - `paths` has shape (n_simulations, n_steps) and does NOT include S0
      (column 0 is the price after the first step, last column is S_T).
    - Barriers are checked discretely at every simulated time step.
    - Asian options use the arithmetic average of the simulated prices.
"""

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


# =============================================================================
# 1. TIME GRID  (block 1)
# =============================================================================
def get_time_grid(expiry_days, step_days=None):
    """
    Work out the time step and number of steps for a given expiry.

    Returns
    -------
    dt      : step size in years
    n_steps : number of simulated steps
    T       : time to expiry in years

    Checks: 252 days -> 252 steps, 30 days -> 30 steps, 20 days -> 40 steps
    """
    if step_days is None:
        if expiry_days < 30:
            step_days = 0.5
        else:
            step_days = 1.0

    n_steps = int(round(expiry_days / step_days))
    dt = step_days / TRADING_DAYS
    T = expiry_days / TRADING_DAYS
    return dt, n_steps, T


# =============================================================================
# 2. GBM PATH SIMULATION  (block 1, antithetic option in block 4)
# =============================================================================
def simulate_gbm_paths(S0, r, sigma, expiry_days, n_simulations,
                       seed=None, antithetic=False, step_days=None):
    """
    Simulate risk-neutral GBM price paths, fully vectorised (no loop over paths).

    Returns
    -------
    paths : np.ndarray of shape (n_simulations, n_steps)

    Checks: shape is (n_simulations, n_steps)
            mean of last column ~ S0 * exp(r * T)   (100 -> ~105.13 for 1y, r=5%)
            same seed -> identical output
    """
    dt, n_steps, T = get_time_grid(expiry_days, step_days)
    rng = np.random.default_rng(seed)

    if antithetic:                      # pair every Z with -Z
        half = n_simulations // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z_half, -Z_half])
    else:
        Z = rng.standard_normal((n_simulations, n_steps))

    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_path = np.cumsum(log_increments, axis=1)
    S_n = S0 * np.exp(log_path)
    return S_n


# =============================================================================
# 3. MONTE CARLO STATISTICS  (block 2)
# =============================================================================
def mc_summary(discounted_payoffs, confidence=0.95):
    """
    Turn one discounted payoff per path into a price with its uncertainty.

    Returns
    -------
    dict with keys: "price", "std_error", "ci_low", "ci_high"

    Formulas: price = mean, std_error = std / sqrt(N),
              CI = price +/- z * std_error   (z = 1.96 for 95%)
    """
    price = np.mean(discounted_payoffs)
    n = len(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n)
    Confidence_Interval = norm.ppf(0.5 + confidence / 2)

    return {
        "price": price,
        "std_error": std_error,
        "ci_low": price - Confidence_Interval * std_error,
        "ci_high": price + Confidence_Interval * std_error,
    }


# =============================================================================
# 4. BLACK-SCHOLES BENCHMARK  (block 2)
# =============================================================================
def black_scholes(S0, K, T, r, sigma, option_type="call"):
    """
    Closed-form European option price (no dividends). Used ONLY to validate MC.

    Check: S0=K=100, T=1, r=0.05, sigma=0.20 -> call ~ 10.45, put ~ 5.57
    """
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")


# =============================================================================
# 5. PAYOFFS  (blocks 2-3)
# =============================================================================
def vanilla_payoff(prices, K, option_type="call"):
    """
    Call: max(price - K, 0)   Put: max(K - price, 0)
    `prices` is a 1-D array (one value per path). Use np.maximum, not max.
    """
    if option_type == "call":
        return np.maximum(prices - K, 0)
    elif option_type == "put":
        return np.maximum(K - prices, 0)


def european_payoff(paths, K, option_type="call"):
    """
    Payoff using only the LAST price of each path (S_T).
    """
    S_T = paths[:, -1]
    return vanilla_payoff(S_T, K, option_type)


def asian_payoff(paths, K, option_type="call", asian_average="arithmetic"):
    """
    Payoff using the AVERAGE price along each path.
    asian_average: "arithmetic" (default) or "geometric"
    Check: Asian call < European call (same inputs).
    """
    if asian_average == "arithmetic":
        average = paths.mean(axis=1)                    # 每条路径求算术平均
    elif asian_average == "geometric":
        average = np.exp(np.log(paths).mean(axis=1))    # 几何平均
    else:
        raise ValueError("asian_average must be 'arithmetic' or 'geometric'")
    return vanilla_payoff(average, K, option_type)


def barrier_payoff(paths, K, barrier, barrier_type, option_type="call"):
    """
    barrier_type: "up-and-in", "up-and-out", "down-and-in", "down-and-out"

    Knock-out: pays the European payoff only if the barrier was NEVER touched.
    Knock-in : pays the European payoff only if the barrier WAS touched.

    Check (in-out parity): up-and-in + up-and-out = European  (same for down)
    """
    if barrier_type.startswith("up"):
        hit = (paths >= barrier).any(axis=1)
    elif barrier_type.startswith("down"):
        hit = (paths <= barrier).any(axis=1)
    else:
        raise ValueError("barrier_type must start with 'up' or 'down'")

    if barrier_type.endswith("in"):
        alive = hit
    elif barrier_type.endswith("out"):
        alive = ~hit
    else:
        raise ValueError("barrier_type must end with 'in' or 'out'")

    return european_payoff(paths, K, option_type) * alive


# =============================================================================
# 6. AMERICAN OPTIONS: LONGSTAFF-SCHWARTZ  (block 5)
# =============================================================================
def american_lsm_payoffs(paths, K, r, dt, option_type="put"):
    """
    Least-Squares Monte Carlo. Returns ONE discounted-to-today cash flow per path,
    so the result can go straight into mc_summary.

    Checks: American put > European put (~6.0 vs ~5.57 with standard inputs)
            American call ~ European call (no dividends -> never exercise early)
    """
    n_sims, n_steps = paths.shape

    # 一开始假设所有路径都在最后一天行权
    cash_flow = vanilla_payoff(paths[:, -1], K, option_type)   # 每条路径的现金流
    exercise_step = np.full(n_sims, n_steps - 1)              # 每条路径在第几步行权

    # 从倒数第二步，倒着走到第 0 步
    for t in range(n_steps - 2, -1, -1):
        S_t = paths[:, t]
        immediate = vanilla_payoff(S_t, K, option_type)        # 现在行权能拿多少
        itm = immediate > 0                                    # 价内路径（True/False）

        if itm.sum() < 3:                                      # 价内路径太少，没法回归
            continue

        # Y：未来现金流折现到今天（只看价内路径）
        steps_ahead = exercise_step[itm] - t
        Y = cash_flow[itm] * np.exp(-r * dt * steps_ahead)

        # X：今天的股价（除以 K，让数字在 1 附近，回归更稳定）
        X = S_t[itm] / K

        # 回归 Y = a + b*X + c*X^2，得到"继续持有"的估计价值
        coeffs = np.polyfit(X, Y, 2)
        continuation = np.polyval(coeffs, X)

        # 现在行权更划算的路径 → 改成今天行权
        exercise_now = immediate[itm] > continuation
        idx = np.where(itm)[0][exercise_now]                   # 这些路径在原数组里的位置
        cash_flow[idx] = immediate[idx]
        exercise_step[idx] = t

    # 把每条路径的现金流从行权那一天折现回今天
    # （paths 的第 t 列对应时间 (t+1)*dt，因为矩阵不包括 S0）
    return cash_flow * np.exp(-r * dt * (exercise_step + 1))


# =============================================================================
# 7. MAIN ENTRY POINT  (block 5) -- signature follows the project document
# =============================================================================
def price_option(S0, K, expiry_days, r, sigma,
                 option_type="call",
                 option_style="european",
                 n_simulations=100_000,
                 barrier=None,
                 barrier_type=None,
                 asian_average="arithmetic",
                 compute_greeks=False,
                 antithetic=False,
                 seed=None,
                 step_days=None):
    """
    Price one option by Monte Carlo.

    option_style: "european", "asian", "barrier", "american"

    Returns
    -------
    dict: price, std_error, ci_low, ci_high, n_steps, dt
          (+ "greeks" sub-dict if compute_greeks=True)
    """
    if option_style == "barrier" and (barrier is None or barrier_type is None):
        raise ValueError("barrier options need both barrier and barrier_type")

    dt, n_steps, T = get_time_grid(expiry_days, step_days)
    paths = simulate_gbm_paths(S0, r, sigma, expiry_days, n_simulations,
                               seed=seed, antithetic=antithetic, step_days=step_days)
    discount = np.exp(-r * T)

    if option_style == "european":
        discounted = european_payoff(paths, K, option_type) * discount
    elif option_style == "asian":
        discounted = asian_payoff(paths, K, option_type, asian_average) * discount
    elif option_style == "barrier":
        discounted = barrier_payoff(paths, K, barrier, barrier_type, option_type) * discount
    elif option_style == "american":
        discounted = american_lsm_payoffs(paths, K, r, dt, option_type)   # 已经折现过了
    else:
        raise ValueError("option_style must be european, asian, barrier or american")

    if antithetic:
        half = len(discounted) // 2
        discounted = 0.5 * (discounted[:half] + discounted[half:])   # 每对 (Z, -Z) 先取平均

    result = mc_summary(discounted)
    result["n_steps"] = n_steps
    result["dt"] = dt

    if compute_greeks:
        result["greeks"] = numerical_greeks(
            S0, K, expiry_days, r, sigma,
            seed=42 if seed is None else seed,
            option_type=option_type, option_style=option_style,
            n_simulations=n_simulations, barrier=barrier, barrier_type=barrier_type,
            asian_average=asian_average, antithetic=antithetic,
        )
    return result



def up_and_out_call_continuous(S0, K, H, T, r, sigma):
    """Closed-form price of a continuously monitored up-and-out call (H > K, no dividends).
    Benchmark for the discrete-monitoring study (Hull, 'Barrier options')."""
    sqrtT = sigma * np.sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S0 / H) / sqrtT + lam * sqrtT
    y = np.log(H**2 / (S0 * K)) / sqrtT + lam * sqrtT
    y1 = np.log(H / S0) / sqrtT + lam * sqrtT
    vanilla = black_scholes(S0, K, T, r, sigma, "call")
    up_and_in = (S0 * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sqrtT)
                 - S0 * (H / S0)**(2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
                 + K * np.exp(-r * T) * (H / S0)**(2 * lam - 2) * (norm.cdf(-y + sqrtT) - norm.cdf(-y1 + sqrtT)))
    return vanilla - up_and_in                      # in-out parity: out = vanilla - in


# =============================================================================
# 8. NUMERICAL GREEKS  (block 6, optional)
# =============================================================================
def numerical_greeks(S0, K, expiry_days, r, sigma, seed=42, **option_kwargs):
    """
    Finite-difference Greeks by bumping one input and re-pricing.
    ALWAYS use the SAME seed for the bumped prices (common random numbers),
    otherwise simulation noise swamps the bump.

    delta = (V(S0+h) - V(S0-h)) / (2h)
    gamma = (V(S0+h) - 2V(S0) + V(S0-h)) / h^2
    vega  = (V(sigma+0.01) - V(sigma-0.01)) / 0.02
    theta, rho similarly (bump expiry_days by 1, r by 0.0001)

    Check: European call delta ~ N(d1) ~ 0.64 with standard inputs
    """
    def V(s=S0, vol=sigma, days=expiry_days, rate=r):
        """Price with ONE input changed. Same seed every time (common random numbers)."""
        return price_option(s, K, days, rate, vol, seed=seed, **option_kwargs)["price"]

    h = 0.01 * S0                     # bump the stock price by 1%
    v0 = V()                          # price with the original inputs
    v_up = V(s=S0 + h)
    v_down = V(s=S0 - h)

    return {
        "delta": (v_up - v_down) / (2 * h),
        "gamma": (v_up - 2 * v0 + v_down) / h**2,
        "vega": (V(vol=sigma + 0.01) - V(vol=sigma - 0.01)) / 2,                  # per 1 volatility point
        "theta": V(days=expiry_days - 1) - v0,                                     # per trading day
        "rho": (V(rate=r + 0.0001) - V(rate=r - 0.0001)) / 0.0002 / 100,          # per 1 interest-rate point
    }
