# %%
#Figure 2 Generation Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility: fix RNG seed
np.random.seed(123)


class FinancialMarket:
    def __init__(self, params: dict):
        self.params = params

        # --- Parameter definitions ---
        # alpha: liquidity sensitivity
        # delta: curvature (power) of the liquidity term
        # s_base: baseline inventory cost (as a fraction of notional)
        # eta: AI-driven reduction rate for inventory cost
        # kappa: replication-floor coefficient (inter-dealer / client rate)
        # usd_spread: USD vehicle route cost to clients (as a fraction of notional)
        self.alpha = params["alpha"]
        self.delta = params["delta"]
        self.s_base = params["s_base"]
        self.eta = params["eta"]
        self.kappa = params["kappa"]
        self.usd_spread = params["usd_spread"]

        # Initial liquidity seed (exogenous volume to help escape lock-in)
        initial_seed = params["initial_volume_seed"]
        self.volume_history = [initial_seed] * 10

        self.ai_level = 0.0
        self.current_spreads = {}

        # Fixed administrative cost parameter F(0)
        # Note: this is a normalized amount level (not a rate)
        self.base_threshold_fixed = params["base_threshold_fixed"]

    def update_ai(self, step: int) -> None:
        """AI diffusion level (logistic curve)."""
        k = 0.15
        t0 = 50
        self.ai_level = 1 / (1 + np.exp(-k * (step - t0)))

    def pre_step_update(self, step: int) -> None:
        """Step 1: update market environment and compute current spreads."""
        self.update_ai(step)

        # 1) Volume: moving average over the last 10 periods
        recent_vols = self.volume_history[-10:]
        avg_volume = np.mean(recent_vols)

        # 2) Direct spread calculation
        # A) Liquidity cost
        liquidity_term = self.alpha / (avg_volume**self.delta)

        # B) Inventory cost (reduced by AI)
        inventory_term = self.s_base * (1 - self.eta * self.ai_level)

        theoretical_spread = liquidity_term + inventory_term

        # C) Replication floor (lower bound based on inter-dealer hedging cost)
        floor_spread = self.kappa * self.usd_spread

        # Final direct spread
        final_spread = max(theoretical_spread, floor_spread)

        self.current_spreads = {
            "USD": self.usd_spread,
            "Direct": final_spread,
        }

    def get_current_fixed_cost(self) -> float:
        """Current fixed administrative cost F(t).

        Assumption: AI diffusion reduces F(0) by up to 50% at full adoption.
        """
        return self.base_threshold_fixed * (1 - self.ai_level * 0.5)

    def post_step_record(self, transactions: list[dict]) -> None:
        """Step 3: record transaction outcomes and update volume history."""
        direct_vol = sum(tx["size"] for tx in transactions if tx["currency"] == "Direct")

        # Add exogenous liquidity seed when updating the history
        effective_vol = direct_vol + self.params["initial_volume_seed"]
        self.volume_history.append(effective_vol)


class TradingFirm:
    def __init__(self, uid: int, size: float):
        self.uid = uid
        self.size = size  # transaction size A_i
        self.current_currency = "USD"

    def decide_and_trade(self, market: FinancialMarket) -> dict:
        """Step 2: decide settlement route with hysteresis and execute trade."""
        spreads = market.current_spreads

        # --- Size-dependent switching cost ---
        # phi_i(t) = F(t) / A_i
        fixed_cost = market.get_current_fixed_cost()

        # Guard against extremely small sizes to avoid division blow-ups
        safe_size = max(self.size, 1e-6)
        my_threshold = fixed_cost / safe_size

        cost_usd = spreads["USD"]
        cost_direct = spreads["Direct"]

        # Decision rule (with hysteresis)
        if self.current_currency == "USD":
            # Switch if direct cost is sufficiently below USD cost net of admin burden
            if cost_direct < (cost_usd - my_threshold):
                self.current_currency = "Direct"
        else:
            # Switch back if USD cost is sufficiently below direct cost net of admin burden
            if cost_usd < (cost_direct - my_threshold):
                self.current_currency = "USD"

        return {
            "id": self.uid,
            "currency": self.current_currency,
            "size": self.size,
        }


def run_simulation() -> tuple[pd.DataFrame, dict]:
    # Baseline parameter set (chosen to satisfy the intended inequality conditions)
    params = {
        "alpha": 0.02,  # liquidity term coefficient
        "delta": 0.5,  # power-law curvature
        "s_base": 0.04,  # baseline inventory cost (4%)
        "eta": 0.95,  # AI efficiency (95% reduction)
        "kappa": 0.6,  # replication-floor coefficient (inter-dealer rate ratio)
        "usd_spread": 0.01,  # USD vehicle route cost (1% all-in)

        # Fixed administrative cost F(0) = 0.05 (normalized currency units)
        "base_threshold_fixed": 0.05,

        # Initial liquidity seed (to help escape lock-in)
        "initial_volume_seed": 50.0,
    }

    steps = 150
    n_agents = 200

    market = FinancialMarket(params)

    # Firm-size distribution (Pareto)
    # A_i ~ Pareto(2.0) scaled
    agent_sizes = np.random.pareto(a=2.0, size=n_agents) * 10 + 1.0
    agents = [TradingFirm(i, size) for i, size in enumerate(agent_sizes)]

    results: list[dict] = []

    for t in range(steps):
        market.pre_step_update(t)

        daily_transactions = [agent.decide_and_trade(market) for agent in agents]
        market.post_step_record(daily_transactions)

        # Aggregation
        count_share = sum(tx["currency"] == "Direct" for tx in daily_transactions) / n_agents

        total_vol = sum(tx["size"] for tx in daily_transactions)
        direct_vol = sum(tx["size"] for tx in daily_transactions if tx["currency"] == "Direct")
        vol_share = direct_vol / total_vol if total_vol > 0 else 0.0

        spreads = market.current_spreads

        results.append(
            {
                "step": t,
                "ai_level": market.ai_level,
                "usd_spread": spreads["USD"],
                "direct_spread": spreads["Direct"],
                "count_share": count_share,
                "volume_share": vol_share,
            }
        )

    return pd.DataFrame(results), params


# ==========================================
# Visualization (Figure 2)
# ==========================================
if __name__ == "__main__":
    df, params = run_simulation()

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Cost plot
    ax1.set_xlabel("Time Step (AI Progression)")
    ax1.set_ylabel("Transaction Cost (All-in Spread)", color="tab:red", fontsize=12)
    ax1.plot(df["step"], df["usd_spread"], color="gray", linestyle="--", label="USD Cost (Client Rate)")
    ax1.plot(df["step"], df["direct_spread"], color="tab:red", linewidth=2.5, label="Direct Cost (JPY/THB)")

    # Replication floor
    floor_val = params["usd_spread"] * params["kappa"]
    ax1.axhline(y=floor_val, color="orange", linestyle=":", label="Replication Floor (Inter-dealer)")

    max_cost = max(df["direct_spread"].max(), df["usd_spread"].max())
    ax1.set_ylim(0, max_cost * 1.1)
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Share plot
    ax2 = ax1.twinx()
    ax2.set_ylabel("Direct Share / AI Level", color="tab:blue", fontsize=12)

    ax2.plot(df["step"], df["ai_level"], color="green", linestyle=":", alpha=0.5, label="AI Level")

    # Dual-structure visualization
    ax2.plot(df["step"], df["volume_share"], color="tab:blue", linewidth=2.5, label="Volume Share (Amount)")
    ax2.plot(df["step"], df["count_share"], color="tab:blue", linestyle="--", linewidth=1.5, label="Count Share (Firms)")

    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.05)

    # Final-values annotation
    final_vol = df["volume_share"].iloc[-1]
    final_count = df["count_share"].iloc[-1]
    ax2.text(145, final_vol, f"Vol: {final_vol:.1%}", color="tab:blue", fontweight="bold", ha="right", va="bottom")
    ax2.text(145, final_count, f"Count: {final_count:.1%}", color="tab:blue", ha="right", va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", bbox_to_anchor=(1.0, 0.62))

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save before showing (safer across backends)
    fig.savefig("Figure2.png", dpi=300, bbox_inches="tight")
    plt.show()


# %%
# =========================
# Figure 3: Adoption-by-size with analytical threshold A*
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _is_direct(tx):
    """
    Robustly infer whether a transaction used direct settlement.
    Adjust this if your tx dict uses different keys.
    """
    if tx is None:
        return False
    # Common patterns
    if isinstance(tx, dict):
        if 'currency' in tx:
            return str(tx['currency']).lower() in ['direct', 'dir', 'direct_pair']
        if 'settlement' in tx:
            return str(tx['settlement']).lower() in ['direct', 'dir', 'direct_pair']
        if 'is_direct' in tx:
            return bool(tx['is_direct'])
    # Fallback: try attribute access
    if hasattr(tx, 'currency'):
        return str(getattr(tx, 'currency')).lower() in ['direct', 'dir', 'direct_pair']
    if hasattr(tx, 'is_direct'):
        return bool(getattr(tx, 'is_direct'))
    return False


def run_simulation_panel(
    seed=123,
    steps=150,
    n_agents=200,
    params=None,
    pareto_shape=2.0,
    pareto_scale=10.0,
    pareto_shift=1.0,
):
    """
    Run the same baseline simulation as Figure 2, but record agent-level adoption.
    Requires FinancialMarket, TradingFirm to be defined (from your Figure 2 notebook).

    Returns:
      A: (n_agents,) array of relationship sizes A_i
      adopt: (steps, n_agents) array of 1 if Direct else 0
      params: dict used
    """
    np.random.seed(seed)

    # ---- Baseline params (match your Figure 2 notebook) ----
    if params is None:
        params = {
            'alpha': 0.02,
            'delta': 0.5,
            's_base': 0.04,
            'eta': 0.95,
            'kappa': 0.6,
            'usd_spread': 0.01,
            'base_threshold_fixed': 0.05,
            'initial_volume_seed': 50.0,
            # add any other keys your FinancialMarket expects here
        }

    market = FinancialMarket(params)

    # ---- Draw heterogeneous relationship sizes ----
    A = np.random.pareto(a=pareto_shape, size=n_agents) * pareto_scale + pareto_shift
    agents = [TradingFirm(i, size) for i, size in enumerate(A)]

    adopt = np.zeros((steps, n_agents), dtype=int)

    for t in range(steps):
        # If your class uses a different method name, change here.
        market.pre_step_update(t)

        transactions = []
        for j, agent in enumerate(agents):
            tx = agent.decide_and_trade(market)
            transactions.append(tx)
            adopt[t, j] = 1 if _is_direct(tx) else 0

        market.post_step_record(transactions)

    return A, adopt, params


def compute_A_star(params, fixed_cost_ai_slope=0.5):
    """
    Computes A* under the steady-state condition:
      S_dir = kappa * S_USD  and  F -> F(infty).
    In your baseline Figure 2 code: F(t)=F0*(1 - fixed_cost_ai_slope*AI(t)).
    So F(infty)=F0*(1 - fixed_cost_ai_slope).
    """
    S_usd = params['usd_spread']
    kappa = params['kappa']
    F0 = params['base_threshold_fixed']
    F_inf = F0 * (1 - fixed_cost_ai_slope)

    denom = (1 - kappa) * S_usd
    if denom <= 0:
        return np.inf
    return F_inf / denom


# -------------------------
# Run (single seed baseline)
# -------------------------
A, adopt, params = run_simulation_panel(seed=123)

# Use last W periods to estimate steady-state adoption probability
W = 30
adopt_rate = adopt[-W:, :].mean(axis=0)

A_star = compute_A_star(params, fixed_cost_ai_slope=0.5)

# Sort by size and smooth
order = np.argsort(A)
x = A[order]
y = adopt_rate[order]

roll_n = max(5, int(0.05 * len(x)))  # 5% rolling window
y_smooth = pd.Series(y).rolling(roll_n, center=True, min_periods=1).mean().to_numpy()

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.scatter(x, y, s=14, alpha=0.25, label=f'Adoption rate by relationship (last {W} periods)')
ax.plot(x, y_smooth, linewidth=2.0, label='Smoothed adoption profile')

# Safe mathtext: use \ast (NOT \*)
ax.axvline(
    A_star,
    linestyle='--',
    linewidth=2.0,
    label=rf'Analytical threshold $A^{{\ast}}={A_star:.2f}$'
)

ax.set_xscale('log')
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(r"Relationship size $A_i$ (log scale)")
ax.set_ylabel(f"Direct-settlement adoption rate (last {W} periods)")

ax.legend(loc='best')
plt.show()

# Optional save:
fig.savefig("Figure3.png", dpi=300, bbox_inches="tight")


# %%
# =========================
# Figure 4: Channel decomposition (Supply-only vs Demand-only vs Both)
#   - Supply-side AI: AI lowers direct-pair variable cost via eta (inventory_term)
#   - Demand-side AI: AI lowers fixed operational cost via get_current_fixed_cost()
# =========================

import numpy as np
import matplotlib.pyplot as plt
import copy
import types

def run_scenario(A, base_params, steps=150, scenario="both"):
    """
    Runs a simulation under a channel toggle while keeping:
      - identical relationship sizes A (same draw across scenarios)
      - identical AI path (FinancialMarket.update_ai is unchanged)
    Returns:
      value_share_direct: (steps,) array
      count_share_direct: (steps,) array
      ai_path:            (steps,) array
    """

    params = copy.deepcopy(base_params)

    # ---- Toggle channels ----
    # Demand-only: shut down supply-side AI effect in spreads by setting eta = 0
    if scenario == "demand_only":
        params["eta"] = 0.0

    market = FinancialMarket(params)

    # Supply-only: shut down demand-side AI effect in fixed cost by making fixed cost constant
    if scenario == "supply_only":
        def _fixed_cost_constant(self):
            return self.base_threshold_fixed  # F(t) = F(0)
        market.get_current_fixed_cost = types.MethodType(_fixed_cost_constant, market)

    # Both: baseline (no changes)

    # Initialize agents with identical sizes and identical initial currency (USD)
    agents = [TradingFirm(i, float(A[i])) for i in range(len(A))]

    value_share_direct = np.zeros(steps)
    count_share_direct = np.zeros(steps)
    ai_path = np.zeros(steps)

    for t in range(steps):
        market.pre_step_update(t)
        ai_path[t] = market.ai_level

        transactions = []
        for agent in agents:
            tx = agent.decide_and_trade(market)
            transactions.append(tx)

        # Record market depth for next period (same as Figure 2 mechanism)
        market.post_step_record(transactions)

        # Compute shares (direct vs USD) from realized choices
        direct_value = sum(tx["size"] for tx in transactions if tx["currency"] == "Direct")
        total_value  = sum(tx["size"] for tx in transactions)

        direct_count = sum(1 for tx in transactions if tx["currency"] == "Direct")
        total_count  = len(transactions)

        value_share_direct[t] = direct_value / total_value if total_value > 0 else 0.0
        count_share_direct[t] = direct_count / total_count if total_count > 0 else 0.0

    return value_share_direct, count_share_direct, ai_path


# -------------------------
# Baseline params: keep identical to Figure 2
# -------------------------
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,
    "initial_volume_seed": 50.0,
}

steps = 150
n_agents = 200

# Fix the A_i draw once and reuse across scenarios (critical for clean decomposition)
np.random.seed(123)
A = np.random.pareto(a=2.0, size=n_agents) * 10.0 + 1.0

# -------------------------
# Run three scenarios
# -------------------------
vs_supply, cs_supply, ai_path = run_scenario(A, base_params, steps=steps, scenario="supply_only")
vs_demand, cs_demand, _       = run_scenario(A, base_params, steps=steps, scenario="demand_only")
vs_both,   cs_both,   _       = run_scenario(A, base_params, steps=steps, scenario="both")

t = np.arange(steps)

# -------------------------
# Plot Figure 4 (focus: Supply-only vs Both = marginal effect of demand-side AI)
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

# Panel A: Value share
ax1.plot(t, vs_supply, label="Supply-side AI only")
ax1.plot(t, vs_both,   label="Both channels (baseline)")
ax1.fill_between(t, vs_supply, vs_both, alpha=0.15, label="Marginal effect of demand-side AI")
ax1.set_ylabel("Direct settlement value share")
ax1.set_ylim(-0.02, 1.02)
ax1.legend(loc="best")

# Panel B: Count share
ax2.plot(t, cs_supply, label="Supply-side AI only")
ax2.plot(t, cs_both,   label="Both channels (baseline)")
ax2.fill_between(t, cs_supply, cs_both, alpha=0.15, label="Marginal effect of demand-side AI")
ax2.set_xlabel("Time step (AI progression)")
ax2.set_ylabel("Direct settlement count share")
ax2.set_ylim(-0.02, 1.02)
ax2.legend(loc="best")

plt.show()
fig.savefig("Figure4.png", dpi=300, bbox_inches="tight")


# %%
# =========================
# Figure 5: Robustness summary over kappa (replication/hedging floor)
#   - x-axis: kappa
#   - y-axis: long-run (last W periods) direct-settlement value share & count share
# =========================

import numpy as np
import matplotlib.pyplot as plt
import copy

def run_one_simulation(A, base_params, steps=150, W=30, seed_agents_state=0):
    """
    Run one simulation with fixed relationship sizes A and given params.
    Returns long-run averages over last W periods:
      (value_share_direct, count_share_direct)
    """
    # If your agents/market have any internal randomness beyond A,
    # you can set a seed here for reproducibility.
    np.random.seed(seed_agents_state)

    params = copy.deepcopy(base_params)
    market = FinancialMarket(params)
    agents = [TradingFirm(i, float(A[i])) for i in range(len(A))]

    value_share = np.zeros(steps)
    count_share = np.zeros(steps)

    for t in range(steps):
        market.pre_step_update(t)

        transactions = []
        for agent in agents:
            tx = agent.decide_and_trade(market)
            transactions.append(tx)

        market.post_step_record(transactions)

        direct_value = sum(tx["size"] for tx in transactions if tx["currency"] == "Direct")
        total_value  = sum(tx["size"] for tx in transactions)

        direct_count = sum(1 for tx in transactions if tx["currency"] == "Direct")
        total_count  = len(transactions)

        value_share[t] = direct_value / total_value if total_value > 0 else 0.0
        count_share[t] = direct_count / total_count if total_count > 0 else 0.0

    vs = value_share[-W:].mean()
    cs = count_share[-W:].mean()
    return vs, cs


# -------------------------
# Baseline params (match Figure 2 / Table 1 baseline)
# -------------------------
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,                 # will be overwritten in the sweep
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,
    "initial_volume_seed": 50.0,
}

steps = 150
W = 30
n_agents = 200

# Fix the relationship-size draw ONCE and reuse across all kappa values
np.random.seed(123)
A = np.random.pareto(a=2.0, size=n_agents) * 10.0 + 1.0

# Sweep kappa over the disciplined range (edit points as you like)
kappa_grid = np.linspace(0.40, 0.90, 11)  # 11 points: 0.40, 0.45, ..., 0.90

vs_list, cs_list = [], []
for k in kappa_grid:
    p = copy.deepcopy(base_params)
    p["kappa"] = float(k)
    vs, cs = run_one_simulation(A, p, steps=steps, W=W, seed_agents_state=0)
    vs_list.append(vs)
    cs_list.append(cs)

vs_arr = np.array(vs_list)
cs_arr = np.array(cs_list)

# -------------------------
# Plot Figure 5
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(kappa_grid, vs_arr, marker="o", linewidth=2, label="Direct settlement value share (long-run)")
ax.plot(kappa_grid, cs_arr, marker="o", linewidth=2, label="Direct settlement count share (long-run)")

# Shade the wedge (value - count), which corresponds to the dual-structure gap
ax.fill_between(kappa_grid, cs_arr, vs_arr, alpha=0.15, label="Value–count wedge")

# Baseline vertical reference line
ax.axvline(base_params["kappa"], linestyle="--", linewidth=1.5, label="Baseline kappa")

ax.set_xlabel(r"Replication/hedging floor ratio $\kappa$")
ax.set_ylabel(f"Long-run direct-settlement share (avg. last {W} periods)")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="best")

plt.show()

# Optional save:
fig.savefig("Figure5.png", dpi=300, bbox_inches="tight")


# %%
# =========================
# Figure 6: Robustness over residual fixed cost F(infty)
#   - x-axis: F(infty) (residual operational fixed cost in AI limit)
#   - y-axis: long-run (last W periods) direct-settlement value share & count share
# =========================

import numpy as np
import matplotlib.pyplot as plt
import copy

def run_one_simulation(A, params, steps=150, W=30, seed_agents_state=0):
    """
    Run one simulation with fixed relationship sizes A and given params.
    Returns long-run averages over last W periods:
      (value_share_direct, count_share_direct)
    """
    np.random.seed(seed_agents_state)

    market = FinancialMarket(params)
    agents = [TradingFirm(i, float(A[i])) for i in range(len(A))]

    value_share = np.zeros(steps)
    count_share = np.zeros(steps)

    for t in range(steps):
        market.pre_step_update(t)

        transactions = []
        for agent in agents:
            tx = agent.decide_and_trade(market)
            transactions.append(tx)

        market.post_step_record(transactions)

        direct_value = sum(tx["size"] for tx in transactions if tx["currency"] == "Direct")
        total_value  = sum(tx["size"] for tx in transactions)

        direct_count = sum(1 for tx in transactions if tx["currency"] == "Direct")
        total_count  = len(transactions)

        value_share[t] = direct_value / total_value if total_value > 0 else 0.0
        count_share[t] = direct_count / total_count if total_count > 0 else 0.0

    return value_share[-W:].mean(), count_share[-W:].mean()


# -------------------------
# Baseline params (match Table 1 / Figure 2 baseline)
# -------------------------
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,   # F0 (will be overwritten via F_infty targeting)
    "initial_volume_seed": 50.0,
}

# Simulation settings
steps = 150
W = 30
n_agents = 200

# Fix A_i draw once and reuse across sweep
np.random.seed(123)
A = np.random.pareto(a=2.0, size=n_agents) * 10.0 + 1.0

# -------------------------
# Fixed-cost decline law assumption used in your code
#   F(t) = F0 * (1 - slope * AI(t))
#   => F(infty) = F0 * (1 - slope)
# -------------------------
slope = 0.5  # MUST match your FinancialMarket.get_current_fixed_cost() implementation

def F0_from_Finf(F_inf, slope=0.5):
    denom = (1 - slope)
    if denom <= 0:
        raise ValueError("Invalid slope: must be < 1 to have positive F(infty).")
    return F_inf / denom


# Sweep residual fixed cost F(infty) over a disciplined range
F_inf_grid = np.linspace(0.01, 0.05, 9)  # e.g., 0.01, 0.015, ..., 0.05

vs_list, cs_list = [], []
for F_inf in F_inf_grid:
    p = copy.deepcopy(base_params)
    p["base_threshold_fixed"] = float(F0_from_Finf(F_inf, slope=slope))  # back out F0
    vs, cs = run_one_simulation(A, p, steps=steps, W=W, seed_agents_state=0)
    vs_list.append(vs)
    cs_list.append(cs)

vs_arr = np.array(vs_list)
cs_arr = np.array(cs_list)

# Baseline marker: compute baseline F(infty) implied by base_params
F0_baseline = base_params["base_threshold_fixed"]
F_inf_baseline = F0_baseline * (1 - slope)

# -------------------------
# Plot Figure 6
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(F_inf_grid, vs_arr, marker="o", linewidth=2, label="Direct settlement value share (long-run)")
ax.plot(F_inf_grid, cs_arr, marker="o", linewidth=2, label="Direct settlement count share (long-run)")
ax.fill_between(F_inf_grid, cs_arr, vs_arr, alpha=0.15, label="Value–count wedge")

ax.axvline(F_inf_baseline, linestyle="--", linewidth=1.5, label=r"Baseline $F(\infty)$")

ax.set_xlabel(r"Residual fixed operational cost $F(\infty)$ (per period)")
ax.set_ylabel(f"Long-run direct-settlement share (avg. last {W} periods)")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="best")

plt.show()

# Optional save:
fig.savefig("Figure6.png", dpi=300, bbox_inches="tight")


# %%
import copy
import numpy as np

def compute_long_run_shares_for_case(A, base_params, overrides, steps=150, W=30, seed_agents_state=0):
    p = copy.deepcopy(base_params)
    p.update(overrides)
    vs, cs = run_one_simulation(A, p, steps=steps, W=W, seed_agents_state=seed_agents_state)
    return vs, cs

def pass_fail(vs, cs, target, tol=1e-3):
    if target == 0:
        return (vs < tol) and (cs < tol)
    if target == 1:
        return (vs > 1 - tol) and (cs > 1 - tol)
    return None  # for non-binary targets

# --- baseline (Table 1 / Fig 2) ---
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,
    "initial_volume_seed": 50.0,
}

steps, W = 150, 30
n_agents = 200

# Fix A_i draw once
np.random.seed(123)
A = np.random.pareto(a=2.0, size=n_agents) * 10.0 + 1.0

cases = [
    ("kappa=1 (floor equals USD)", {"kappa": 1.0}, 0),
    ("F=0 (no fixed cost)", {"base_threshold_fixed": 0.0}, 1),
    ("eta=0 (no supply-side AI in spreads)", {"eta": 0.0}, 0),
    ("V_seed=0 (liquidity trap)", {"initial_volume_seed": 0.0}, 0),
]

rows = []
for name, overrides, target in cases:
    vs, cs = compute_long_run_shares_for_case(A, base_params, overrides, steps=steps, W=W, seed_agents_state=0)
    rows.append((name, vs, cs, pass_fail(vs, cs, target)))

rows


# %%
# =========================
# Figure C1: Liquidity feedback robustness (V_seed sweep)
#   x-axis: V_seed (exogenous liquidity seed)
#   y-axis: long-run direct-settlement value share & count share (avg last W periods)
# =========================

import numpy as np
import matplotlib.pyplot as plt
import copy

def _is_direct(tx):
    # Robust check (adjust if your transaction dict uses different keys)
    if isinstance(tx, dict):
        if "currency" in tx:
            return str(tx["currency"]).lower() in ["direct", "dir", "direct_pair"]
        if "settlement" in tx:
            return str(tx["settlement"]).lower() in ["direct", "dir", "direct_pair"]
        if "is_direct" in tx:
            return bool(tx["is_direct"])
    return False

def run_one_simulation(A, params, steps=150, W=30, seed_agents_state=0):
    """
    Run one simulation with fixed relationship sizes A and given params.
    Returns long-run averages over last W periods:
      (value_share_direct, count_share_direct)
    """
    np.random.seed(seed_agents_state)

    market = FinancialMarket(params)
    agents = [TradingFirm(i, float(A[i])) for i in range(len(A))]

    value_share = np.zeros(steps)
    count_share = np.zeros(steps)

    for t in range(steps):
        market.pre_step_update(t)

        transactions = []
        for agent in agents:
            tx = agent.decide_and_trade(market)
            transactions.append(tx)

        market.post_step_record(transactions)

        direct_value = sum(tx["size"] for tx in transactions if _is_direct(tx))
        total_value  = sum(tx["size"] for tx in transactions)

        direct_count = sum(1 for tx in transactions if _is_direct(tx))
        total_count  = len(transactions)

        value_share[t] = direct_value / total_value if total_value > 0 else 0.0
        count_share[t] = direct_count / total_count if total_count > 0 else 0.0

    return value_share[-W:].mean(), count_share[-W:].mean()


# -------------------------
# Baseline params (match Table 1 / Figure 2 baseline)
# -------------------------
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,
    "initial_volume_seed": 50.0,  # V_seed baseline
}

steps = 150
W = 30
n_agents = 200

# Fix A_i draw once and reuse across the sweep
np.random.seed(123)
A = np.random.pareto(a=2.0, size=n_agents) * 10.0 + 1.0

# -------------------------
# Sweep V_seed over a disciplined range
# (avoid exactly 0 unless you want the boundary "liquidity trap" case)
# -------------------------
V_seed_grid = np.array([10, 25, 50, 75, 100, 150, 200], dtype=float)

vs_list, cs_list = [], []
for vseed in V_seed_grid:
    p = copy.deepcopy(base_params)
    p["initial_volume_seed"] = float(vseed)
    vs, cs = run_one_simulation(A, p, steps=steps, W=W, seed_agents_state=0)
    vs_list.append(vs)
    cs_list.append(cs)

vs_arr = np.array(vs_list)
cs_arr = np.array(cs_list)

# -------------------------
# Plot Figure C1
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(V_seed_grid, vs_arr, marker="o", linewidth=2,
        label="Direct settlement value share (long-run)")
ax.plot(V_seed_grid, cs_arr, marker="o", linewidth=2,
        label="Direct settlement count share (long-run)")
ax.fill_between(V_seed_grid, cs_arr, vs_arr, alpha=0.15,
                label="Value–count wedge")

# Baseline vertical reference
ax.axvline(base_params["initial_volume_seed"], linestyle="--", linewidth=1.5,
           label="Baseline $V_{seed}$")

ax.set_xlabel(r"Exogenous liquidity seed $V_{\mathrm{seed}}$")
ax.set_ylabel(f"Long-run direct-settlement share (avg. last {W} periods)")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="best")

plt.show()

# Optional save:
fig.savefig("FigureC1.png", dpi=300, bbox_inches="tight")


# %%
# =========================
# Figure C2: Heterogeneity robustness (Pareto tail sweep) -- Table 1 disciplined range
#   a in [1.5, 3.0], baseline a=2.0
#   A_i = Pareto(a)*s + shift with baseline s=10, shift=1
# =========================

import numpy as np
import matplotlib.pyplot as plt
import copy

def _is_direct(tx):
    if isinstance(tx, dict):
        if "currency" in tx:
            return str(tx["currency"]).lower() in ["direct", "dir", "direct_pair"]
        if "settlement" in tx:
            return str(tx["settlement"]).lower() in ["direct", "dir", "direct_pair"]
        if "is_direct" in tx:
            return bool(tx["is_direct"])
    return False

def run_one_simulation(A, params, steps=150, W=30, seed_agents_state=0):
    np.random.seed(seed_agents_state)

    market = FinancialMarket(params)
    agents = [TradingFirm(i, float(A[i])) for i in range(len(A))]

    value_share = np.zeros(steps)
    count_share = np.zeros(steps)

    for t in range(steps):
        market.pre_step_update(t)

        transactions = []
        for agent in agents:
            tx = agent.decide_and_trade(market)
            transactions.append(tx)

        market.post_step_record(transactions)

        direct_value = sum(tx["size"] for tx in transactions if _is_direct(tx))
        total_value  = sum(tx["size"] for tx in transactions)

        direct_count = sum(1 for tx in transactions if _is_direct(tx))
        total_count  = len(transactions)

        value_share[t] = direct_value / total_value if total_value > 0 else 0.0
        count_share[t] = direct_count / total_count if total_count > 0 else 0.0

    return value_share[-W:].mean(), count_share[-W:].mean()


# --- Baseline params (Table 1 / Fig. 2) ---
base_params = {
    "alpha": 0.02,
    "delta": 0.5,
    "s_base": 0.04,
    "eta": 0.95,
    "kappa": 0.6,
    "usd_spread": 0.01,
    "base_threshold_fixed": 0.05,
    "initial_volume_seed": 50.0,
}

steps = 150
W = 30
n_agents = 200

# Common random numbers for clean comparison across a
np.random.seed(123)
U = np.random.uniform(low=1e-12, high=1.0 - 1e-12, size=n_agents)

# Table 1 baseline normalization
scale = 10.0
shift = 1.0

def pareto_sample_from_U(U, a):
    # numpy.pareto(a) equivalent: X = (1-U)^(-1/a) - 1
    return (1.0 - U) ** (-1.0 / a) - 1.0

# Disciplined range per Table 1
a_grid = np.array([1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], dtype=float)
a_baseline = 2.0

vs_list, cs_list = [], []
for a in a_grid:
    A = pareto_sample_from_U(U, a) * scale + shift
    vs, cs = run_one_simulation(A, copy.deepcopy(base_params), steps=steps, W=W, seed_agents_state=0)
    vs_list.append(vs)
    cs_list.append(cs)

vs_arr = np.array(vs_list)
cs_arr = np.array(cs_list)

# Plot
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(a_grid, vs_arr, marker="o", linewidth=2, label="Direct settlement value share (long-run)")
ax.plot(a_grid, cs_arr, marker="o", linewidth=2, label="Direct settlement count share (long-run)")
ax.fill_between(a_grid, cs_arr, vs_arr, alpha=0.15, label="Value–count wedge")

ax.axvline(a_baseline, linestyle="--", linewidth=1.5, label="Baseline Pareto shape $a$")
ax.set_xlabel(r"Pareto shape parameter $a$ (smaller $a$ = thicker tail)")
ax.set_ylabel(f"Long-run direct-settlement share (avg. last {W} periods)")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="best")

plt.show()

# Optional save:
fig.savefig("FigureC2.png", dpi=300, bbox_inches="tight")



