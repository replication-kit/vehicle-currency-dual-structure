from pathlib import Path
import importlib.util
import sys
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_fig5", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_fig5"] = m
spec.loader.exec_module(m)


def long_run_shares(params, sizes):
    res = m.run_model(params, sizes=sizes, depth_mode="endogenous")
    h = res.history
    W = params.long_run_window
    return float(h["VS"].tail(W).mean()), float(h["CS"].tail(W).mean())


def main():
    p = m.ModelParams()
    sizes = m.draw_relationship_sizes(p)
    W = p.long_run_window

    kappa_grid = np.linspace(0.40, 0.90, 11)
    vs_kappa, cs_kappa = [], []
    for kappa in kappa_grid:
        pp = replace(p, kappa=float(kappa))
        vs, cs = long_run_shares(pp, sizes)
        vs_kappa.append(vs)
        cs_kappa.append(cs)

    F_inf_grid = np.linspace(0.010, 0.050, 9)
    vs_finf, cs_finf = [], []
    for F_inf in F_inf_grid:
        pp = replace(p, F_inf=float(F_inf))
        vs, cs = long_run_shares(pp, sizes)
        vs_finf.append(vs)
        cs_finf.append(cs)

    pd.concat([
        pd.DataFrame({
            "parameter": "kappa",
            "value": kappa_grid,
            "long_run_VS": vs_kappa,
            "long_run_CS": cs_kappa,
        }),
        pd.DataFrame({
            "parameter": "F_inf",
            "value": F_inf_grid,
            "long_run_VS": vs_finf,
            "long_run_CS": cs_finf,
        }),
    ], ignore_index=True).to_csv(HERE / "Fig5_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))

    ax = axes[0]
    ax.plot(kappa_grid, vs_kappa, linestyle="-", linewidth=1.8, color="black",
            marker="o", markersize=4, label="Value share")
    ax.plot(kappa_grid, cs_kappa, linestyle="--", linewidth=1.6, color="black",
            marker="s", markersize=4, markerfacecolor="white", label="Count share")
    ax.axvline(p.kappa, linestyle=":", linewidth=1.3, color="0.4", label="Baseline")
    ax.set_xlabel(r"Replication-floor parameter $\kappa$")
    ax.set_ylabel(f"Long-run direct-settlement share\n(avg. last {W} periods)")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.02, 0.94, "(a)", transform=ax.transAxes)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(False)

    ax = axes[1]
    ax.plot(F_inf_grid, vs_finf, linestyle="-", linewidth=1.8, color="black",
            marker="o", markersize=4, label="Value share")
    ax.plot(F_inf_grid, cs_finf, linestyle="--", linewidth=1.6, color="black",
            marker="s", markersize=4, markerfacecolor="white", label="Count share")
    ax.axvline(p.F_inf, linestyle=":", linewidth=1.3, color="0.4", label="Baseline")
    ax.set_xlabel(r"Residual operational burden $F(\infty)$")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.02, 0.94, "(b)", transform=ax.transAxes)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "Fig5.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "Fig5.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "Fig5.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
