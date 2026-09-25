from pathlib import Path
import importlib.util
import sys
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_figc2", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_figc2"] = m
spec.loader.exec_module(m)


def pareto_sizes_from_u(U, a, scale, shift):
    return ((1.0 - U) ** (-1.0 / a) - 1.0) * scale + shift


def main():
    p = m.ModelParams()
    W = p.long_run_window

    rng = np.random.RandomState(p.seed)
    U = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=p.N)

    a_grid = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00])
    vs_list, cs_list = [], []

    for a in a_grid:
        sizes = pareto_sizes_from_u(U, a, p.pareto_scale, p.pareto_shift)
        pp = replace(p, pareto_a=float(a))
        res = m.run_model(pp, sizes=sizes, depth_mode="endogenous")
        h = res.history
        vs_list.append(float(h["VS"].tail(W).mean()))
        cs_list.append(float(h["CS"].tail(W).mean()))

    vs_arr = np.array(vs_list)
    cs_arr = np.array(cs_list)

    pd.DataFrame({
        "pareto_shape_a": a_grid,
        "long_run_VS": vs_arr,
        "long_run_CS": cs_arr,
        "value_count_wedge": vs_arr - cs_arr,
    }).to_csv(HERE / "FigC2_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(a_grid, vs_arr, linestyle="-", linewidth=1.8, color="black",
            marker="o", markersize=4, label="Value share")
    ax.plot(a_grid, cs_arr, linestyle="--", linewidth=1.6, color="black",
            marker="s", markersize=4, markerfacecolor="white", label="Count share")
    ax.axvline(p.pareto_a, linestyle=":", linewidth=1.3, color="0.4", label="Baseline")

    ax.set_xlabel(r"Pareto shape parameter $a$ (smaller $a$ = thicker tail)")
    ax.set_ylabel(f"Long-run direct-settlement share\n(avg. last {W} periods)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "FigC2.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "FigC2.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "FigC2.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
