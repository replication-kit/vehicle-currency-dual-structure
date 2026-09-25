from pathlib import Path
import importlib.util
import sys
from dataclasses import replace
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_fig4", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_fig4"] = m
spec.loader.exec_module(m)


def main():
    p = m.ModelParams()
    sizes = m.draw_relationship_sizes(p)

    baseline = m.run_model(p, sizes=sizes, depth_mode="endogenous")
    p_variable_only = replace(p, F_inf=p.F0)
    variable_only = m.run_model(p_variable_only, sizes=sizes, depth_mode="endogenous")

    h_b = baseline.history.copy()
    h_v = variable_only.history.copy()

    pd.DataFrame({
        "t": h_b["t"],
        "VS_both_channels": h_b["VS"],
        "CS_both_channels": h_b["CS"],
        "VS_variable_cost_only": h_v["VS"],
        "CS_variable_cost_only": h_v["CS"],
        "F_both_channels": h_b["F"],
        "F_variable_cost_only": h_v["F"],
        "Sdir_both_channels": h_b["S_dir"],
        "Sdir_variable_cost_only": h_v["S_dir"],
    }).to_csv(HERE / "Fig4_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    ax = axes[0]
    ax.plot(h_v["t"], h_v["VS"], linestyle="--", linewidth=1.6, color="black",
            marker="o", markevery=12, markersize=3.0, markerfacecolor="white",
            label="Variable-cost channel only")
    ax.plot(h_b["t"], h_b["VS"], linestyle="-", linewidth=1.8, color="black",
            marker="o", markevery=12, markersize=3.2,
            label="Variable and operational-cost channels")
    ax.set_ylabel("Direct-settlement value share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.90, "(a)", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)

    ax = axes[1]
    ax.plot(h_v["t"], h_v["CS"], linestyle="--", linewidth=1.6, color="black",
            marker="s", markevery=12, markersize=3.0, markerfacecolor="white",
            label="Variable-cost channel only")
    ax.plot(h_b["t"], h_b["CS"], linestyle="-", linewidth=1.8, color="black",
            marker="s", markevery=12, markersize=3.2,
            label="Variable and operational-cost channels")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Direct-settlement count share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.90, "(b)", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "Fig4.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "Fig4.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "Fig4.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
