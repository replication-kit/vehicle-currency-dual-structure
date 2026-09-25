from pathlib import Path
import importlib.util
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_fig3", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_fig3"] = m
spec.loader.exec_module(m)


def main():
    p = m.ModelParams()
    sizes = m.draw_relationship_sizes(p)

    res_on = m.run_model(p, sizes=sizes, depth_mode="endogenous")
    res_off = m.run_model(p, sizes=sizes, depth_mode="fixed", fixed_depth=p.V_seed)

    h_on = res_on.history.copy()
    h_off = res_off.history.copy()

    pd.DataFrame({
        "t": h_on["t"],
        "Sdir_endogenous": h_on["S_dir"],
        "Sdir_fixed_depth": h_off["S_dir"],
        "VS_endogenous": h_on["VS"],
        "VS_fixed_depth": h_off["VS"],
        "CS_endogenous": h_on["CS"],
        "CS_fixed_depth": h_off["CS"],
        "Veff_endogenous": h_on["V_eff"],
        "Veff_fixed_depth": h_off["V_eff"],
    }).to_csv(HERE / "Fig3_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(h_on["t"], h_on["S_dir"], linestyle="-", linewidth=1.8, color="black",
            label="Direct spread: endogenous depth")
    ax.plot(h_off["t"], h_off["S_dir"], linestyle="--", linewidth=1.6, color="black",
            label="Direct spread: fixed depth")
    ax.axhline(p.kappa * p.S_USD, linestyle=":", linewidth=1.4, color="0.45",
               label="Replication floor")
    ax.set_ylabel("Direct-route spread")
    ax.text(0.01, 0.90, "(a)", transform=ax.transAxes)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(False)

    ax = axes[1]
    ax.plot(h_on["t"], h_on["VS"], linestyle="-", linewidth=1.8, color="black",
            marker="o", markevery=12, markersize=3.3,
            label="Value share: endogenous depth")
    ax.plot(h_off["t"], h_off["VS"], linestyle="--", linewidth=1.6, color="black",
            marker="o", markevery=12, markersize=3.0, markerfacecolor="white",
            label="Value share: fixed depth")
    ax.plot(h_on["t"], h_on["CS"], linestyle="-.", linewidth=1.6, color="0.30",
            marker="s", markevery=12, markersize=3.0,
            label="Count share: endogenous depth")
    ax.plot(h_off["t"], h_off["CS"], linestyle=":", linewidth=1.8, color="0.30",
            marker="s", markevery=12, markersize=2.8, markerfacecolor="white",
            label="Count share: fixed depth")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Direct-settlement share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.90, "(b)", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "Fig3.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "Fig3.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "Fig3.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
