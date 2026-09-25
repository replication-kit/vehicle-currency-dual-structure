from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_fig1", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_fig1"] = m
spec.loader.exec_module(m)


def main():
    p = m.ModelParams()
    res = m.run_model(p, depth_mode="endogenous")
    h = res.history.copy()

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True)

    # Panel (a): technology and settlement costs
    ax = axes[0]
    ax.plot(h["t"], h["S_dir"], linestyle="-", linewidth=1.8, color="black",
            label="Direct-route spread")
    ax.plot(h["t"], [p.S_USD] * len(h), linestyle="--", linewidth=1.4, color="black",
            label="Vehicle-route cost")
    ax.plot(h["t"], [p.kappa * p.S_USD] * len(h), linestyle=":", linewidth=1.5, color="black",
            label="Replication floor")
    ax.set_ylabel("Settlement cost")
    ax2 = ax.twinx()
    ax2.plot(h["t"], h["z"], linestyle="-.", linewidth=1.3, color="0.45",
             label="Technology index")
    ax2.set_ylabel("Technology index")
    ax2.set_ylim(0, 1.05)
    ax.text(0.01, 0.92, "(a)", transform=ax.transAxes)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", frameon=False)

    # Panel (b): endogenous depth formation
    ax = axes[1]
    ax.plot(h["t"], h["V_eff"], linestyle="-", linewidth=1.8, color="black",
            label="Effective direct-market depth")
    ax.plot(h["t"], h["V_dir"], linestyle="--", linewidth=1.5, color="black",
            label="Direct-settlement volume")
    ax.axhline(p.V_seed, linestyle=":", linewidth=1.3, color="0.4",
               label=r"Liquidity seed $V_{\mathrm{seed}}$")
    ax.set_ylabel("Market depth / volume")
    ax.text(0.01, 0.92, "(b)", transform=ax.transAxes)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.58), frameon=False)

    # Panel (c): aggregate adoption
    ax = axes[2]
    ax.plot(h["t"], h["VS"], linestyle="-", linewidth=1.8, color="black",
            marker="o", markevery=12, markersize=3.5, label="Value share")
    ax.plot(h["t"], h["CS"], linestyle="--", linewidth=1.5, color="black",
            marker="s", markevery=12, markersize=3.2, label="Count share")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Direct-settlement share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.92, "(c)", transform=ax.transAxes)
    ax.legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(False)

    fig.tight_layout()

    fig.savefig(HERE / "Fig1.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "Fig1.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "Fig1.pdf", format="pdf", bbox_inches="tight")
    h[["t","z","S_dir","S_floor","V_eff","V_dir","VS","CS"]].to_csv(HERE / "Fig1_data.csv", index=False)


if __name__ == "__main__":
    main()
