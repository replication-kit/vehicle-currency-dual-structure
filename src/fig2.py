from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_fig2", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_fig2"] = m
spec.loader.exec_module(m)


def main():
    p = m.ModelParams()
    res = m.run_model(p, depth_mode="endogenous")

    A = res.sizes
    W = p.long_run_window
    adopt_rate = res.adoption[-W:, :].mean(axis=0)
    A_star = m.steady_state_threshold(p)

    order = np.argsort(A)
    x = A[order]
    y = adopt_rate[order]

    pd.DataFrame({
        "relationship_size_Ai": x,
        "adoption_rate_last_W": y,
    }).to_csv(HERE / "Fig2_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.scatter(
        x, y,
        s=24,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label=f"Simulated adoption rate (last {W} periods)"
    )

    ax.axvline(
        A_star,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=rf"Analytical threshold $A^*={A_star:.2f}$"
    )

    xmin = max(x.min() * 0.85, 1e-6)
    xmax = x.max() * 1.15
    ax.plot(
        [xmin, A_star, A_star, xmax],
        [0, 0, 1, 1],
        linestyle=":",
        linewidth=1.1,
        color="0.45",
        label="Analytical classification"
    )

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Relationship size $A_i$ (log scale)")
    ax.set_ylabel(f"Direct-settlement adoption rate\n(last {W} periods)")
    ax.legend(loc="best", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "Fig2.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "Fig2.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "Fig2.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
