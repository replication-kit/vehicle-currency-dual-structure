from pathlib import Path
import importlib.util
import sys
from dataclasses import replace
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.py"

spec = importlib.util.spec_from_file_location("settlement_model_figc1", MODEL_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["settlement_model_figc1"] = m
spec.loader.exec_module(m)


def main():
    p = m.ModelParams()
    sizes = m.draw_relationship_sizes(p)

    seed_values = [10.0, 50.0, 200.0]
    results = {}

    for vseed in seed_values:
        pp = replace(p, V_seed=vseed)
        res = m.run_model(pp, sizes=sizes, depth_mode="endogenous")
        results[vseed] = res.history.copy()

    data = pd.DataFrame({"t": results[50.0]["t"]})
    for vseed, h in results.items():
        tag = str(int(vseed))
        data[f"VS_Vseed_{tag}"] = h["VS"]
        data[f"CS_Vseed_{tag}"] = h["CS"]
        data[f"Sdir_Vseed_{tag}"] = h["S_dir"]
        data[f"Veff_Vseed_{tag}"] = h["V_eff"]
    data.to_csv(HERE / "FigC1_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    line_specs = {
        10.0: dict(linestyle=":", marker="^", markerfacecolor="white"),
        50.0: dict(linestyle="-", marker="o", markerfacecolor="black"),
        200.0: dict(linestyle="--", marker="s", markerfacecolor="white"),
    }

    ax = axes[0]
    for vseed in seed_values:
        h = results[vseed]
        spec_line = line_specs[vseed]
        ax.plot(
            h["t"], h["VS"],
            color="black",
            linewidth=1.6 if vseed != 50.0 else 1.8,
            linestyle=spec_line["linestyle"],
            marker=spec_line["marker"],
            markevery=12,
            markersize=3.2,
            markerfacecolor=spec_line["markerfacecolor"],
            label=rf"$V_{{\mathrm{{seed}}}}={int(vseed)}$",
        )
    ax.set_ylabel("Direct-settlement value share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.90, "(a)", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)

    ax = axes[1]
    for vseed in seed_values:
        h = results[vseed]
        spec_line = line_specs[vseed]
        ax.plot(
            h["t"], h["CS"],
            color="black",
            linewidth=1.6 if vseed != 50.0 else 1.8,
            linestyle=spec_line["linestyle"],
            marker=spec_line["marker"],
            markevery=12,
            markersize=3.2,
            markerfacecolor=spec_line["markerfacecolor"],
            label=rf"$V_{{\mathrm{{seed}}}}={int(vseed)}$",
        )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Direct-settlement count share")
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 0.90, "(b)", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(HERE / "FigC1.png", dpi=600, bbox_inches="tight")
    fig.savefig(HERE / "FigC1.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "FigC1.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
