# AI, Direct Settlement, and the Persistence of Vehicle Currency: A Dual-Structure Mechanism

This repository contains the simulation code used in the paper:

**“AI, Direct Settlement, and the Persistence of Vehicle Currency: A Dual-Structure Mechanism”**

The model is an agent-based simulation of settlement currency choice between a vehicle route (USD) and direct settlement. It illustrates a “dual-structure” outcome: value shares can shift toward direct settlement while count shares remain more inertial.

## Contents

* `simulation.py` — main simulation script (runs the model and generates the figure)
* `requirements.txt` — Python dependencies

## Requirements

* Python 3.10+ (recommended)
* Packages listed in `requirements.txt`

## Setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the simulation and generate the figure.

```bash
python simulation.py
```

## Reproducibility

* The random seed is fixed in the script (`np.random.seed(123)`) to ensure reproducibility of the simulation outcomes and the generated figure.

## Customization

Key parameters (baseline calibration) are defined inside `run_simulation()` in `simulation.py`.
To run alternative configurations (e.g., for robustness checks), edit the `params` dictionary and re-run:

* Liquidity and inventory cost parameters: `alpha`, `delta`, `s_base`, `eta`
* Replication floor: `kappa`, `usd_spread`
* Fixed administrative cost level: `base_threshold_fixed`
* Exogenous liquidity seed: `initial_volume_seed`
* Simulation horizon / population: `steps`, `n_agents`

## Notes

* The simulation is intended to illustrate the mechanism and comparative statics; it is not designed to forecast time-series outcomes.

## Citation

If you use this code, please cite the paper:

> AI, Direct Settlement, and the Persistence of Vehicle Currency: A Dual-Structure Mechanism.

A BibTeX entry can be added here once the working paper / journal version is finalized.

## License

Add your preferred license (e.g., MIT, BSD-3, or CC BY-NC for code/data packages) and include a `LICENSE` file.

## Contact

For questions or replication issues, please open an issue in this repository.