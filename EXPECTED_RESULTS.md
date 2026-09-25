# Expected numerical results

## Baseline

| Quantity | Expected value |
|---|---:|
| Analytical threshold A* | 6.25 |
| First direct adoption | 63 |
| First period at direct-spread floor | 66 |
| Long-run value share | 0.7958757905 |
| Long-run count share | 0.425 |
| Largest final non-adopter size | 6.232221 |
| Smallest final adopter size | 6.293612 |

## Endogenous depth versus fixed depth

| Quantity | Endogenous | Fixed depth |
|---|---:|---:|
| First period at spread floor | 66 | 73 |
| First period VS >= 0.50 | 64 | 66 |
| First period CS >= 0.25 | 64 | 68 |
| Long-run value share | 0.7958757905 | 0.7958757905 |
| Long-run count share | 0.425 | 0.425 |

## Seed-liquidity robustness

| V_seed | First adoption | First VS >= 0.50 | First CS >= 0.25 | Long-run VS | Long-run CS |
|---:|---:|---:|---:|---:|---:|
| 10 | 72 | 73 | 73 | 0.795876 | 0.425 |
| 50 | 63 | 64 | 64 | 0.795876 | 0.425 |
| 200 | 61 | 62 | 63 | 0.795876 | 0.425 |

## Pareto-shape robustness

| a | Long-run VS | Long-run CS | Value-count wedge |
|---:|---:|---:|---:|
| 1.5 | 0.901806 | 0.545 | 0.356806 |
| 2.0 | 0.795876 | 0.425 | 0.370876 |
| 3.0 | 0.573911 | 0.250 | 0.323911 |

The wedge is not claimed to be monotonic in the Pareto shape parameter.
