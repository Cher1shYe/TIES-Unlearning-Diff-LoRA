# Mechanism probe (seed 42, pre-FT, shared phase-2 weights)

Selected layers: encoder.layer.8, encoder.layer.9, encoder.layer.10, encoder.layer.11

| Condition | MNLI | HANS | H-ent | H-non | ΔH-non | ΔMNLI |
|---|---:|---:|---:|---:|---:|---:|
| p_only(control) | 85.04 | 57.25 | 97.97 | 16.53 | +0.00 | +0.00 |
| full/selected/b0.5 | 85.02 | 57.21 | 98.08 | 16.33 | -0.19 | -0.02 |
| full/selected/b1 | 85.02 | 57.81 | 97.67 | 17.96 | +1.43 | -0.02 |
| full/selected/b2 | 81.12 | 56.09 | 57.44 | 54.73 | +38.21 | -3.92 |
| full/global/b0.5 | 84.96 | 57.93 | 97.61 | 18.25 | +1.73 | -0.08 |
| full/global/b1 | 84.30 | 61.01 | 93.75 | 28.27 | +11.74 | -0.74 |
| full/global/b2 | 54.40 | 50.70 | 7.86 | 93.53 | +77.01 | -30.64 |
| naive/selected/b0.5 | 84.82 | 55.84 | 98.70 | 12.98 | -3.55 | -0.22 |
| naive/selected/b1 | 82.76 | 53.39 | 98.98 | 7.79 | -8.73 | -2.28 |
| naive/selected/b2 | 47.44 | 49.99 | 99.78 | 0.21 | -16.32 | -37.60 |
| naive/global/b0.5 | 83.68 | 58.28 | 97.52 | 19.04 | +2.51 | -1.36 |
| naive/global/b1 | 65.22 | 51.56 | 94.79 | 8.32 | -8.21 | -19.82 |
| naive/global/b2 | 35.22 | 49.97 | 99.67 | 0.27 | -16.25 | -49.82 |
