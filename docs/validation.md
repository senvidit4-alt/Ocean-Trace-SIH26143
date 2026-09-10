# OceanTrace Validation

Validation is organized across the three functional modules to ensure robustness.

## 1. Detection Validation (Module 1)
- Validation against ground-truth oil spill masks.
- Evaluation metrics: Intersection over Union (IoU), Precision, Recall, and F1-score.
- Testing robustness against known look-alikes.

## 2. Drift Validation (Module 2)
- **Forward Trajectory Testing**: Ensuring simulated oil behaves according to known physics (e.g., current displacement).
- **Backward Reconstruction Testing**: Ensuring a reconstructed point matches a known source.
- Validated on controlled constant-current synthetics and real-world environmental subsets (e.g., Bay Marchand historical data) without leaking true source location to the algorithm.

## 3. Attribution Validation (Module 3)
- Validating the evidence ranking against known attribution cases.
- Confirming that the highest-ranked candidate logically correlates with the synthesized evidence dimensions.
- Unit testing evidence extraction features (e.g., proper normalization of Spatial Proximity).
- Data Contracts testing: validating the structural integrity of the inter-module JSON payloads.
