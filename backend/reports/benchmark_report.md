Test set: 5,605 entity windows, 5.58% anomalous

| Detector | Precision | Recall | F1 | FPR | ROC AUC | False positives |
|---|---:|---:|---:|---:|---:|---:|
| Static rule baseline | 0.458 | 0.888 | 0.604 | 0.0622 | 0.931 | 329 |
| K8Shield (Isolation Forest) | 0.815 | 0.776 | 0.795 | 0.0104 | 0.992 | 55 |

FPR reduction at default operating points: 83.3%
FPR reduction at matched recall (0.888): 63.2% (Isolation Forest FPR 0.0229 vs rules 0.0622)

| Anomaly type | Windows | Isolation Forest recall | Rules recall |
|---|---:|---:|---:|
| exec_anomaly | 86 | 0.663 | 1.000 |
| rbac_escalation | 112 | 0.902 | 1.000 |
| secret_burst | 115 | 0.739 | 0.696 |
