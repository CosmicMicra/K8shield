<div align="center">

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0d1f3c,4f8ef7&height=140&section=header&text=K8Shield%20🛡&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=42&desc=Kubernetes%20Audit-Log%20Anomaly%20Detection%20·%20Isolation%20Forest%20%2B%20Elasticsearch&descAlignY=65&descColor=a0c4ff&descSize=16"/>
</div>

<br/>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-k8shield.netlify.app-4f8ef7?style=for-the-badge&logo=netlify&logoColor=white)](https://k8shield.netlify.app/)
[![Published](https://img.shields.io/badge/Published-IRJET%202024-bc8cff?style=for-the-badge&logo=google-scholar&logoColor=white)](https://www.irjet.net)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-f7931e?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

<br/>

> **Spot attacks in your Kubernetes audit logs.**
> K8Shield learns what normal API activity looks like and flags privilege escalation, suspicious pod exec and secret-scraping bursts.

</div>

---

## 🎯 What it does

K8Shield reads Kubernetes API-server audit logs and summarises behaviour per user and source IP every 5 minutes: request volume, verb mix, RBAC changes, namespaces touched, client rarity, 401/403 rate, exec and secret access. An unsupervised **Isolation Forest** scores each window, and flagged windows are sent to **Elasticsearch** with a severity.

```
Audit logs ──▶ 5-min behaviour features ──▶ Isolation Forest ──▶ Elasticsearch + alerts
                                                    │
                                                    └──▶ Benchmark vs. static rules
```

## 📊 Results

Mean of 5 runs on a synthetic benchmark with injected attacks (30% of them deliberately stealthy):

| Metric | Static rules | K8Shield |
|---|---:|---:|
| Precision | 0.43 | **0.78** |
| Recall | **0.91** | 0.77 |
| F1 | 0.59 | **0.77** |
| False positive rate | 6.6% | **1.2%** |
| ROC AUC | — | **0.99** |

At equal recall, K8Shield raises **38.5% ± 25.2%** fewer false alarms than the rules. These numbers come from synthetic data, not a production cluster. The full breakdown, plots and caveats are in [`backend/README.md`](backend/README.md).

<p align="center">
  <img src="backend/reports/roc_curve.png" width="820" alt="ROC curve: K8Shield vs static rules"><br>
  <img src="backend/reports/confusion_matrices.png" width="820" alt="Confusion matrices: K8Shield vs static rules">
</p>

## 🚀 Quick start

```bash
git clone https://github.com/CosmicMicra/k8shield.git
cd k8shield/backend
pip install -r requirements.txt
python main.py run
```

This generates labelled audit logs, trains the model, evaluates it against the rule baseline, and writes metrics and plots to `backend/reports/`. It takes about 15 seconds.

## 📁 Repository

| Path | Contents |
|---|---|
| [`backend/`](backend/) | Python pipeline: log generator, feature extraction, model training, Elasticsearch ingestion, evaluation, tests |
| [`site/`](site/) | The static project page served at [k8shield.netlify.app](https://k8shield.netlify.app/) |

## 🛠 Tech stack

Python · scikit-learn · pandas · NumPy · matplotlib · Elasticsearch

---

## 📄 Publication

**K8Shield: Real-Time Kubernetes Security Monitoring using Machine Learning**
*Soniya Phaltane (First Author) · IRJET Volume 11, Issue 9 · 2024*

This repository is a rebuilt, reproducible implementation of the pipeline.

## 👩‍💻 Author

**Soniya Phaltane** · [@CosmicMicra](https://github.com/CosmicMicra)
ML Engineer · AI Security · [soniyaphaltane-portfolio.netlify.app](https://soniyaphaltane-portfolio.netlify.app)

---

<div align="center">
  <i>If you find this useful, leave a ⭐ — it helps!</i>
</div>
