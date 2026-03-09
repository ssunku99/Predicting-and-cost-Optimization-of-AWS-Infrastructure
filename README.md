# Predicting and Optimizing Cloud Infrastructure Costs

**Sai Teja Sunku** — MSDS 692 Data Science Practicum I, Regis University, Spring 2026

---

## about

This project builds an ML pipeline that predicts EC2 instance costs, forecasts daily cloud spending, detects pricing anomalies, and quantifies how much money organizations leave on the table by sticking with on-demand pricing. It uses two datasets from two cloud providers — AWS for regression and cost optimization, Azure for time series forecasting and anomaly detection.

## datasets

| Dataset | Rows | Source |
|---------|------|--------|
| AWS EC2 Pricing | 7,260 | [Kaggle — Berkay Alan](https://www.kaggle.com/datasets/berkayalan/aws-ec2-instance-comparison) |
| Azure Billing (anonymized) | 93,605 | [Kaggle — Carrucciu](https://www.kaggle.com/datasets/carrucciu/azure-costs) |

The AWS CSV is included in this repo. The Azure CSV is too large for GitHub — download it from the Kaggle link above and place `anonymized_costs.csv` in the same directory as the notebook.

## what the notebook does

**Part 1 — AWS EC2 Pricing**
- EDA: cost distributions, pricing model savings, regional pricing, correlation analysis
- AutoML regression using FLAML (Microsoft's AutoML library) — CatBoost selected as best model, R² = 0.9999
- Pricing anomaly detection using Isolation Forest — flags ~5% of instances as overpriced
- Cost optimization analysis — reserved instances save 40-60%, spot saves ~70%

**Part 2 — Azure Billing**
- Daily cost aggregation and service breakdown
- Prophet forecasting (MAPE 6-12%) and ARIMA(2,1,2) forecasting (MAPE 8-15%)
- Cost anomaly detection using Isolation Forest with engineered temporal features
- Service-level volatility analysis

## results

| Task | Target | Result |
|------|--------|--------|
| Regression R² | > 0.80 | 0.9999 |
| Prophet MAPE | < 15% | 6-12% |
| ARIMA MAPE | < 15% | 8-15% |
| Pricing anomalies | flag outliers | ~5% flagged |
| Cost savings | quantify | 40-70% |

## how to run

```
pip install -r requirements.txt
```

Then open the notebook and run all cells. Make sure both CSV files are in the same directory:
- `aws_infrastructure_costs.csv` (included in repo)
- `anonymized_costs.csv` (download from Kaggle link above)

## repo structure

```
├── DSP_11.ipynb                  # main notebook
├── aws_infrastructure_costs.csv  # AWS EC2 pricing data
├── main.tex                      # IEEE paper (LaTeX source)
├── references.bib                # paper references
├── requirements.txt              # Python dependencies
└── README.md
```

## tools used

- **FLAML** — AutoML (model selection + hyperparameter tuning)
- **CatBoost** — best regression model selected by FLAML
- **Prophet** — time series forecasting
- **ARIMA** — time series forecasting (statsmodels)
- **Isolation Forest** — unsupervised anomaly detection (scikit-learn)
- **pandas, numpy, matplotlib, seaborn** — data work and visualization
