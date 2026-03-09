# Predicting and Optimizing Cloud Infrastructure Costs

**Sai Teja Sunku**
MSDS 692 — Data Science Practicum I
Regis University, Spring 2026

---

## what this project is about

Cloud bills are messy. Most companies dont really know what theyre going to pay next month, they cant explain why last months bill was so high, and they havent done the math on how much they could save by switching pricing plans. This project tries to fix all three of those problems using machine learning.

I built a pipeline that does four things:
- predicts what any EC2 instance configuration will cost per hour (regression)
- forecasts daily cloud spending for the next few weeks (time series)
- flags instances that are overpriced and days where spending looks abnormal (anomaly detection)
- calculates exactly how much you could save by switching from on-demand to reserved or spot pricing (cost optimization)

I used two datasets from two different cloud providers because thats how it works in practice — most companies run on multiple clouds.

---

## datasets

**AWS EC2 Pricing** — 7,260 rows of instance-level pricing across 121 instance types, 5 regions, 4 pricing models, and 3 operating systems. This is a structured pricing catalog from Kaggle.
- source: https://www.kaggle.com/datasets/berkayalan/aws-ec2-instance-comparison

**Azure Anonymized Billing** — 93,605 line items of actual daily cloud spending across 50 services over 89 days (Dec 2022 to Mar 2023). I aggregated this down to daily totals for the forecasting and anomaly detection work.
- source: https://www.kaggle.com/datasets/carrucciu/azure-costs

---

## whats in this repo

```
├── DATA SCIENCE.ipynb                  # the main notebook — everything runs from here
├── aws_infrastructure_costs.csv  # AWS EC2 pricing dataset
├── anonymized_costs.csv          # Azure daily billing dataset
├── main_updated.tex              # IEEE format paper (LaTeX)
├── references.bib                # bibliography file for the paper
└── README.md                     # this file
```

---

## how to run it

you need python 3.11+ and these packages:

```
pip install flaml catboost prophet statsmodels scikit-learn pandas numpy matplotlib seaborn
```

then just open `DSP_11.ipynb` in jupyter and run all cells top to bottom. the whole thing takes about 5 minutes — most of that is the FLAML automl training (2 min time budget).

make sure both csv files are in the same folder as the notebook.

---

## project walkthrough

### part 1 — AWS EC2 pricing

**EDA:** looked at cost distributions (heavily right-skewed, median $0.57/hr but mean $1.43 because of expensive GPU instances), regional pricing differences (us-east-1 is cheapest), windows vs linux premium (~20%), and feature correlations (vCPUs and memory both >0.85 correlation with cost).

**AutoML regression:** used FLAML with a 120-second time budget. it tried CatBoost, XGBoost, Random Forest, and Extra Trees with automatic hyperparameter tuning. CatBoost won. test set results:
- R² = 0.9999
- MAE = $0.0012/hr
- RMSE = $0.0089/hr
- MAPE < 2%

the R² looks crazy high but it makes sense — AWS pricing is basically a lookup table. more hardware = more cost. its not noisy data like stock prices or user behavior.

**Pricing anomaly detection:** ran Isolation Forest on the price-to-value features (vCPUs, memory, network, cost, cost-per-vCPU, cost-per-GiB). flagged ~5% of instances as overpriced. those instances cost about 9x more than similar configs on average.

**Cost optimization:** compared on-demand vs reserved vs spot pricing across all instance families. reserved 1-year saves ~40%, reserved 3-year saves ~60%, spot saves ~70%. for a company spending $50K/month thats $240K-$420K in annual savings. also ranked instance families by efficiency (compute per dollar).

### part 2 — Azure billing (time series)

**Data prep:** aggregated 93K line items down to 89 daily cost totals. clear upward trend from ~$70/day to ~$130/day over 3 months. weekly seasonality visible too (lower on weekends).

**Prophet forecasting:** configured with weekly seasonality on, yearly off (only 3 months of data), changepoint prior scale 0.3. MAPE around 6-12% on the test set.

**ARIMA forecasting:** ran ADF test (non-stationary as expected), checked ACF/PACF to pick orders, tried (1,1,1) and (5,1,2) before settling on (2,1,2). MAPE around 8-15%.

**Cost anomaly detection:** engineered temporal features (day of week, rolling 7-day stats, percent change, etc) and ran Isolation Forest. flagged ~5% of days. drilled into which services caused the spikes by checking the most volatile services.

---

## results summary

| task | metric | target | result |
|------|--------|--------|--------|
| regression | R² | > 0.80 | 0.9999 |
| prophet forecast | MAPE | < 15% | 6-12% |
| arima forecast | MAPE | < 15% | 8-15% |
| pricing anomalies | flag outliers | — | 5% flagged |
| cost anomalies | flag spikes | — | 5% flagged |
| cost savings | quantify | actionable | 40-70% |

all targets met.

---

## limitations

- 89 days isnt a lot for time series — cant learn yearly patterns, would want 12+ months for production
- azure billing data is from one anonymous org, might not generalize
- no ground truth for anomaly detection — the 5% contamination rate is a hyperparameter choice not a verified number
- AWS pricing is a snapshot, it changes over time
- only gave FLAML 2 minutes — more time might find slightly better configs (though theres not much room at R²=0.9999)

---

## future work

- connect to AWS Cost Explorer API / Azure Cost Management API for live data instead of static CSVs
- build automated alerting (slack/email when forecasts predict budget overruns or anomalies are detected)
- try LSTM or transformer models for forecasting with more data
- build a streamlit dashboard so non-technical people can use it without jupyter
- get 12+ months of billing data to properly validate the time series models

---

## references

- FLAML: https://github.com/microsoft/FLAML
- CatBoost paper: https://arxiv.org/abs/1706.09516
- Prophet paper: https://peerj.com/preprints/3190/
- Isolation Forest paper: https://ieeexplore.ieee.org/document/4781136
- scikit-learn: https://jmlr.org/papers/v12/pedregosa11a.html
- AWS EC2 pricing: https://aws.amazon.com/ec2/pricing/
- Flexera 2024 State of the Cloud: https://www.flexera.com/blog/finops/cloud-computing-trends-flexera-2024-state-of-the-cloud-report/
- FinOps Foundation: https://www.finops.org/introduction/what-is-finops/
