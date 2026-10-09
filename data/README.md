# Data

This project uses the **IBM Telco Customer Churn** sample dataset: 7,043 customers of a
*fictional* telecom company, 21 columns, one row per customer. The raw file is **not committed**
to this repository.

## Get the file

```bash
python -m src.data_loader
```

This downloads the CSV from IBM's public GitHub repository to `data/raw/Telco-Customer-Churn.csv`
and verifies its SHA-256 checksum:

| | |
|---|---|
| Source | <https://github.com/IBM/telco-customer-churn-on-icp4d> (`data/Telco-Customer-Churn.csv`; repository archived, Apache-2.0) |
| Direct URL | `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Size | 970,457 bytes |

`python -m src.train` also downloads the file automatically if it is missing.

## No internet / URL gone?

1. Get the same dataset from another mirror, e.g. Kaggle
   [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   (file `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Save it as `data/raw/Telco-Customer-Churn.csv`.
3. Run `python -m src.data_loader`. If the checksum differs, the loader stops and says so. Mirrors
   may differ slightly, so check the cleaning report printed by `python -m src.train` before
   trusting results.

Never replace the file with generated or made-up data.

## Folders

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | the downloaded CSV, untouched | no |
| `processed/` | `telco_clean.csv`, written by `python -m src.train` after cleaning | no |
