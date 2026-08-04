# MAC-Mobility and DIMNet-CPC generation code

This repository contains the minimal public implementation used to calibrate national daily mobility totals and generate the released DIMNet-CPC OD networks.

## Files

- `mac_mobility_pipeline.py`: calibration, model tournament, full-data refit, OD reconciliation, and OD-network generation.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License for the code.

The released data, calibration anchors, external-validation records, and data documentation are deposited on Figshare and should be cited using the DOI shown in the associated data record.

## Required inputs

1. `DIMNet-CPC_Metadata_and_Validation.xlsx` from Figshare. The script reads its `Calibration_Anchors` sheet, including the frozen 88/22 daily training/hold-out assignment and the nine selected monthly totals.
2. A national LBS-index workbook with `Date` and `Adjusted_BMI` columns. The Chinese aliases `日期` and `修正后迁徙指数` are also accepted.
3. Optional daily OD-input workbooks in one directory. Each filename must contain `YYYYMMDD`. Supported fields are:

| English field | Accepted Chinese field |
|---|---|
| `Unit` | `所在城市` |
| `Inflow_Origin` | `迁入来源地` |
| `Outflow_Destination` | `迁出目的地` |
| `Inflow_Total_Index` | `当日迁入总规模` |
| `Outflow_Total_Index` | `当日迁出总规模` |
| `Inflow_Share_Percent` | `迁入比例` or the first column containing `比例` |
| `Outflow_Share_Percent` | `迁出比例` or the second column containing `比例` |

The underlying Baidu source records are not redistributed in this repository.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Run calibration only

```bash
python mac_mobility_pipeline.py \
  --metadata DIMNet-CPC_Metadata_and_Validation.xlsx \
  --national-index national_adjusted_bmi.xlsx \
  --output-dir output
```

## Run calibration and OD generation

```bash
python mac_mobility_pipeline.py \
  --metadata DIMNet-CPC_Metadata_and_Validation.xlsx \
  --national-index national_adjusted_bmi.xlsx \
  --od-dir od_inputs \
  --output-dir output \
  --daily-excel
```

The pipeline writes `holdout_model_metrics.csv`, `mac_mobility_model.joblib`, and, when `--od-dir` is supplied, `DIMNet-CPC_2020-2025.parquet`. The optional `--daily-excel` flag also writes one Excel OD table per processed date.

## Methodological behavior preserved

- The nine monthly totals are deterministically allocated across their constituent dates in proportion to within-month adjusted BMI shares.
- Direct daily training records and monthly-derived pseudo-daily records receive weights of 3.0 and 1.0, respectively.
- Ridge, Random Forest, XGBoost, and CatBoost are compared on the 22 held-out directly observed daily records; the model with the highest hold-out R² is selected, while WAPE is also reported.
- The selected model is refitted on all 110 daily anchors and all pseudo-daily constraints before production estimation.
- When both disclosed directional records support the same OD link, the smaller estimate is retained. Singly disclosed links are retained.
- Undisclosed links are not imputed, and retained OD links are not renormalized to the national total.

## Licenses

The code is released under the MIT License. The Figshare dataset and author-created metadata use their separately stated data license. Third-party source products remain subject to the terms of their respective providers.
