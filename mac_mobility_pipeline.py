"""Minimal public pipeline for MAC-Mobility calibration and DIMNet-CPC OD generation."""

from __future__ import annotations

import argparse
import re
from datetime import timedelta
from pathlib import Path

import chinese_calendar as calendar
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


RANDOM_STATE = 42
FEATURES = [
    "Adjusted_BMI",
    "Days_to_Spring_Festival",
    "Statutory_Holiday",
    "Ordinary_Weekend",
    "Compensatory_Workday",
    "Holiday_Peak",
]
CATEGORICAL_FEATURES = [
    "Statutory_Holiday",
    "Ordinary_Weekend",
    "Compensatory_Workday",
    "Holiday_Peak",
]
SPRING_FESTIVAL_EVE = {
    2020: "2020-01-24",
    2021: "2021-02-11",
    2022: "2022-01-31",
    2023: "2023-01-21",
    2024: "2024-02-09",
    2025: "2025-01-28",
}


def parse_dates(values: pd.Series) -> pd.Series:
    digits = values.astype(str).str.replace(r"\D", "", regex=True).str[:8]
    parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(values, errors="coerce"))


def rename_aliases(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    rename = {}
    for canonical, choices in aliases.items():
        match = next((name for name in choices if name in frame.columns), None)
        if match is not None:
            rename[match] = canonical
    return frame.rename(columns=rename)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    def one_day(timestamp: pd.Timestamp) -> tuple[int, int, int, int, int]:
        day = timestamp.date()
        on_holiday, holiday_name = calendar.get_holiday_detail(day)
        festive = int(on_holiday and holiday_name is not None)
        extra_workday = int(day.weekday() >= 5 and calendar.is_workday(day))
        normal_weekend = int(
            day.weekday() >= 5
            and not calendar.is_workday(day)
            and holiday_name is None
        )
        peak = int(
            calendar.is_holiday(day)
            and (
                not calendar.is_holiday(day - timedelta(days=1))
                or not calendar.is_holiday(day + timedelta(days=1))
            )
        )
        spring_festival = SPRING_FESTIVAL_EVE.get(day.year)
        distance = 999
        if spring_festival:
            distance_raw = (timestamp - pd.Timestamp(spring_festival)).days
            if -15 <= distance_raw <= 25:
                distance = distance_raw
        return distance, festive, normal_weekend, extra_workday, peak

    values = frame["Date"].apply(one_day)
    frame[
        [
            "Days_to_Spring_Festival",
            "Statutory_Holiday",
            "Ordinary_Weekend",
            "Compensatory_Workday",
            "Holiday_Peak",
        ]
    ] = pd.DataFrame(values.tolist(), index=frame.index)
    return frame


def read_national_index(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    frame = rename_aliases(
        frame,
        {
            "Date": ["Date", "日期", "时间"],
            "Adjusted_BMI": ["Adjusted_BMI", "修正后迁徙指数"],
        },
    )
    required = {"Date", "Adjusted_BMI"}
    if not required.issubset(frame.columns):
        raise ValueError(f"National index must contain {sorted(required)}.")
    frame = frame[["Date", "Adjusted_BMI"]].copy()
    frame["Date"] = parse_dates(frame["Date"])
    frame["Adjusted_BMI"] = pd.to_numeric(frame["Adjusted_BMI"], errors="coerce")
    frame = frame.dropna().drop_duplicates("Date").sort_values("Date")
    if (frame["Adjusted_BMI"] <= 0).any():
        raise ValueError("Adjusted_BMI values must be positive.")
    frame["Month"] = frame["Date"].dt.strftime("%Y%m")
    return frame


def prepare_calibration(metadata: Path, national: pd.DataFrame):
    anchors = pd.read_excel(metadata, sheet_name="Calibration_Anchors")
    required = {
        "Anchor_Type",
        "Period",
        "MoT_Total_10k_Person_Times",
        "Role",
        "Assigned_Training_Weight",
    }
    if not required.issubset(anchors.columns):
        raise ValueError(f"Calibration_Anchors must contain {sorted(required)}.")

    is_daily = anchors["Anchor_Type"].astype(str).str.contains("daily", case=False)
    daily = anchors.loc[is_daily].copy()
    monthly = anchors.loc[~is_daily].copy()
    daily["Date"] = parse_dates(daily["Period"])
    daily["Target"] = pd.to_numeric(
        daily["MoT_Total_10k_Person_Times"], errors="coerce"
    )
    daily = daily.merge(national[["Date", "Adjusted_BMI"]], on="Date", how="left")
    if daily[["Date", "Target", "Adjusted_BMI"]].isna().any().any():
        raise ValueError("A daily anchor could not be matched to the national index.")

    direct_weight = pd.to_numeric(
        daily["Assigned_Training_Weight"], errors="coerce"
    ).max()
    daily["Weight"] = direct_weight
    train_daily = daily[daily["Role"].eq("Training")].copy()
    test_daily = daily[daily["Role"].eq("Hold-out validation")].copy()
    if len(train_daily) != 88 or len(test_daily) != 22:
        raise ValueError("Expected the published 88/22 daily training/hold-out split.")

    monthly["Month"] = parse_dates(monthly["Period"]).dt.strftime("%Y%m")
    monthly["Month_Total"] = pd.to_numeric(
        monthly["MoT_Total_10k_Person_Times"], errors="coerce"
    )
    month_totals = monthly.set_index("Month")["Month_Total"]
    pseudo = national[national["Month"].isin(month_totals.index)].copy()
    pseudo["Target"] = (
        pseudo["Adjusted_BMI"]
        / pseudo.groupby("Month")["Adjusted_BMI"].transform("sum")
        * pseudo["Month"].map(month_totals)
    )
    pseudo["Weight"] = pd.to_numeric(
        monthly["Assigned_Training_Weight"], errors="coerce"
    ).dropna().iloc[0]
    if pseudo["Month"].nunique() != len(month_totals) or pseudo["Target"].isna().any():
        raise ValueError("Not all selected monthly anchors could be disaggregated.")

    train = add_calendar_features(pd.concat([train_daily, pseudo], ignore_index=True))
    test = add_calendar_features(test_daily)
    full = add_calendar_features(pd.concat([daily, pseudo], ignore_index=True))
    return train, test, full


def model_tournament(train: pd.DataFrame, test: pd.DataFrame):
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=12, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "CatBoost": CatBoostRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = np.log1p(train["Target"]), test["Target"].to_numpy()
    metrics, fitted = [], {}
    for name, model in models.items():
        kwargs = {"sample_weight": train["Weight"]}
        if name == "CatBoost":
            kwargs["cat_features"] = CATEGORICAL_FEATURES
        model.fit(x_train, y_train, **kwargs)
        prediction = np.expm1(model.predict(x_test))
        metrics.append(
            {
                "Model": name,
                "R2": r2_score(y_test, prediction),
                "WAPE": np.abs(y_test - prediction).sum() / np.abs(y_test).sum(),
            }
        )
        fitted[name] = model
    metrics = pd.DataFrame(metrics).sort_values("R2", ascending=False).reset_index(drop=True)
    return metrics, metrics.loc[0, "Model"], fitted[metrics.loc[0, "Model"]]


def refit(model, model_name: str, full: pd.DataFrame):
    kwargs = {"sample_weight": full["Weight"]}
    if model_name == "CatBoost":
        kwargs["cat_features"] = CATEGORICAL_FEATURES
    model.fit(full[FEATURES], np.log1p(full["Target"]), **kwargs)
    return model


def read_od_file(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    frame = rename_aliases(
        frame,
        {
            "Unit": ["Unit", "所在城市"],
            "Inflow_Origin": ["Inflow_Origin", "迁入来源地"],
            "Outflow_Destination": ["Outflow_Destination", "迁出目的地"],
            "Inflow_Total_Index": ["Inflow_Total_Index", "当日迁入总规模"],
            "Outflow_Total_Index": ["Outflow_Total_Index", "当日迁出总规模"],
            "Inflow_Share_Percent": ["Inflow_Share_Percent", "迁入比例"],
            "Outflow_Share_Percent": ["Outflow_Share_Percent", "迁出比例"],
        },
    )
    if "Inflow_Share_Percent" not in frame or "Outflow_Share_Percent" not in frame:
        share_columns = [column for column in frame.columns if "比例" in str(column)]
        if len(share_columns) >= 2:
            frame = frame.rename(
                columns={
                    share_columns[0]: "Inflow_Share_Percent",
                    share_columns[1]: "Outflow_Share_Percent",
                }
            )
    return frame


def generate_one_day(
    path: Path, date: pd.Timestamp, national_row: pd.Series, model
) -> pd.DataFrame:
    frame = read_od_file(path)
    prediction_row = national_row[FEATURES].astype(float).to_frame().T
    factor = (
        max(0.0, float(np.expm1(model.predict(prediction_row)[0])))
        * 10_000
        / float(national_row["Adjusted_BMI"])
    )
    candidates = []

    inflow_columns = {
        "Unit",
        "Inflow_Origin",
        "Inflow_Total_Index",
        "Inflow_Share_Percent",
    }
    if inflow_columns.issubset(frame.columns):
        part = frame[list(inflow_columns)].dropna().copy()
        part["Origin"] = part["Inflow_Origin"]
        part["Destination"] = part["Unit"]
        part["Flow"] = (
            pd.to_numeric(part["Inflow_Total_Index"], errors="coerce")
            * pd.to_numeric(part["Inflow_Share_Percent"], errors="coerce")
            / 100
            * factor
        )
        candidates.append(part[["Origin", "Destination", "Flow"]])

    outflow_columns = {
        "Unit",
        "Outflow_Destination",
        "Outflow_Total_Index",
        "Outflow_Share_Percent",
    }
    if outflow_columns.issubset(frame.columns):
        part = frame[list(outflow_columns)].dropna().copy()
        part["Origin"] = part["Unit"]
        part["Destination"] = part["Outflow_Destination"]
        part["Flow"] = (
            pd.to_numeric(part["Outflow_Total_Index"], errors="coerce")
            * pd.to_numeric(part["Outflow_Share_Percent"], errors="coerce")
            / 100
            * factor
        )
        candidates.append(part[["Origin", "Destination", "Flow"]])

    if not candidates:
        raise ValueError(f"No supported OD columns found in {path.name}.")
    result = pd.concat(candidates, ignore_index=True).dropna()
    result = result[result["Flow"] > 0]
    result = result.groupby(["Origin", "Destination"], as_index=False)["Flow"].min()
    result["Flow"] = result["Flow"].round().astype("int64")
    result.insert(0, "Date", date.strftime("%Y-%m-%d"))
    return result


def generate_networks(
    od_dir: Path,
    output_dir: Path,
    national: pd.DataFrame,
    model,
    daily_excel: bool,
) -> None:
    featured_national = add_calendar_features(national)
    by_date = featured_national.set_index("Date")
    records = []
    excel_dir = output_dir / "daily_excel"
    if daily_excel:
        excel_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(od_dir.glob("*.xlsx")):
        match = re.search(r"(\d{8})", path.stem)
        if match is None:
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        if date not in by_date.index:
            continue
        result = generate_one_day(path, date, by_date.loc[date], model)
        records.append(result)
        if daily_excel:
            result.drop(columns="Date").to_excel(
                excel_dir / f"DIMNet_CPC_{match.group(1)}.xlsx", index=False
            )

    if not records:
        raise ValueError("No dated OD workbooks could be processed.")
    pd.concat(records, ignore_index=True).to_parquet(
        output_dir / "DIMNet-CPC_2020-2025.parquet",
        index=False,
        compression="snappy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--national-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--od-dir", type=Path)
    parser.add_argument("--daily-excel", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    national = read_national_index(args.national_index)
    train, test, full = prepare_calibration(args.metadata, national)
    metrics, champion_name, champion = model_tournament(train, test)
    metrics.to_csv(args.output_dir / "holdout_model_metrics.csv", index=False)
    champion = refit(champion, champion_name, full)
    joblib.dump(
        {"model": champion, "model_name": champion_name, "features": FEATURES},
        args.output_dir / "mac_mobility_model.joblib",
    )
    print(metrics.to_string(index=False))
    print(f"Selected model: {champion_name}")

    if args.od_dir is not None:
        generate_networks(
            args.od_dir, args.output_dir, national, champion, args.daily_excel
        )


if __name__ == "__main__":
    main()
