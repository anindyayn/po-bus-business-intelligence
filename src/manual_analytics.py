from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    Metric,
    SKLEARN_AVAILABLE,
    STATSMODELS_AVAILABLE,
    clean_for_export,
    format_workbook,
)
from s1_operasional import FORECAST_HORIZON_DAYS, run_s1_operasional
from s2_perawatan_armada import run_s2_perawatan_armada
from s3_keuangan import run_s3_keuangan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "po_bus_dummy_datasets_rich.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
EXCEL_PATH = OUTPUT_DIR / "analytics_results.xlsx"

REQUIRED_SHEETS = {
    "S1_Operasional": [
        "operational_id",
        "date",
        "route_id",
        "route_name",
        "departure_slot",
        "fleet_id",
        "tickets_sold",
        "occupancy_rate",
        "recommended_frequency",
        "delay_minutes",
        "complaint_category",
        "feedback_text_clean",
    ],
    "S2_Perawatan_Armada": [
        "maintenance_id",
        "date",
        "fleet_id",
        "km_since_service",
        "days_to_service",
        "health_score",
        "fleet_status",
        "risk_score",
        "maintenance_priority",
        "inspection_result",
        "downtime_hours",
        "recurring_issue",
        "spare_part_id",
        "spare_part_name",
        "stock_qty",
        "min_stock",
        "reorder_point",
        "reorder_status",
        "days_to_kir_expiry",
        "kir_status",
    ],
    "S3_Keuangan": [
        "finance_id",
        "date",
        "route_id",
        "route_name",
        "sales_channel",
        "tickets_sold",
        "net_revenue",
        "settlement_amount",
        "settlement_diff",
        "settlement_status",
        "total_operational_cost",
        "net_profit",
        "profit_margin",
        "budget_category",
        "budget_amount",
        "actual_amount",
        "variance_amount",
        "budget_usage_pct",
        "budget_status",
    ],
}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(DATA_PATH)
    validate_input(data)

    metrics: list[Metric] = []
    outputs: dict[str, pd.DataFrame] = {}
    outputs.update(run_s1_operasional(data["S1_Operasional"], metrics))
    outputs.update(run_s2_perawatan_armada(data["S2_Perawatan_Armada"], metrics))
    outputs.update(run_s3_keuangan(data["S3_Keuangan"], metrics))

    write_outputs(outputs, metrics)
    print(f"Analytics workbook written to: {EXCEL_PATH}")
    print(f"CSV files written to: {CSV_DIR}")


def load_data(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    xls = pd.ExcelFile(path)
    missing = sorted(set(REQUIRED_SHEETS) - set(xls.sheet_names))
    if missing:
        raise ValueError(f"Missing required sheets: {missing}")

    data = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in REQUIRED_SHEETS}
    for df in data.values():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return data


def validate_input(data: dict[str, pd.DataFrame]) -> None:
    id_columns = {
        "S1_Operasional": "operational_id",
        "S2_Perawatan_Armada": "maintenance_id",
        "S3_Keuangan": "finance_id",
    }
    errors: list[str] = []
    for sheet, columns in REQUIRED_SHEETS.items():
        df = data[sheet]
        missing_columns = sorted(set(columns) - set(df.columns))
        if missing_columns:
            errors.append(f"{sheet} missing columns: {missing_columns}")
        if df["date"].isna().any():
            errors.append(f"{sheet} contains unparseable date values")
        id_col = id_columns[sheet]
        if id_col in df.columns and df[id_col].duplicated().any():
            errors.append(f"{sheet} contains duplicate {id_col}")
    if errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(errors))


def write_outputs(outputs: dict[str, pd.DataFrame], metrics: list[Metric]) -> None:
    readme = pd.DataFrame(
        [
            {"item": "project", "value": "Manual Analytics Python untuk Proyek BI PO Bus"},
            {"item": "source_dataset", "value": str(DATA_PATH)},
            {"item": "generated_at", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"item": "forecast_horizon_days", "value": FORECAST_HORIZON_DAYS},
            {"item": "sklearn_available", "value": SKLEARN_AVAILABLE},
            {"item": "statsmodels_available", "value": STATSMODELS_AVAILABLE},
        ]
        + [{"item": f"csv_{name}", "value": str(CSV_DIR / f"{name}.csv")} for name in outputs]
    )
    metrics_df = pd.DataFrame([metric.__dict__ for metric in metrics])

    for name, df in outputs.items():
        clean = clean_for_export(df)
        clean.to_csv(CSV_DIR / f"{name}.csv", index=False)
        outputs[name] = clean

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        for name, df in outputs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    format_workbook(EXCEL_PATH)


if __name__ == "__main__":
    main()
