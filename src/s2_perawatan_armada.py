from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import (
    DecisionTreeClassifier,
    Metric,
    SKLEARN_AVAILABLE,
    accuracy_score,
    latest_by_date,
    normalize_0_100,
    top_value,
    train_test_split,
)


def run_s2_perawatan_armada(df: pd.DataFrame, metrics: list[Metric]) -> dict[str, pd.DataFrame]:
    return {
        "I2_1_fleet_health": fleet_health(df, metrics),
        "I2_2_service_priority": service_priority(df, metrics),
        "I2_3_sparepart_kir_alerts": sparepart_kir_alerts(df, metrics),
    }


def fleet_health(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    score_cols = ["engine_score", "brake_score", "tire_score", "ac_score", "electrical_score", "health_score", "risk_score"]
    if SKLEARN_AVAILABLE and work["fleet_status"].nunique() > 1:
        features = work[score_cols + ["km_since_service", "days_to_service", "downtime_hours"]].fillna(0)
        target = work["fleet_status"].astype(str)
        stratify = target if target.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25, random_state=42, stratify=stratify
        )
        model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=42)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        work["predicted_fleet_status"] = model.predict(features)
        method = "DecisionTreeClassifier"
        metric_value: Any = round(float(accuracy), 4)
    else:
        work["predicted_fleet_status"] = np.select(
            [
                (work["health_score"] < 65) | (work["risk_score"] >= 70) | (work["inspection_result"].astype(str).str.lower() != "lulus"),
                (work["health_score"] < 78) | (work["risk_score"] >= 40) | (work["days_to_service"] <= 7),
            ],
            ["Tidak Layak Jalan", "Butuh Servis"],
            default="Siap Jalan",
        )
        method = "Rule-based health classification"
        metric_value = "Not available"

    latest = latest_by_date(work, "fleet_id")
    output = latest[
        [
            "date",
            "fleet_id",
            "fleet_type",
            "bus_class",
            "health_score",
            "risk_score",
            "engine_score",
            "brake_score",
            "tire_score",
            "inspection_result",
            "fleet_status",
            "predicted_fleet_status",
            "maintenance_priority",
        ]
    ].sort_values(["risk_score", "health_score"], ascending=[False, True])
    metrics.append(Metric("I2.1", "Classification", "Accuracy", metric_value, f"Method: {method}. Latest row per fleet exported."))
    return output


def service_priority(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    latest = latest_by_date(df.copy(), "fleet_id")
    work = latest.copy()
    work["recurring_issue_flag"] = work["recurring_issue"].astype(str).str.lower().eq("ya").astype(int)
    work["days_to_service_pressure"] = (30 - work["days_to_service"]).clip(lower=0)
    components = {
        "km_since_service": 0.25,
        "days_to_service_pressure": 0.20,
        "health_gap": 0.25,
        "downtime_hours": 0.15,
        "recurring_issue_flag": 0.15,
    }
    work["health_gap"] = (100 - work["health_score"]).clip(lower=0)
    total = np.zeros(len(work), dtype=float)
    for col, weight in components.items():
        total += normalize_0_100(work[col]) * weight
    work["service_priority_score"] = np.round(total, 2)
    work["service_priority_python"] = pd.cut(
        work["service_priority_score"],
        bins=[-0.01, 35, 65, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    output = work[
        [
            "date",
            "fleet_id",
            "fleet_type",
            "bus_class",
            "km_since_service",
            "days_to_service",
            "health_score",
            "risk_score",
            "downtime_hours",
            "recurring_issue",
            "maintenance_priority",
            "service_priority_score",
            "service_priority_python",
            "action_recommendation",
        ]
    ].sort_values("service_priority_score", ascending=False)
    metrics.append(
        Metric(
            "I2.2",
            "Risk Scoring",
            "High Priority Fleet Count",
            int((output["service_priority_python"] == "High").sum()),
            "Weighted score from mileage, service due date, health gap, downtime, and recurring issues.",
        )
    )
    return output


def sparepart_kir_alerts(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    part_summary = (
        work.groupby(["spare_part_id", "spare_part_name", "spare_part_category"], as_index=False)
        .agg(
            latest_date=("date", "max"),
            avg_stock_qty=("stock_qty", "mean"),
            min_stock_qty=("stock_qty", "min"),
            avg_min_stock=("min_stock", "mean"),
            avg_reorder_point=("reorder_point", "mean"),
            critical_records=("reorder_status", lambda s: int(s.astype(str).str.lower().eq("critical").sum())),
            reorder_status=("reorder_status", top_value),
        )
    )
    part_summary["part_alert_level"] = np.select(
        [
            part_summary["min_stock_qty"] <= part_summary["avg_min_stock"],
            part_summary["avg_stock_qty"] <= part_summary["avg_reorder_point"],
        ],
        ["Critical", "Reorder"],
        default="OK",
    )

    latest_fleet = latest_by_date(work, "fleet_id")
    kir_alerts = latest_fleet[
        [
            "fleet_id",
            "kir_expiry_date",
            "days_to_kir_expiry",
            "kir_status",
            "inspection_result",
            "action_recommendation",
        ]
    ].copy()
    kir_alerts["kir_alert_level"] = np.select(
        [
            kir_alerts["kir_status"].astype(str).str.lower().ne("aktif") | (kir_alerts["days_to_kir_expiry"] <= 0),
            kir_alerts["days_to_kir_expiry"] <= 30,
        ],
        ["Expired/Invalid", "Due <= 30 Days"],
        default="OK",
    )

    part_alerts = part_summary.loc[part_summary["part_alert_level"] != "OK"].copy()
    part_alerts["alert_type"] = "Spare Part"
    part_alerts["alert_level"] = part_alerts["part_alert_level"]
    part_alerts["entity_id"] = part_alerts["spare_part_id"]
    part_alerts["entity_name"] = part_alerts["spare_part_name"]
    part_alerts["alert_detail"] = (
        "Stock min "
        + part_alerts["min_stock_qty"].round(0).astype(int).astype(str)
        + "; avg stock "
        + part_alerts["avg_stock_qty"].round(1).astype(str)
        + "; reorder point "
        + part_alerts["avg_reorder_point"].round(1).astype(str)
    )

    kir_only = kir_alerts.loc[kir_alerts["kir_alert_level"] != "OK"].copy()
    kir_only["alert_type"] = "KIR"
    kir_only["alert_level"] = kir_only["kir_alert_level"]
    kir_only["entity_id"] = kir_only["fleet_id"]
    kir_only["entity_name"] = kir_only["fleet_id"]
    kir_only["alert_detail"] = (
        "KIR status "
        + kir_only["kir_status"].astype(str)
        + "; expires in "
        + kir_only["days_to_kir_expiry"].astype(str)
        + " days"
    )

    output = pd.concat(
        [
            part_alerts[
                [
                    "alert_type",
                    "alert_level",
                    "entity_id",
                    "entity_name",
                    "alert_detail",
                    "spare_part_category",
                    "avg_stock_qty",
                    "min_stock_qty",
                    "avg_min_stock",
                    "avg_reorder_point",
                    "critical_records",
                    "reorder_status",
                ]
            ],
            kir_only[
                [
                    "alert_type",
                    "alert_level",
                    "entity_id",
                    "entity_name",
                    "alert_detail",
                    "kir_expiry_date",
                    "days_to_kir_expiry",
                    "kir_status",
                    "inspection_result",
                    "action_recommendation",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    if "kir_expiry_date" in output.columns:
        output["kir_expiry_date"] = pd.to_datetime(output["kir_expiry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    metrics.append(
        Metric(
            "I2.3",
            "Association / Rule-Based Alert",
            "Critical Part Count",
            int((part_summary["part_alert_level"] == "Critical").sum()),
            "Rule-based spare part reorder and KIR expiry alerts.",
        )
    )
    return output.sort_values(["alert_type", "alert_level", "critical_records"], ascending=[False, True, False])
