from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import IsolationForest, Metric, SKLEARN_AVAILABLE, StandardScaler, z_score


def run_s3_keuangan(df: pd.DataFrame, metrics: list[Metric]) -> dict[str, pd.DataFrame]:
    return {
        "I3_1_settlement_recon": settlement_recon(df, metrics),
        "I3_2_route_profitability": route_profitability(df, metrics),
        "I3_3_budget_anomalies": budget_anomalies(df, metrics),
    }


def settlement_recon(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    work["abs_settlement_diff"] = work["settlement_diff"].abs()
    output = (
        work.groupby(["sales_channel", "settlement_status"], as_index=False)
        .agg(
            transaction_count=("finance_id", "count"),
            tickets_sold=("tickets_sold", "sum"),
            net_revenue=("net_revenue", "sum"),
            settlement_amount=("settlement_amount", "sum"),
            total_settlement_diff=("settlement_diff", "sum"),
            avg_abs_settlement_diff=("abs_settlement_diff", "mean"),
            max_abs_settlement_diff=("abs_settlement_diff", "max"),
        )
        .sort_values("avg_abs_settlement_diff", ascending=False)
    )
    output["reconciliation_priority"] = pd.cut(
        output["avg_abs_settlement_diff"],
        bins=[-0.01, 10_000, 50_000, math.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    metrics.append(
        Metric(
            "I3.1",
            "Classification / Rule-Based Reconciliation",
            "Non-Matched Transaction Count",
            int((work["settlement_status"].astype(str).str.lower() != "cocok").sum()),
            "Settlement grouped by sales channel and status.",
        )
    )
    return output


def route_profitability(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    output = (
        df.groupby(["route_id", "route_name"], as_index=False)
        .agg(
            trip_count=("finance_id", "count"),
            tickets_sold=("tickets_sold", "sum"),
            net_revenue=("net_revenue", "sum"),
            total_operational_cost=("total_operational_cost", "sum"),
            net_profit=("net_profit", "sum"),
            avg_profit_margin=("profit_margin", "mean"),
            avg_break_even_load_factor=("break_even_load_factor", "mean"),
        )
        .sort_values("net_profit", ascending=False)
    )
    output["profitability_status"] = np.select(
        [
            output["avg_profit_margin"] < 0,
            output["avg_profit_margin"] < 0.1,
        ],
        ["Rugi", "Margin Tipis"],
        default="Untung",
    )
    output["profit_rank"] = output["net_profit"].rank(ascending=False, method="dense").astype(int)
    metrics.append(
        Metric(
            "I3.2",
            "Estimation",
            "Loss Route Count",
            int((output["profitability_status"] == "Rugi").sum()),
            "Aggregated net revenue, cost, profit, and margin by route.",
        )
    )
    return output


def budget_anomalies(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    feature_cols = ["budget_amount", "actual_amount", "variance_amount", "budget_usage_pct", "total_operational_cost"]
    method = "z-score fallback"
    if SKLEARN_AVAILABLE and len(work) >= 50:
        features = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        scaled = StandardScaler().fit_transform(features)
        model = IsolationForest(contamination=0.08, random_state=42)
        work["anomaly_flag"] = np.where(model.fit_predict(scaled) == -1, "Anomaly", "Normal")
        work["anomaly_score"] = -model.score_samples(scaled)
        method = "IsolationForest"
    else:
        usage_z = z_score(work["budget_usage_pct"])
        variance_z = z_score(work["variance_amount"].abs())
        work["anomaly_score"] = (usage_z.abs() + variance_z.abs()) / 2
        work["anomaly_flag"] = np.where((work["budget_usage_pct"] > 1.0) | (work["anomaly_score"] >= 2), "Anomaly", "Normal")

    output = (
        work.groupby(["budget_category", "budget_status", "anomaly_flag"], as_index=False)
        .agg(
            record_count=("finance_id", "count"),
            budget_amount=("budget_amount", "sum"),
            actual_amount=("actual_amount", "sum"),
            variance_amount=("variance_amount", "sum"),
            avg_budget_usage_pct=("budget_usage_pct", "mean"),
            avg_anomaly_score=("anomaly_score", "mean"),
        )
        .sort_values(["anomaly_flag", "avg_budget_usage_pct"], ascending=[True, False])
    )
    output["overbudget_amount"] = (output["actual_amount"] - output["budget_amount"]).clip(lower=0)
    metrics.append(
        Metric(
            "I3.3",
            "Estimation / Classification",
            "Anomaly Group Count",
            int((output["anomaly_flag"] == "Anomaly").sum()),
            f"Method: {method}.",
        )
    )
    return output
