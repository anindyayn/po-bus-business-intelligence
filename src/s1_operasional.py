from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import (
    DecisionTreeClassifier,
    ExponentialSmoothing,
    KMeans,
    Metric,
    SKLEARN_AVAILABLE,
    STATSMODELS_AVAILABLE,
    TfidfVectorizer,
    accuracy_score,
    mean_absolute_percentage_error,
    rolling_forecast,
    safe_ceil,
    silhouette_score,
    top_keywords,
    top_value,
    train_test_split,
)


FORECAST_HORIZON_DAYS = 7


def run_s1_operasional(df: pd.DataFrame, metrics: list[Metric]) -> dict[str, pd.DataFrame]:
    return {
        "I1_1_forecast_demand": forecast_demand(df, metrics),
        "I1_2_delay_risk": delay_risk(df, metrics),
        "I1_3_feedback_clusters": feedback_clusters(df, metrics),
    }


def forecast_demand(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    daily = (
        df.groupby(["route_id", "route_name", "departure_slot", "date"], as_index=False)
        .agg(
            tickets_sold=("tickets_sold", "sum"),
            occupancy_rate=("occupancy_rate", "mean"),
            current_frequency=("current_frequency", "mean"),
            recommended_frequency_actual=("recommended_frequency", "mean"),
        )
        .sort_values(["route_id", "departure_slot", "date"])
    )

    rows: list[dict[str, Any]] = []
    mapes: list[float] = []
    method_counts = {"holt_winters": 0, "rolling_average": 0}

    for (route_id, route_name, slot), group in daily.groupby(["route_id", "route_name", "departure_slot"]):
        series = group.set_index("date")["tickets_sold"].asfreq("D").interpolate(limit_direction="both")
        last_date = series.index.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS)

        method = "rolling_average"
        if STATSMODELS_AVAILABLE and len(series.dropna()) >= 28 and series.nunique() > 1:
            try:
                train = series.iloc[:-7]
                test = series.iloc[-7:]
                model = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit(optimized=True)
                test_pred = model.forecast(7)
                mapes.append(mean_absolute_percentage_error(test, test_pred))
                final_model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit(optimized=True)
                forecast_values = final_model.forecast(FORECAST_HORIZON_DAYS)
                method = "holt_winters"
            except Exception:
                forecast_values = rolling_forecast(series, FORECAST_HORIZON_DAYS)
        else:
            forecast_values = rolling_forecast(series, FORECAST_HORIZON_DAYS)

        method_counts[method] += 1
        avg_capacity = (
            df.loc[(df["route_id"] == route_id) & (df["departure_slot"] == slot), "capacity_seats"].mean()
            if "capacity_seats" in df.columns
            else np.nan
        )
        for date, predicted_demand in zip(future_dates, forecast_values, strict=False):
            recommended_frequency = safe_ceil(predicted_demand / avg_capacity) if avg_capacity and avg_capacity > 0 else np.nan
            rows.append(
                {
                    "forecast_date": date.date().isoformat(),
                    "route_id": route_id,
                    "route_name": route_name,
                    "departure_slot": slot,
                    "predicted_tickets_sold": round(float(max(predicted_demand, 0)), 2),
                    "avg_capacity_seats": round(float(avg_capacity), 2) if not pd.isna(avg_capacity) else np.nan,
                    "recommended_frequency_python": recommended_frequency,
                    "model_method": method,
                }
            )

    metrics.append(
        Metric(
            "I1.1",
            "Forecasting",
            "MAPE",
            round(float(np.nanmean(mapes)), 4) if mapes else "Not available",
            f"Holt-Winters groups: {method_counts['holt_winters']}; rolling average groups: {method_counts['rolling_average']}.",
        )
    )
    return pd.DataFrame(rows).sort_values(["forecast_date", "route_id", "departure_slot"])


def delay_risk(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    work["delay_risk_label"] = np.where(work["delay_minutes"] >= 30, "High Risk", "Normal")
    feature_cols = ["distance_km", "tickets_sold", "occupancy_rate", "demand_index", "current_frequency"]
    categorical_cols = ["route_name", "departure_slot", "bus_class", "season_period", "day_name"]

    if SKLEARN_AVAILABLE and work["delay_risk_label"].nunique() == 2:
        features = pd.get_dummies(work[feature_cols + categorical_cols], dummy_na=True)
        target = (work["delay_minutes"] >= 30).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25, random_state=42, stratify=target
        )
        model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=42)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        work["delay_risk_probability"] = model.predict_proba(features)[:, 1]
        method = "DecisionTreeClassifier"
        metric_value: Any = round(float(accuracy), 4)
    else:
        base = work["delay_minutes"].clip(lower=0)
        denominator = base.quantile(0.95) or base.max() or 1
        work["delay_risk_probability"] = (base / denominator).clip(upper=1)
        method = "Rule-based delay score"
        metric_value = "Not available"

    summary = (
        work.groupby(["route_id", "route_name", "departure_slot", "fleet_id"], as_index=False)
        .agg(
            trip_count=("operational_id", "count"),
            avg_delay_minutes=("delay_minutes", "mean"),
            max_delay_minutes=("delay_minutes", "max"),
            high_delay_count=("delay_risk_label", lambda s: int((s == "High Risk").sum())),
            avg_delay_risk_probability=("delay_risk_probability", "mean"),
            main_delay_reason=("delay_reason", top_value),
        )
        .sort_values(["avg_delay_risk_probability", "avg_delay_minutes"], ascending=False)
    )
    summary["risk_rank"] = np.arange(1, len(summary) + 1)
    summary["risk_category"] = pd.cut(
        summary["avg_delay_risk_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    metrics.append(Metric("I1.2", "Classification", "Accuracy", metric_value, f"Method: {method}. Target delay >= 30 minutes."))
    return summary


def feedback_clusters(df: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
    work = df.copy()
    work["feedback_text_clean"] = work["feedback_text_clean"].fillna("").astype(str)
    text_mask = work["feedback_text_clean"].str.strip().ne("")
    text_df = work.loc[text_mask].copy()

    if SKLEARN_AVAILABLE and len(text_df) >= 20 and text_df["feedback_text_clean"].nunique() >= 3:
        vectorizer = TfidfVectorizer(max_features=80, ngram_range=(1, 2), min_df=2)
        matrix = vectorizer.fit_transform(text_df["feedback_text_clean"])
        n_clusters = min(5, max(2, text_df["complaint_category"].nunique()))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        text_df["feedback_cluster"] = labels
        sil = silhouette_score(matrix, labels) if n_clusters > 1 and len(set(labels)) > 1 else np.nan
        terms = np.array(vectorizer.get_feature_names_out())
        keywords_by_cluster = {}
        for cluster_id in sorted(set(labels)):
            center = model.cluster_centers_[cluster_id]
            keywords_by_cluster[cluster_id] = ", ".join(terms[center.argsort()[-6:]][::-1])
        method = "TF-IDF + KMeans"
        metric_value: Any = round(float(sil), 4) if not pd.isna(sil) else "Not available"
    else:
        text_df["feedback_cluster"] = pd.factorize(text_df["complaint_category"])[0]
        keywords_by_cluster = {
            cluster: top_keywords(group["feedback_text_clean"])
            for cluster, group in text_df.groupby("feedback_cluster")
        }
        method = "Complaint category grouping"
        metric_value = "Not available"

    output = (
        text_df.groupby("feedback_cluster", as_index=False)
        .agg(
            feedback_count=("operational_id", "count"),
            avg_passenger_rating=("passenger_rating", "mean"),
            dominant_complaint_category=("complaint_category", top_value),
            dominant_route=("route_name", top_value),
            sample_feedback=("feedback_text_clean", lambda s: s.iloc[0]),
        )
        .sort_values("feedback_count", ascending=False)
    )
    output["cluster_keywords"] = output["feedback_cluster"].map(keywords_by_cluster)
    output["avg_passenger_rating"] = output["avg_passenger_rating"].round(2)
    metrics.append(Metric("I1.3", "Clustering", "Silhouette Score", metric_value, f"Method: {method}."))
    return output
