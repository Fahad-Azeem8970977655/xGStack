"""
xG / PSxG Analytics Suite — Advanced Streamlit Dashboard (v2)
----------------------------------------------------------------
Run with:  streamlit run app.py

Feed it data one of two ways:
  1. Unzip your results zip into a `results/` folder next to this app.py, and
     put your shots CSV (xg_psxg_shots.csv or shots_full_features.csv) next
     to it too. Just run `streamlit run app.py`.
  2. Or run the app with nothing in place and use the sidebar's
     "📦 Load / refresh data" uploaders — the results zip is extracted
     in-memory, no unzipping needed.

Expected files (all optional — every page degrades gracefully if a file is
missing and tells you what it needs):
  results/model_comparison_results.csv
  results/xg_test_predictions.csv
  results/*.png
  results/best_xg_model__*.joblib
  results/best_psxg_model__*.joblib
  xg_psxg_shots.csv  OR  shots_full_features.csv
"""

import io
import os
import glob
import zipfile
import tempfile
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="xG / PSxG Analytics Suite",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_ORDER = ["distance", "angle_deg", "is_header", "is_left_foot", "is_right_foot",
                 "is_open_play", "is_set_piece", "under_pressure"]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stMetric { background-color: rgba(120, 120, 120, 0.08); border-radius: 12px;
                padding: 10px 6px; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .block-container { padding-top: 1.4rem; }
    h1, h2, h3 { letter-spacing: -0.3px; }
    .badge { display:inline-block; padding:2px 10px; border-radius: 999px;
             background: rgba(46,139,87,0.15); font-size: 0.8rem; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data / results loading
# ---------------------------------------------------------------------------

def resolve_results_dir():
    if "results_dir" in st.session_state:
        return st.session_state["results_dir"]
    if os.path.isdir("results"):
        return "results"
    return None


@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


@st.cache_resource(show_spinner=False)
def load_joblib_model(path):
    return joblib.load(path)


def find_first(results_dir, patterns):
    if not results_dir:
        return None
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(results_dir, pattern)))
        if matches:
            return matches[0]
    return None


def template():
    return st.session_state.get("chart_theme", "plotly")


def style_fig(fig, height=None):
    fig.update_layout(template=template())
    if height:
        fig.update_layout(height=height)
    return fig


def pitch_lines_traces(color="grey"):
    lines = [
        ([60, 60, 120, 120, 60], [0, 80, 80, 0, 0]),
        ([102, 102, 120, 120, 102], [18, 62, 62, 18, 18]),
        ([114, 114, 120, 120, 114], [30, 50, 50, 30, 30]),
    ]
    return [go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color), showlegend=False) for xs, ys in lines]


# ---------------------------------------------------------------------------
# Sidebar — navigation + data sources + appearance
# ---------------------------------------------------------------------------

st.sidebar.markdown("## ⚽ xG / PSxG Suite")
st.sidebar.caption("Advanced analytics dashboard")

with st.sidebar.expander("📦 Load / refresh data", expanded=(resolve_results_dir() is None)):
    zip_upload = st.file_uploader("Upload results zip", type="zip", key="zip_uploader")
    if zip_upload is not None:
        tmp_dir = tempfile.mkdtemp(prefix="xg_results_")
        with zipfile.ZipFile(io.BytesIO(zip_upload.read())) as zf:
            zf.extractall(tmp_dir)
        st.session_state["results_dir"] = tmp_dir
        st.success(f"Extracted {len(os.listdir(tmp_dir))} files")

    shots_upload = st.file_uploader("Upload shots CSV (optional)", type="csv", key="shots_uploader")

with st.sidebar.expander("🎨 Appearance"):
    st.selectbox(
        "Chart theme", ["plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation"],
        key="chart_theme",
    )

results_dir = resolve_results_dir()

shots_df = None
if shots_upload is not None:
    shots_df = load_csv(shots_upload)
else:
    for candidate in ["xg_psxg_shots.csv", "shots_full_features.csv"]:
        if os.path.exists(candidate):
            shots_df = load_csv(candidate)
            break

results_csv_path = find_first(results_dir, ["model_comparison_results.csv"])
preds_csv_path = find_first(results_dir, ["xg_test_predictions.csv"])
results_df = load_csv(results_csv_path) if results_csv_path else None
preds_df = load_csv(preds_csv_path) if preds_csv_path else None
best_xg_model_path = find_first(results_dir, ["best_xg_model__*.joblib"])

st.sidebar.divider()

PAGES = [
    "🏠 Home", "🗒️ Auto Insights", "📊 Model Leaderboard", "🔬 Model Deep-Dive", "🥊 Model Battle",
    "🧬 Ensemble Builder", "🧠 Explainability", "🛠️ AutoML Lab", "📥 Batch Predictor",
    "🗺️ Shot Explorer", "🏟️ Match Center", "🔮 Live Predictor",
    "🖼️ Chart Gallery", "📁 Raw Data",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Data status")
st.sidebar.write(f"{'✅' if results_df is not None else '⚠️'} Model results")
st.sidebar.write(f"{'✅' if preds_df is not None else '⚠️'} Test predictions")
st.sidebar.write(f"{'✅' if shots_df is not None else '⚠️'} Shots dataset")
st.sidebar.write(f"{'✅' if best_xg_model_path else '⚠️'} Trained model file")

# ===========================================================================
# PAGE: HOME
# ===========================================================================

if page == "🏠 Home":
    st.title("⚽ xG / PSxG Analytics Suite")
    st.caption("Expected Goals & Post-Shot Expected Goals — model comparison and shot analysis")

    if results_df is None and preds_df is None and shots_df is None:
        st.info(
            "👋 Nothing loaded yet. Open **📦 Load / refresh data** in the sidebar and upload your "
            "`xg_project_results.zip` (and optionally your shots CSV) to get started."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🧠 Models compared", int(results_df["model"].nunique()) if results_df is not None else "—")
        with c2:
            if results_df is not None:
                best_row = results_df.loc[results_df["roc_auc"].idxmax()]
                st.metric("🏆 Best ROC AUC", f"{best_row['roc_auc']:.3f}", help=best_row["model"])
            else:
                st.metric("🏆 Best ROC AUC", "—")
        with c3:
            st.metric("🎯 Shots in dataset", f"{len(shots_df):,}" if shots_df is not None else "—")
        with c4:
            goals = int(shots_df["goal"].sum()) if shots_df is not None and "goal" in shots_df.columns else None
            st.metric("⚡ Goals", f"{goals:,}" if goals is not None else "—")

        st.divider()

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.subheader("📈 What's inside")
            st.markdown("""
<span class="badge">📊 Leaderboard</span>
<span class="badge">🔬 Deep-Dive</span>
<span class="badge">🥊 Model Battle</span>
<span class="badge">🧠 Explainability</span>
<span class="badge">🗺️ Shot Explorer</span>
<span class="badge">🏟️ Match Center</span>
<span class="badge">🔮 Live Predictor</span>
<span class="badge">🖼️ Gallery</span>

- **📊 Model Leaderboard** — every model side by side on ROC AUC, Brier score, and log loss
- **🔬 Model Deep-Dive** — interactive ROC/calibration/PR curves + a live confusion-matrix threshold slider
- **🥊 Model Battle** — pick any two models and see exactly where they disagree
- **🧠 Explainability** — live permutation feature importance + partial-dependence style curves
- **🏟️ Match Center** — pick a match and watch cumulative xG unfold minute by minute
- **🗺️ Shot Explorer** — filterable shot map and per-player over/under-performance vs xG
- **🔮 Live Predictor** — set up a shot and get an instant probability from your best model
- **🖼️ Chart Gallery** — every PNG your notebook exported, in one place
            """, unsafe_allow_html=True)
        with col_b:
            if results_df is not None:
                st.subheader("🏅 Top 5 models")
                top5 = results_df.sort_values("roc_auc", ascending=False).head(5)[["model", "roc_auc", "brier"]]
                st.dataframe(
                    top5.style.format({"roc_auc": "{:.3f}", "brier": "{:.3f}"}).background_gradient(
                        subset=["roc_auc"], cmap="Greens"
                    ),
                    width='stretch', hide_index=True,
                )

        st.divider()
        st.subheader("📄 Export a summary report")
        if st.button("Generate HTML report"):
            best = results_df.loc[results_df["roc_auc"].idxmax()] if results_df is not None else None
            report_html = f"""
            <html><head><meta charset='utf-8'><title>xG Report</title>
            <style>body{{font-family:Arial, sans-serif; margin:40px;}}
            h1{{color:#b91c1c;}} table{{border-collapse:collapse;width:100%;}}
            td,th{{border:1px solid #ddd;padding:8px;}} th{{background:#f4f4f4;text-align:left;}}</style>
            </head><body>
            <h1>⚽ xG / PSxG Summary Report</h1>
            <p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <h2>Headline numbers</h2>
            <ul>
                <li>Shots analyzed: {len(shots_df) if shots_df is not None else 'n/a'}</li>
                <li>Goals: {int(shots_df['goal'].sum()) if shots_df is not None and 'goal' in shots_df.columns else 'n/a'}</li>
                <li>Best model: {best['model'] if best is not None else 'n/a'}
                    (ROC AUC {best['roc_auc']:.3f}, Brier {best['brier']:.3f})</li>
            </ul>
            <h2>Full model comparison</h2>
            {results_df.to_html(index=False) if results_df is not None else '<p>No results loaded.</p>'}
            </body></html>
            """
            st.download_button("⬇️ Download report.html", report_html, file_name="xg_report.html", mime="text/html")

# ===========================================================================
# PAGE: AUTO INSIGHTS
# ===========================================================================

elif page == "🗒️ Auto Insights":
    st.title("🗒️ Auto Insights")
    st.caption("Automatically mined, plain-English findings from whatever data is currently loaded.")

    insights = []

    if results_df is not None:
        xg_rows = results_df[results_df["task"] == "xG"]
        if len(xg_rows) > 1:
            best_row = xg_rows.loc[xg_rows["roc_auc"].idxmax()]
            baseline_row = xg_rows[xg_rows["model"] == "Logistic Regression"]
            if len(baseline_row):
                uplift = best_row["roc_auc"] - baseline_row["roc_auc"].iloc[0]
                insights.append(
                    f"🏆 **{best_row['model']}** is the strongest xG model (ROC AUC {best_row['roc_auc']:.3f}), "
                    f"beating plain Logistic Regression by **{uplift:+.3f}** ROC AUC."
                )
            best_calibrated = xg_rows.loc[xg_rows["brier"].idxmin()]
            insights.append(
                f"📉 **{best_calibrated['model']}** has the best-calibrated probabilities "
                f"(lowest Brier score, {best_calibrated['brier']:.3f}) — its xG numbers are the most "
                "trustworthy at face value, even if another model ranks shots better."
            )
        psxg_rows = results_df[results_df["task"] == "PSxG"]
        if len(psxg_rows):
            xg_best = xg_rows["roc_auc"].max() if len(xg_rows) else None
            psxg_best = psxg_rows["roc_auc"].max()
            if xg_best is not None:
                gain = psxg_best - xg_best
                insights.append(
                    f"🎯 Adding shot-placement features lifts the best ROC AUC from {xg_best:.3f} (xG) to "
                    f"{psxg_best:.3f} (PSxG) — a **{gain:+.3f}** gain from knowing where the shot actually went."
                )

    if shots_df is not None and "goal" in shots_df.columns:
        conv_rate = shots_df["goal"].mean()
        insights.append(f"⚡ Overall conversion rate in this dataset: **{conv_rate:.1%}** across {len(shots_df):,} shots.")

        if "shot_body_part" in shots_df.columns:
            bp_conv = shots_df.groupby("shot_body_part")["goal"].agg(["mean", "count"])
            bp_conv = bp_conv[bp_conv["count"] >= 20]
            if len(bp_conv):
                best_bp = bp_conv["mean"].idxmax()
                worst_bp = bp_conv["mean"].idxmin()
                insights.append(
                    f"👟 **{best_bp}** shots convert best ({bp_conv.loc[best_bp, 'mean']:.1%}), while "
                    f"**{worst_bp}** shots convert worst ({bp_conv.loc[worst_bp, 'mean']:.1%}) among body parts "
                    "with at least 20 shots."
                )

        xg_col_options = [c for c in ["statsbomb_xg", "my_xg", "my_psxg"] if c in shots_df.columns]
        if xg_col_options and "player" in shots_df.columns:
            xg_col = xg_col_options[0]
            player_agg = shots_df.groupby("player").agg(shots=("player", "size"), goals=("goal", "sum"),
                                                          xg=(xg_col, "sum"))
            player_agg = player_agg[player_agg["shots"] >= 5]
            player_agg["diff"] = player_agg["goals"] - player_agg["xg"]
            if len(player_agg):
                over = player_agg["diff"].idxmax()
                under = player_agg["diff"].idxmin()
                insights.append(
                    f"🔥 **{over}** is the biggest overperformer vs {xg_col} "
                    f"({player_agg.loc[over, 'diff']:+.2f} goals above expected)."
                )
                insights.append(
                    f"🥶 **{under}** is the biggest underperformer vs {xg_col} "
                    f"({player_agg.loc[under, 'diff']:+.2f} goals vs expected)."
                )

        if "minute" in shots_df.columns:
            first_half = shots_df[shots_df["minute"] <= 45]
            second_half = shots_df[shots_df["minute"] > 45]
            if len(first_half) and len(second_half):
                fh_conv, sh_conv = first_half["goal"].mean(), second_half["goal"].mean()
                phase = "second" if sh_conv > fh_conv else "first"
                insights.append(
                    f"⏱️ Shots convert more often in the **{phase} half** "
                    f"({max(fh_conv, sh_conv):.1%} vs {min(fh_conv, sh_conv):.1%})."
                )

        if {"x", "y", "goal"}.issubset(shots_df.columns):
            binned = shots_df.copy()
            binned["x_bin"] = pd.cut(binned["x"], bins=6)
            binned["y_bin"] = pd.cut(binned["y"], bins=4)
            zone_stats = binned.groupby(["x_bin", "y_bin"], observed=True)["goal"].agg(["mean", "count"])
            zone_stats = zone_stats[zone_stats["count"] >= 15]
            if len(zone_stats):
                best_zone = zone_stats["mean"].idxmax()
                insights.append(
                    f"🌡️ The most dangerous pitch zone (≥15 shots) converts at "
                    f"**{zone_stats.loc[best_zone, 'mean']:.1%}** — roughly x∈{best_zone[0]}, y∈{best_zone[1]}."
                )

    if preds_df is not None:
        pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
        if len(pred_cols) >= 2:
            corr_raw = preds_df[pred_cols].corr()
            corr_arr = corr_raw.to_numpy(copy=True)
            np.fill_diagonal(corr_arr, np.nan)
            corr = pd.DataFrame(corr_arr, index=corr_raw.index, columns=corr_raw.columns)
            most_similar = corr.stack().idxmax()
            insights.append(
                f"🤝 **{most_similar[0].replace('pred_', '')}** and **{most_similar[1].replace('pred_', '')}** "
                f"make the most similar predictions of any pair (correlation {corr.stack().max():.3f})."
            )

    if not insights:
        st.info("👋 Nothing to analyze yet — load your results zip and/or shots CSV from the sidebar.")
    else:
        for line in insights:
            st.markdown(f"- {line}")
        st.caption(f"Generated from {len(insights)} automated checks — refresh the page after loading new data to re-run them.")

# ===========================================================================
# PAGE: MODEL LEADERBOARD
# ===========================================================================

elif page == "📊 Model Leaderboard":
    st.title("📊 Model Leaderboard")

    if results_df is None:
        st.warning("⚠️ No `model_comparison_results.csv` found. Upload your results zip from the sidebar.")
    else:
        tasks = sorted(results_df["task"].unique())
        tab_objs = st.tabs([f"🧪 {t}" for t in tasks] + ["⚖️ Compare tasks"])

        for tab, task in zip(tab_objs[:-1], tasks):
            with tab:
                task_df = results_df[results_df["task"] == task].sort_values("roc_auc", ascending=False)

                best = task_df.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("🏆 Best model", best["model"])
                m2.metric("🎯 ROC AUC", f"{best['roc_auc']:.3f}")
                m3.metric("📉 Brier score", f"{best['brier']:.3f}")

                left, right = st.columns([2, 3])
                with left:
                    st.dataframe(
                        task_df[["model", "roc_auc", "brier", "log_loss"]]
                        .style.format({"roc_auc": "{:.3f}", "brier": "{:.3f}", "log_loss": "{:.3f}"})
                        .background_gradient(subset=["roc_auc"], cmap="Greens")
                        .background_gradient(subset=["brier", "log_loss"], cmap="Reds_r"),
                        width='stretch', hide_index=True, height=420,
                    )
                with right:
                    metric_choice = st.radio(
                        "Metric to chart", ["roc_auc", "brier", "log_loss"],
                        horizontal=True, key=f"metric_{task}",
                        format_func=lambda m: {"roc_auc": "🎯 ROC AUC (higher better)",
                                                "brier": "📉 Brier (lower better)",
                                                "log_loss": "📉 Log loss (lower better)"}[m],
                    )
                    ascending = metric_choice != "roc_auc"
                    chart_df = task_df.sort_values(metric_choice, ascending=ascending)
                    fig = px.bar(
                        chart_df, x=metric_choice, y="model", orientation="h",
                        color=metric_choice, color_continuous_scale="Viridis",
                        title=f"{task} models ranked by {metric_choice}",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending" if not ascending else "total descending"},
                                       showlegend=False)
                    st.plotly_chart(style_fig(fig, 440), width='stretch')

        with tab_objs[-1]:
            if len(tasks) > 1:
                fig = px.bar(
                    results_df, x="roc_auc", y="model", color="task", orientation="h",
                    barmode="group", title="ROC AUC — xG vs PSxG models side by side",
                )
                st.plotly_chart(style_fig(fig, 500), width='stretch')
            else:
                st.info("Only one task (xG) is present in this results file — nothing to compare yet. "
                        "Re-run the notebook with placement features to also get PSxG results.")

# ===========================================================================
# PAGE: MODEL DEEP-DIVE
# ===========================================================================

elif page == "🔬 Model Deep-Dive":
    st.title("🔬 Model Deep-Dive")

    if preds_df is None:
        st.warning("⚠️ No `xg_test_predictions.csv` found. Upload your results zip from the sidebar.")
    else:
        pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
        model_names = [c.replace("pred_", "") for c in pred_cols]
        y_true = preds_df["actual_goal"]

        tab_roc, tab_calib, tab_cm, tab_pr, tab_boot = st.tabs(
            ["📈 ROC Curves", "🎚️ Calibration", "🧮 Confusion Matrix", "🔁 Precision / Recall", "🎲 Bootstrap CI"]
        )

        with tab_roc:
            selected = st.multiselect("Models to plot", model_names, default=model_names[:5])
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
            for name in selected:
                proba = preds_df[f"pred_{name}"]
                fpr, tpr, _ = roc_curve(y_true, proba)
                score = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={score:.3f})"))
            fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(style_fig(fig, 520), width='stretch')

        with tab_calib:
            selected_c = st.multiselect("Models to plot", model_names, default=model_names[:5], key="calib_models")
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
            for name in selected_c:
                proba = preds_df[f"pred_{name}"]
                frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
                fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=name))
            fig.update_layout(title="Calibration curves (perfectly calibrated = diagonal)",
                               xaxis_title="Mean predicted probability", yaxis_title="Fraction of actual goals")
            st.plotly_chart(style_fig(fig, 520), width='stretch')
            st.caption("A model above the diagonal under-predicts risk in that bucket; below it over-predicts.")

        with tab_cm:
            model_for_cm = st.selectbox("Model", model_names, key="cm_model")
            proba = preds_df[f"pred_{model_for_cm}"]

            precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
            best_thresh_idx = np.nanargmax(f1s[:-1]) if len(thresholds) else 0
            best_threshold = float(thresholds[best_thresh_idx]) if len(thresholds) else 0.5

            auto_col, slider_col = st.columns([1, 3])
            with auto_col:
                use_optimal = st.toggle("🎯 Use best-F1 threshold", value=False)
            with slider_col:
                threshold = st.slider("Decision threshold", 0.05, 0.95,
                                       round(best_threshold, 2) if use_optimal else 0.5, 0.01,
                                       disabled=use_optimal)
            if use_optimal:
                threshold = round(best_threshold, 2)
                st.caption(f"Auto-selected threshold {threshold:.2f} maximizes F1 on the test set.")

            y_pred = (proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            fig = px.imshow(
                cm, text_auto=True, x=["Pred: No Goal", "Pred: Goal"], y=["Actual: No Goal", "Actual: Goal"],
                color_continuous_scale="Blues",
                title=f"Confusion matrix — {model_for_cm} @ threshold {threshold:.2f}",
            )
            st.plotly_chart(style_fig(fig), width='stretch')

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = f1_score(y_true, y_pred)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ Precision", f"{precision:.1%}")
            c2.metric("🔎 Recall", f"{recall:.1%}")
            c3.metric("⚖️ F1", f"{f1:.3f}")
            c4.metric("📉 Brier score", f"{brier_score_loss(y_true, proba):.4f}")

        with tab_pr:
            selected_pr = st.multiselect("Models to plot", model_names, default=model_names[:5], key="pr_models")
            fig = go.Figure()
            for name in selected_pr:
                proba = preds_df[f"pred_{name}"]
                precision, recall, _ = precision_recall_curve(y_true, proba)
                fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
            fig.update_layout(title="Precision–Recall curves", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(style_fig(fig, 520), width='stretch')

        with tab_boot:
            st.caption("Bootstrap resampling of the test set to see how stable ROC AUC really is (not just a single number).")
            model_boot = st.selectbox("Model", model_names, key="boot_model")
            n_boot = st.slider("Bootstrap samples", 100, 1000, 300, 50)
            if st.button("🎲 Run bootstrap"):
                proba = preds_df[f"pred_{model_boot}"].values
                y_arr = y_true.values
                rng = np.random.default_rng(42)
                scores = []
                with st.spinner("Resampling..."):
                    for _ in range(n_boot):
                        idx = rng.integers(0, len(y_arr), len(y_arr))
                        if len(np.unique(y_arr[idx])) < 2:
                            continue
                        fpr, tpr, _ = roc_curve(y_arr[idx], proba[idx])
                        scores.append(auc(fpr, tpr))
                scores = np.array(scores)
                lo, hi = np.percentile(scores, [2.5, 97.5])

                c1, c2, c3 = st.columns(3)
                c1.metric("Mean ROC AUC", f"{scores.mean():.3f}")
                c2.metric("95% CI low", f"{lo:.3f}")
                c3.metric("95% CI high", f"{hi:.3f}")

                fig = px.histogram(scores, nbins=40, title=f"Bootstrap distribution of ROC AUC — {model_boot}")
                fig.add_vline(x=lo, line_dash="dash", line_color="red")
                fig.add_vline(x=hi, line_dash="dash", line_color="red")
                fig.update_layout(xaxis_title="ROC AUC", showlegend=False)
                st.plotly_chart(style_fig(fig, 420), width='stretch')

# ===========================================================================
# PAGE: MODEL BATTLE
# ===========================================================================

elif page == "🥊 Model Battle":
    st.title("🥊 Model Battle")
    st.caption("Pick two models and see exactly where their predictions agree and disagree.")

    if preds_df is None:
        st.warning("⚠️ No `xg_test_predictions.csv` found. Upload your results zip from the sidebar.")
    else:
        pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
        model_names = [c.replace("pred_", "") for c in pred_cols]

        c1, c2 = st.columns(2)
        with c1:
            model_a = st.selectbox("🔵 Model A", model_names, index=0)
        with c2:
            default_b = 1 if len(model_names) > 1 else 0
            model_b = st.selectbox("🔴 Model B", model_names, index=default_b)

        proba_a = preds_df[f"pred_{model_a}"]
        proba_b = preds_df[f"pred_{model_b}"]
        y_true = preds_df["actual_goal"]

        agree_rate = ((proba_a >= 0.5) == (proba_b >= 0.5)).mean()
        avg_gap = (proba_a - proba_b).abs().mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("🤝 Classification agreement (@0.5)", f"{agree_rate:.1%}")
        m2.metric("↔️ Avg. probability gap", f"{avg_gap:.3f}")
        m3.metric("🎯 Shots compared", f"{len(preds_df):,}")

        left, right = st.columns([3, 2])
        with left:
            plot_df = pd.DataFrame({
                "A": proba_a, "B": proba_b,
                "Outcome": np.where(y_true == 1, "Goal ⚡", "No goal"),
            })
            fig = px.scatter(
                plot_df, x="A", y="B", color="Outcome",
                color_discrete_map={"Goal ⚡": "crimson", "No goal": "lightslategrey"},
                labels={"A": model_a, "B": model_b},
                title=f"{model_a} vs {model_b} — predicted probability per shot",
                opacity=0.6,
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
            st.plotly_chart(style_fig(fig, 480), width='stretch')

        with right:
            st.subheader("🔥 Biggest disagreements")
            disagree_df = preds_df.copy()
            disagree_df["gap"] = (proba_a - proba_b).abs()
            disagree_df["actual"] = np.where(y_true == 1, "⚡ Goal", "No goal")
            top_disagree = disagree_df.nlargest(10, "gap")[
                ["distance", "angle_deg", "actual", f"pred_{model_a}", f"pred_{model_b}", "gap"]
            ]
            st.dataframe(
                top_disagree.style.format({
                    "distance": "{:.1f}", "angle_deg": "{:.1f}",
                    f"pred_{model_a}": "{:.3f}", f"pred_{model_b}": "{:.3f}", "gap": "{:.3f}",
                }),
                width='stretch', hide_index=True,
            )

# ===========================================================================
# PAGE: ENSEMBLE BUILDER
# ===========================================================================

elif page == "🧬 Ensemble Builder":
    st.title("🧬 Ensemble Builder")
    st.caption("Blend any combination of your trained models with custom weights and see if the mix beats them all.")

    if preds_df is None:
        st.warning("⚠️ No `xg_test_predictions.csv` found. Upload your results zip from the sidebar.")
    else:
        pred_cols = [c for c in preds_df.columns if c.startswith("pred_")]
        model_names = [c.replace("pred_", "") for c in pred_cols]
        y_true = preds_df["actual_goal"]

        chosen = st.multiselect("Models to blend", model_names, default=model_names[:3])

        if len(chosen) < 2:
            st.info("Pick at least two models to build a blend.")
        else:
            st.subheader("⚖️ Weights")
            weights = {}
            weight_cols = st.columns(len(chosen))
            for col, name in zip(weight_cols, chosen):
                with col:
                    weights[name] = st.slider(name, 0.0, 1.0, round(1 / len(chosen), 2), 0.05, key=f"w_{name}")

            total_weight = sum(weights.values())
            if total_weight == 0:
                st.warning("All weights are zero — bump at least one slider up.")
            else:
                blended = sum(preds_df[f"pred_{name}"] * w for name, w in weights.items()) / total_weight

                blend_auc = auc(*roc_curve(y_true, blended)[:2])
                blend_brier = brier_score_loss(y_true, blended)

                individual_aucs = {name: auc(*roc_curve(y_true, preds_df[f"pred_{name}"])[:2]) for name in chosen}
                best_individual_name = max(individual_aucs, key=individual_aucs.get)
                best_individual_auc = individual_aucs[best_individual_name]

                c1, c2, c3 = st.columns(3)
                c1.metric("🧬 Blend ROC AUC", f"{blend_auc:.3f}",
                          f"{(blend_auc - best_individual_auc):+.3f} vs best single model")
                c2.metric("📉 Blend Brier", f"{blend_brier:.3f}")
                c3.metric("🏆 Best single model", best_individual_name, f"{best_individual_auc:.3f} AUC")

                if blend_auc > best_individual_auc:
                    st.success("🎉 This blend beats every individual model in the mix — worth keeping.")
                else:
                    st.info("This blend doesn't beat the best individual model here — try different weights or models.")

                fig = go.Figure()
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
                for name in chosen:
                    fpr, tpr, _ = roc_curve(y_true, preds_df[f"pred_{name}"])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({individual_aucs[name]:.3f})",
                                              line=dict(dash="dot")))
                fpr_b, tpr_b, _ = roc_curve(y_true, blended)
                fig.add_trace(go.Scatter(x=fpr_b, y=tpr_b, mode="lines", name=f"🧬 Blend ({blend_auc:.3f})",
                                          line=dict(color="crimson", width=4)))
                fig.update_layout(title="Blend vs its ingredients", xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate")
                st.plotly_chart(style_fig(fig, 500), width='stretch')

                with st.expander("⬇️ Export blended predictions"):
                    export_df = preds_df.copy()
                    export_df["blended_prediction"] = blended
                    st.download_button(
                        "Download predictions with blend column",
                        export_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_with_blend.csv", mime="text/csv",
                    )

# ===========================================================================
# PAGE: EXPLAINABILITY
# ===========================================================================

elif page == "🧠 Explainability":
    st.title("🧠 Explainability")
    st.caption("What actually drives the predictions? Computed live from your trained model.")

    if preds_df is None or not best_xg_model_path:
        st.warning("⚠️ Need both `xg_test_predictions.csv` and a `best_xg_model__*.joblib` file. "
                   "Upload your results zip from the sidebar.")
    else:
        model = load_joblib_model(best_xg_model_path)
        model_label = os.path.basename(best_xg_model_path).replace("best_xg_model__", "").replace(".joblib", "").replace("_", " ")

        available_features = [c for c in FEATURE_ORDER if c in preds_df.columns]
        X_test = preds_df[available_features]
        y_test = preds_df["actual_goal"]

        tab_imp, tab_pdp, tab_shap = st.tabs(["📊 Feature Importance", "📈 Partial Dependence", "🧮 SHAP Values"])

        with tab_imp:
            st.caption(f"Permutation importance for **{model_label}** — how much ROC AUC drops when a "
                       "feature's values are shuffled. Bigger drop = more important feature.")
            n_repeats = st.slider("Repeats (more = more stable, slower)", 5, 50, 15, 5)
            if st.button("🧮 Compute permutation importance"):
                with st.spinner("Shuffling features and re-scoring..."):
                    result = permutation_importance(
                        model, X_test, y_test, scoring="roc_auc",
                        n_repeats=n_repeats, random_state=42,
                    )
                imp_df = pd.DataFrame({
                    "feature": available_features,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }).sort_values("importance_mean", ascending=True)

                fig = go.Figure(go.Bar(
                    x=imp_df["importance_mean"], y=imp_df["feature"], orientation="h",
                    error_x=dict(type="data", array=imp_df["importance_std"]),
                    marker_color="teal",
                ))
                fig.update_layout(title="Permutation importance (ROC AUC drop)", xaxis_title="Mean importance")
                st.plotly_chart(style_fig(fig, 420), width='stretch')
                st.dataframe(imp_df.sort_values("importance_mean", ascending=False), width='stretch', hide_index=True)

        with tab_pdp:
            st.caption(f"Holding every other feature at its median, how does **{model_label}**'s "
                       "predicted probability change as one feature varies?")
            pdp_feature = st.selectbox(
                "Feature to vary", [f for f in available_features if X_test[f].nunique() > 2],
            )
            base_row = X_test.median(numeric_only=True)
            grid = np.linspace(X_test[pdp_feature].min(), X_test[pdp_feature].max(), 40)
            pdp_rows = pd.DataFrame([base_row.to_dict()] * len(grid))
            pdp_rows[pdp_feature] = grid
            pdp_rows = pdp_rows[available_features]
            pdp_proba = model.predict_proba(pdp_rows)[:, 1]

            fig = px.line(x=grid, y=pdp_proba, labels={"x": pdp_feature, "y": "Predicted P(goal)"},
                           title=f"Partial dependence — {pdp_feature}")
            st.plotly_chart(style_fig(fig, 420), width='stretch')

        with tab_shap:
            try:
                import shap
                shap_available = True
            except ImportError:
                shap_available = False

            if not shap_available:
                st.info("SHAP isn't installed. Run `pip install shap` and reload the app to unlock this tab — "
                        "it gives per-feature, per-shot contribution values (more granular than permutation "
                        "importance above).")
            else:
                st.caption(f"SHAP explains **{model_label}**'s predictions by attributing each shot's probability "
                           "to individual feature contributions. Slower than permutation importance, so it runs "
                           "on a sample rather than the full test set.")
                sample_size = st.slider("Sample size to explain", 20, min(300, len(X_test)), min(80, len(X_test)), 10)

                if st.button("🧮 Compute SHAP values"):
                    with st.spinner(f"Explaining {sample_size} shots — this can take a moment..."):
                        X_sample = X_test.sample(sample_size, random_state=42)
                        background = X_test.sample(min(50, len(X_test)), random_state=1)
                        explainer = shap.Explainer(model.predict_proba, background)
                        shap_values = explainer(X_sample)
                        # Binary classifier: take the "goal" class output
                        values = shap_values.values[..., 1] if shap_values.values.ndim == 3 else shap_values.values

                    mean_abs = pd.DataFrame({
                        "feature": available_features,
                        "mean_abs_shap": np.abs(values).mean(axis=0),
                    }).sort_values("mean_abs_shap", ascending=True)

                    fig = px.bar(mean_abs, x="mean_abs_shap", y="feature", orientation="h",
                                 title="Mean |SHAP value| — average impact on predicted probability")
                    st.plotly_chart(style_fig(fig, 420), width='stretch')

                    st.subheader("🔎 One shot, explained")
                    row_idx = st.slider("Which sampled shot to break down", 0, sample_size - 1, 0)
                    row_values = values[row_idx]
                    row_features = X_sample.iloc[row_idx]
                    base_value = shap_values.base_values[row_idx]
                    base_value = base_value[1] if hasattr(base_value, "__len__") else base_value

                    waterfall_df = pd.DataFrame({
                        "feature": [f"{f} = {row_features[f]:.2f}" for f in available_features],
                        "contribution": row_values,
                    }).sort_values("contribution")

                    fig2 = go.Figure(go.Waterfall(
                        orientation="h",
                        y=["Base rate"] + waterfall_df["feature"].tolist(),
                        x=[base_value] + waterfall_df["contribution"].tolist(),
                        measure=["absolute"] + ["relative"] * len(waterfall_df),
                    ))
                    predicted = model.predict_proba(X_sample.iloc[[row_idx]])[0, 1]
                    fig2.update_layout(title=f"How this shot's {predicted:.1%} prediction was built")
                    st.plotly_chart(style_fig(fig2, 420), width='stretch')

# ===========================================================================
# PAGE: AUTOML LAB
# ===========================================================================

elif page == "🛠️ AutoML Lab":
    st.title("🛠️ AutoML Lab")
    st.caption("Train a brand-new model, right here, on whatever shots data you've loaded — no notebook required.")

    if shots_df is None or "goal" not in shots_df.columns:
        st.warning("⚠️ Need a shots CSV with a `goal` column. Upload one from the sidebar.")
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        lab_df = shots_df.copy()
        if "is_header" not in lab_df.columns and "shot_body_part" in lab_df.columns:
            lab_df["is_header"] = (lab_df["shot_body_part"] == "Head").astype(int)
            lab_df["is_left_foot"] = (lab_df["shot_body_part"] == "Left Foot").astype(int)
            lab_df["is_right_foot"] = (lab_df["shot_body_part"] == "Right Foot").astype(int)
        if "is_open_play" not in lab_df.columns and "shot_type" in lab_df.columns:
            lab_df["is_open_play"] = (lab_df["shot_type"] == "Open Play").astype(int)
            lab_df["is_set_piece"] = (lab_df["shot_type"].isin(["Free Kick", "Corner"])).astype(int)

        numeric_candidates = [c for c in lab_df.columns
                               if pd.api.types.is_numeric_dtype(lab_df[c]) and c not in ("goal", "match_id")]

        c1, c2 = st.columns([2, 1])
        with c1:
            feature_cols = st.multiselect(
                "🧩 Features to train on", numeric_candidates,
                default=[c for c in FEATURE_ORDER if c in numeric_candidates],
            )
        with c2:
            algo_name = st.selectbox(
                "🤖 Algorithm",
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF)",
                 "K-Nearest Neighbors", "Naive Bayes", "Neural Net (MLP)"],
            )

        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

        with st.expander("⚙️ Advanced tuning"):
            use_cv = st.checkbox("Report cross-validated ROC AUC (more reliable than a single split)", value=False)
            cv_folds = st.slider("CV folds", 3, 10, 5, disabled=not use_cv)
            use_grid_search = st.checkbox("🔍 Auto-tune hyperparameters (GridSearchCV)", value=False)
            st.caption("Grid search tries every combination below and keeps whichever scores best on "
                      "cross-validated ROC AUC — slower, but removes the guesswork.")

        PARAM_GRIDS = {
            "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
            "Random Forest": {"n_estimators": [200, 400], "max_depth": [4, 6, 10]},
            "Gradient Boosting": {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
            "SVM (RBF)": {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", "auto"]},
            "K-Nearest Neighbors": {"clf__n_neighbors": [10, 25, 50]},
            "Naive Bayes": {},
            "Neural Net (MLP)": {"clf__alpha": [0.0001, 0.001, 0.01]},
        }

        if len(feature_cols) < 1:
            st.info("Pick at least one feature to train on.")
        elif st.button("🚀 Train model"):
            model_df = lab_df.dropna(subset=feature_cols + ["goal"])
            X = model_df[feature_cols]
            y = model_df["goal"]

            if y.nunique() < 2:
                st.error("Not enough goal/no-goal variety in this data to train a classifier.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                builders = {
                    "Logistic Regression": lambda: Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
                    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
                    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
                    "SVM (RBF)": lambda: Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
                    "K-Nearest Neighbors": lambda: Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=25))]),
                    "Naive Bayes": lambda: GaussianNB(),
                    "Neural Net (MLP)": lambda: Pipeline([("scale", StandardScaler()),
                                                           ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42))]),
                }

                cv_summary = None
                if use_grid_search and PARAM_GRIDS.get(algo_name):
                    from sklearn.model_selection import GridSearchCV
                    with st.spinner(f"Grid-searching {algo_name} ({cv_folds}-fold CV)..."):
                        search = GridSearchCV(builders[algo_name](), PARAM_GRIDS[algo_name],
                                               scoring="roc_auc", cv=cv_folds, n_jobs=-1)
                        search.fit(X_train, y_train)
                        trained_model = search.best_estimator_
                        proba = trained_model.predict_proba(X_test)[:, 1]
                    st.success(f"Best params: `{search.best_params_}` — CV ROC AUC {search.best_score_:.3f}")
                else:
                    with st.spinner(f"Training {algo_name} on {len(X_train):,} shots..."):
                        trained_model = builders[algo_name]()
                        trained_model.fit(X_train, y_train)
                        proba = trained_model.predict_proba(X_test)[:, 1]

                    if use_cv:
                        from sklearn.model_selection import cross_val_score
                        with st.spinner(f"Running {cv_folds}-fold cross-validation..."):
                            cv_scores = cross_val_score(builders[algo_name](), X, y, cv=cv_folds, scoring="roc_auc")
                        cv_summary = (cv_scores.mean(), cv_scores.std())

                new_auc = auc(*roc_curve(y_test, proba)[:2])
                new_brier = brier_score_loss(y_test, proba)

                m1, m2, m3 = st.columns(3)
                m1.metric("🎯 Test ROC AUC", f"{new_auc:.3f}")
                m2.metric("📉 Brier score", f"{new_brier:.3f}")
                m3.metric("🎲 Train / Test size", f"{len(X_train):,} / {len(X_test):,}")
                if cv_summary:
                    st.info(f"📊 {cv_folds}-fold CV ROC AUC: **{cv_summary[0]:.3f} ± {cv_summary[1]:.3f}** "
                           "(more trustworthy than the single test-set score above)")

                if results_df is not None and "xG" in results_df["task"].values:
                    zoo_best = results_df[results_df["task"] == "xG"]["roc_auc"].max()
                    if new_auc > zoo_best:
                        st.success(f"🎉 This beats the best model from your notebook's zoo ({zoo_best:.3f} ROC AUC)!")
                    else:
                        st.info(f"Notebook's best zoo model still leads at {zoo_best:.3f} ROC AUC.")

                tab_roc, tab_imp = st.tabs(["📈 ROC Curve", "📊 Feature Importance"])
                with tab_roc:
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    fig = go.Figure()
                    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{algo_name} (AUC={new_auc:.3f})",
                                              line=dict(color="crimson", width=3)))
                    fig.update_layout(title="ROC Curve — freshly trained model",
                                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(style_fig(fig, 460), width='stretch')
                with tab_imp:
                    raw_model = trained_model.named_steps["clf"] if hasattr(trained_model, "named_steps") else trained_model
                    if hasattr(raw_model, "feature_importances_"):
                        imp = pd.DataFrame({"feature": feature_cols, "importance": raw_model.feature_importances_}) \
                            .sort_values("importance")
                        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance")
                        st.plotly_chart(style_fig(fig, 420), width='stretch')
                    elif hasattr(raw_model, "coef_"):
                        imp = pd.DataFrame({"feature": feature_cols, "coefficient": raw_model.coef_[0]}) \
                            .sort_values("coefficient")
                        fig = px.bar(imp, x="coefficient", y="feature", orientation="h",
                                     title="Logistic regression coefficients")
                        st.plotly_chart(style_fig(fig, 420), width='stretch')
                    else:
                        st.info(f"{algo_name} doesn't expose a simple feature-importance view.")

                buf = io.BytesIO()
                joblib.dump(trained_model, buf)
                st.download_button(
                    "⬇️ Download this trained model (.joblib)", buf.getvalue(),
                    file_name=f"custom_{algo_name.replace(' ', '_')}.joblib",
                )

# ===========================================================================
# PAGE: BATCH PREDICTOR
# ===========================================================================

elif page == "📥 Batch Predictor":
    st.title("📥 Batch Predictor")
    st.caption("Score a whole spreadsheet of hypothetical or real shots at once with your best trained model.")

    if not best_xg_model_path:
        st.warning("⚠️ No `best_xg_model__*.joblib` found. Upload your results zip from the sidebar.")
    else:
        model = load_joblib_model(best_xg_model_path)
        model_label = os.path.basename(best_xg_model_path).replace("best_xg_model__", "").replace(".joblib", "").replace("_", " ")
        st.success(f"Loaded model: **{model_label}**")

        template_df = pd.DataFrame([
            {"x": 108, "y": 40, "body_part": "Right Foot", "shot_type": "Open Play", "under_pressure": 0},
            {"x": 95, "y": 25, "body_part": "Head", "shot_type": "Open Play", "under_pressure": 1},
        ])
        st.download_button(
            "⬇️ Download a template CSV", template_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_shots_template.csv", mime="text/csv",
        )

        batch_upload = st.file_uploader(
            "Upload shots to score — needs either (x, y) or (distance, angle_deg), "
            "plus body_part, shot_type, under_pressure",
            type="csv",
        )

        if batch_upload is not None:
            batch_df = pd.read_csv(batch_upload)
            st.write(f"Loaded {len(batch_df)} rows")

            if {"x", "y"}.issubset(batch_df.columns) and "distance" not in batch_df.columns:
                GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 7.32
                batch_df["distance"] = np.sqrt((GOAL_X - batch_df["x"]) ** 2 + (GOAL_Y - batch_df["y"]) ** 2)
                batch_df["angle_deg"] = np.degrees(np.abs(np.arctan2(
                    GOAL_WIDTH * (GOAL_X - batch_df["x"]),
                    (GOAL_X - batch_df["x"]) ** 2 + (batch_df["y"] - GOAL_Y) ** 2 - (GOAL_WIDTH / 2) ** 2,
                )))

            body_col = "body_part" if "body_part" in batch_df.columns else "shot_body_part"
            if body_col in batch_df.columns:
                batch_df["is_header"] = (batch_df[body_col] == "Head").astype(int)
                batch_df["is_left_foot"] = (batch_df[body_col] == "Left Foot").astype(int)
                batch_df["is_right_foot"] = (batch_df[body_col] == "Right Foot").astype(int)
            if "shot_type" in batch_df.columns:
                batch_df["is_open_play"] = (batch_df["shot_type"] == "Open Play").astype(int)
                batch_df["is_set_piece"] = (~batch_df["shot_type"].isin(["Open Play"])).astype(int)
            if "under_pressure" not in batch_df.columns:
                batch_df["under_pressure"] = 0

            missing = [c for c in FEATURE_ORDER if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns after preprocessing: {missing}. "
                        "Check your CSV has (x,y) or (distance, angle_deg), plus body_part and shot_type.")
            else:
                batch_df["predicted_xg"] = model.predict_proba(batch_df[FEATURE_ORDER])[:, 1]

                c1, c2, c3 = st.columns(3)
                c1.metric("🎯 Shots scored", f"{len(batch_df):,}")
                c2.metric("Σ Predicted xG", f"{batch_df['predicted_xg'].sum():.2f}")
                c3.metric("Avg. predicted xG", f"{batch_df['predicted_xg'].mean():.3f}")

                fig = px.histogram(batch_df, x="predicted_xg", nbins=30, title="Distribution of predicted xG")
                st.plotly_chart(style_fig(fig, 380), width='stretch')

                st.dataframe(
                    batch_df.sort_values("predicted_xg", ascending=False).style.format({"predicted_xg": "{:.3f}"}),
                    width='stretch', height=420,
                )
                st.download_button(
                    "⬇️ Download scored shots", batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv", mime="text/csv",
                )

# ===========================================================================
# PAGE: SHOT EXPLORER
# ===========================================================================

elif page == "🗺️ Shot Explorer":
    st.title("🗺️ Shot Explorer")

    if shots_df is None:
        st.warning("⚠️ No shots CSV found. Upload `xg_psxg_shots.csv` (or `shots_full_features.csv`) from the sidebar.")
    else:
        with st.expander("🔧 Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                teams = sorted(shots_df["team"].dropna().unique()) if "team" in shots_df.columns else []
                selected_teams = st.multiselect("Team", teams, default=teams)
            with fc2:
                bodyparts = sorted(shots_df["shot_body_part"].dropna().unique()) if "shot_body_part" in shots_df.columns else []
                selected_bp = st.multiselect("Body part", bodyparts, default=bodyparts)
            with fc3:
                xg_col_options = [c for c in ["statsbomb_xg", "my_xg", "my_psxg"] if c in shots_df.columns]
                xg_col = st.selectbox("xG value for sizing/coloring", xg_col_options) if xg_col_options else None

        f = shots_df.copy()
        if teams:
            f = f[f["team"].isin(selected_teams)]
        if bodyparts:
            f = f[f["shot_body_part"].isin(selected_bp)]

        if f.empty:
            st.info("No shots match these filters.")
        else:
            tab_map, tab_heat, tab_players, tab_radar, tab_dist = st.tabs(
                ["🗺️ Shot Map", "🌡️ Danger Zones", "🧑‍🤝‍🧑 Player Breakdown", "🕸️ Player Radar", "📐 Distributions"]
            )

            with tab_map:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 Shots", f"{len(f):,}")
                if "goal" in f.columns:
                    m2.metric("⚡ Goals", f"{int(f['goal'].sum()):,}")
                if xg_col:
                    m3.metric(f"Σ {xg_col}", f"{f[xg_col].sum():.2f}")
                if "goal" in f.columns and xg_col:
                    m4.metric("Goals − xG", f"{(f['goal'].sum() - f[xg_col].sum()):+.2f}")

                if {"x", "y"}.issubset(f.columns):
                    fig = go.Figure()
                    for tr in pitch_lines_traces():
                        fig.add_trace(tr)

                    goal_col = f["goal"] if "goal" in f.columns else pd.Series(0, index=f.index)
                    misses = f[goal_col != 1]
                    goals = f[goal_col == 1]

                    fig.add_trace(go.Scatter(
                        x=misses["x"], y=misses["y"], mode="markers", name="No goal",
                        marker=dict(size=(misses[xg_col] * 40 + 5) if xg_col else 8, color="lightslategrey", opacity=0.45),
                        text=misses.get("player"), hovertemplate="%{text}<br>x=%{x}, y=%{y}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=goals["x"], y=goals["y"], mode="markers", name="Goal ⚡",
                        marker=dict(size=(goals[xg_col] * 40 + 8) if xg_col else 12, color="crimson",
                                    line=dict(width=1, color="black")),
                        text=goals.get("player"), hovertemplate="%{text}<br>x=%{x}, y=%{y}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=f"Shot map — marker size = {xg_col}" if xg_col else "Shot map",
                        xaxis=dict(visible=False, range=[58, 122]),
                        yaxis=dict(visible=False, range=[-2, 82]),
                    )
                    st.plotly_chart(style_fig(fig, 560), width='stretch')
                else:
                    st.info("No x/y location columns available to draw a shot map.")

            with tab_heat:
                if {"x", "y", "goal"}.issubset(f.columns):
                    min_shots_per_bin = st.slider("Smoothness (bin count — lower is smoother)", 6, 30, 16, key="heat_bins")
                    fig = px.density_heatmap(
                        f, x="x", y="y", z="goal", histfunc="avg",
                        nbinsx=min_shots_per_bin, nbinsy=int(min_shots_per_bin * 0.7),
                        color_continuous_scale="Turbo",
                        title="Goal rate by pitch zone — brighter = more dangerous",
                        labels={"z": "Goal rate"},
                    )
                    for tr in pitch_lines_traces(color="white"):
                        fig.add_trace(tr)
                    fig.update_layout(xaxis=dict(visible=False, range=[58, 122]),
                                       yaxis=dict(visible=False, range=[-2, 82]))
                    st.plotly_chart(style_fig(fig, 560), width='stretch')
                    st.caption("Each cell shows the average goal outcome (0–1) for shots taken from that zone — "
                               "not shot volume. A bright cell with very few shots can be noisy; cross-check "
                               "against the Shot Map tab.")
                else:
                    st.info("Needs `x`, `y`, and `goal` columns to build a danger-zone heatmap.")

            with tab_players:
                if "player" in f.columns and xg_col:
                    agg = (
                        f.groupby("player")
                        .agg(shots=("player", "size"),
                             goals=("goal", "sum") if "goal" in f.columns else ("player", "size"),
                             xg=(xg_col, "sum"))
                        .assign(xg_diff=lambda d: d["goals"] - d["xg"])
                    )
                    min_shots = st.slider("Minimum shots to qualify for alerts", 3, 20, 5, key="min_shots_alert")
                    qualified = agg[agg["shots"] >= min_shots]
                    if len(qualified) >= 3 and qualified["xg_diff"].std() > 0:
                        z = (qualified["xg_diff"] - qualified["xg_diff"].mean()) / qualified["xg_diff"].std()
                        overperformers = qualified[z > 1.5].sort_values("xg_diff", ascending=False)
                        underperformers = qualified[z < -1.5].sort_values("xg_diff")
                        if len(overperformers) or len(underperformers):
                            st.markdown("**🚨 Notable performers** *(≥{} shots, >1.5σ from the group average)*".format(min_shots))
                            a1, a2 = st.columns(2)
                            with a1:
                                for name, row in overperformers.iterrows():
                                    st.success(f"🔥 **{name}** — {row['xg_diff']:+.2f} goals vs xG over {int(row['shots'])} shots")
                            with a2:
                                for name, row in underperformers.iterrows():
                                    st.error(f"🥶 **{name}** — {row['xg_diff']:+.2f} goals vs xG over {int(row['shots'])} shots")

                    top20 = agg.sort_values("goals", ascending=False).head(20)
                    fig = px.bar(
                        top20.reset_index(), x="player", y=["goals", "xg"], barmode="group",
                        title="Goals vs xG — top 20 players by goals",
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(style_fig(fig, 480), width='stretch')
                    st.dataframe(
                        top20.style.format({"xg": "{:.2f}", "xg_diff": "{:+.2f}"})
                        .background_gradient(subset=["xg_diff"], cmap="RdYlGn"),
                        width='stretch',
                    )
                else:
                    st.info("Need `player` and an xG column to build this view.")

            with tab_radar:
                if "player" in f.columns:
                    all_players = sorted(f["player"].dropna().unique())
                    if len(all_players) < 2:
                        st.info("Need at least two players in the current filters to compare.")
                    else:
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            player_a = st.selectbox("🔵 Player A", all_players, index=0)
                        with rc2:
                            player_b = st.selectbox("🔴 Player B", all_players, index=min(1, len(all_players) - 1))

                        def player_stats(name):
                            sub = f[f["player"] == name]
                            stats = {
                                "Shots": len(sub),
                                "Goals": sub["goal"].sum() if "goal" in sub.columns else 0,
                                "Avg. xG/shot": sub[xg_col].mean() if xg_col else 0,
                                "Avg. distance": sub["distance"].mean() if "distance" in sub.columns else 0,
                                "Header %": (sub["shot_body_part"] == "Head").mean() * 100 if "shot_body_part" in sub.columns else 0,
                            }
                            return stats

                        stats_a = player_stats(player_a)
                        stats_b = player_stats(player_b)
                        categories = list(stats_a.keys())

                        # Normalize each category 0-1 across the whole filtered dataset so the radar is comparable
                        norm_a, norm_b = [], []
                        for cat in categories:
                            all_vals = [stats_a[cat], stats_b[cat]]
                            if "distance" in cat.lower():
                                # shorter distance = more dangerous, so invert
                                max_v = max(f["distance"].max(), 1) if "distance" in f.columns else 1
                                norm_a.append(1 - (stats_a[cat] / max_v))
                                norm_b.append(1 - (stats_b[cat] / max_v))
                            else:
                                max_v = max(all_vals) or 1
                                norm_a.append(stats_a[cat] / max_v)
                                norm_b.append(stats_b[cat] / max_v)

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=norm_a + [norm_a[0]], theta=categories + [categories[0]],
                                                       fill="toself", name=player_a))
                        fig.add_trace(go.Scatterpolar(r=norm_b + [norm_b[0]], theta=categories + [categories[0]],
                                                       fill="toself", name=player_b))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                           title=f"{player_a} vs {player_b}")
                        st.plotly_chart(style_fig(fig, 480), width='stretch')

                        compare_table = pd.DataFrame({player_a: stats_a, player_b: stats_b})
                        st.dataframe(compare_table.style.format("{:.2f}"), width='stretch')
                else:
                    st.info("Need a `player` column to build this view.")

            with tab_dist:
                d1, d2 = st.columns(2)
                with d1:
                    if "distance" in f.columns:
                        fig = px.histogram(f, x="distance", color="goal" if "goal" in f.columns else None,
                                            nbins=30, barmode="overlay", title="Shot distance distribution")
                        st.plotly_chart(style_fig(fig), width='stretch')
                with d2:
                    if "angle_deg" in f.columns:
                        fig = px.histogram(f, x="angle_deg", color="goal" if "goal" in f.columns else None,
                                            nbins=30, barmode="overlay", title="Shot angle distribution")
                        st.plotly_chart(style_fig(fig), width='stretch')

# ===========================================================================
# PAGE: MATCH CENTER
# ===========================================================================

elif page == "🏟️ Match Center":
    st.title("🏟️ Match Center")
    st.caption("Watch cumulative xG build up minute by minute for a single match.")

    required = {"match_id", "minute", "team", "goal"}
    if shots_df is None or not required.issubset(shots_df.columns):
        st.warning("⚠️ Needs a shots CSV with `match_id`, `minute`, `team`, and `goal` columns "
                   "(that's `xg_psxg_shots.csv`). Upload it from the sidebar.")
    else:
        xg_col_options = [c for c in ["statsbomb_xg", "my_xg", "my_psxg"] if c in shots_df.columns]
        if not xg_col_options:
            st.warning("⚠️ Needs at least one xG column (`statsbomb_xg`, `my_xg`, or `my_psxg`).")
        else:
            xg_col = st.selectbox("xG value to accumulate", xg_col_options)

            match_summary = (
                shots_df.groupby("match_id")
                .agg(teams=("team", lambda s: " vs ".join(sorted(s.dropna().unique())[:2])),
                     shots=("match_id", "size"), goals=("goal", "sum"))
                .reset_index()
            )
            match_summary["label"] = match_summary["teams"] + " — match " + match_summary["match_id"].astype(str)
            chosen_label = st.selectbox("Match", match_summary["label"])
            match_id = match_summary.loc[match_summary["label"] == chosen_label, "match_id"].iloc[0]

            match_shots = shots_df[shots_df["match_id"] == match_id].sort_values("minute").copy()
            teams_in_match = sorted(match_shots["team"].dropna().unique())[:2]

            score_cols = st.columns(len(teams_in_match) if teams_in_match else 1)
            for col, team_name in zip(score_cols, teams_in_match):
                team_goals = int(match_shots.loc[match_shots["team"] == team_name, "goal"].sum())
                team_xg = match_shots.loc[match_shots["team"] == team_name, xg_col].sum()
                col.metric(f"⚽ {team_name}", f"{team_goals} goals", f"{team_xg:.2f} xG")

            fig = go.Figure()
            colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
            for i, team_name in enumerate(teams_in_match):
                team_shots = match_shots[match_shots["team"] == team_name].copy()
                team_shots["cum_xg"] = team_shots[xg_col].cumsum()
                # Extend to minute 90 for a clean step chart
                x_vals = [0] + team_shots["minute"].tolist()
                y_vals = [0] + team_shots["cum_xg"].tolist()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines", name=team_name, line=dict(shape="hv", color=colors[i % 4]),
                ))
                goal_shots = team_shots[team_shots["goal"] == 1]
                if len(goal_shots):
                    fig.add_trace(go.Scatter(
                        x=goal_shots["minute"], y=goal_shots["cum_xg"], mode="markers",
                        marker=dict(size=13, color=colors[i % 4], symbol="star", line=dict(width=1, color="black")),
                        name=f"{team_name} goal ⚡", showlegend=True,
                    ))
            fig.update_layout(
                title=f"Cumulative {xg_col} race — {chosen_label}",
                xaxis_title="Minute", yaxis_title=f"Cumulative {xg_col}",
            )
            st.plotly_chart(style_fig(fig, 520), width='stretch')

            st.subheader("📋 Shot log")
            log_cols = [c for c in ["minute", "player", "team", "shot_body_part", "shot_type", xg_col, "goal"]
                        if c in match_shots.columns]
            st.dataframe(
                match_shots[log_cols].style.format({xg_col: "{:.3f}"}),
                width='stretch', hide_index=True, height=350,
            )

            st.divider()
            st.subheader("🎲 Monte Carlo Match Simulator")
            st.caption(
                f"Treats every shot's {xg_col} as an independent coin flip and replays the match thousands of "
                "times, to see how much of the actual scoreline was down to variance vs. genuine chance creation."
            )

            if len(teams_in_match) != 2:
                st.info("Needs exactly two teams' worth of shots in this match to simulate.")
            else:
                n_sims = st.slider("Number of simulations", 1000, 20000, 5000, 1000)
                if st.button("🎲 Run simulation"):
                    team_a, team_b = teams_in_match
                    xg_a_list = match_shots.loc[match_shots["team"] == team_a, xg_col].clip(0, 1).values
                    xg_b_list = match_shots.loc[match_shots["team"] == team_b, xg_col].clip(0, 1).values

                    rng = np.random.default_rng(42)
                    with st.spinner(f"Replaying the match {n_sims:,} times..."):
                        goals_a = (rng.random((n_sims, len(xg_a_list))) < xg_a_list).sum(axis=1) if len(xg_a_list) else np.zeros(n_sims, dtype=int)
                        goals_b = (rng.random((n_sims, len(xg_b_list))) < xg_b_list).sum(axis=1) if len(xg_b_list) else np.zeros(n_sims, dtype=int)

                    win_a = (goals_a > goals_b).mean()
                    win_b = (goals_b > goals_a).mean()
                    draw = (goals_a == goals_b).mean()

                    actual_goals_a = int(match_shots.loc[match_shots["team"] == team_a, "goal"].sum())
                    actual_goals_b = int(match_shots.loc[match_shots["team"] == team_b, "goal"].sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"🏆 {team_a} win", f"{win_a:.1%}")
                    c2.metric("🤝 Draw", f"{draw:.1%}")
                    c3.metric(f"🏆 {team_b} win", f"{win_b:.1%}")

                    max_goals = int(max(goals_a.max(), goals_b.max(), actual_goals_a, actual_goals_b)) + 1
                    scoreline_counts = np.zeros((max_goals, max_goals))
                    for ga, gb in zip(goals_a, goals_b):
                        if ga < max_goals and gb < max_goals:
                            scoreline_counts[ga, gb] += 1
                    scoreline_prob = scoreline_counts / n_sims

                    fig = px.imshow(
                        scoreline_prob, color_continuous_scale="Turbo",
                        labels=dict(x=f"{team_b} goals", y=f"{team_a} goals", color="Probability"),
                        title="Simulated scoreline probability grid",
                    )
                    fig.add_annotation(
                        x=actual_goals_b, y=actual_goals_a, text="⭐ actual", showarrow=True,
                        arrowhead=2, font=dict(color="white", size=13),
                    )
                    st.plotly_chart(style_fig(fig, 480), width='stretch')

                    most_likely_idx = np.unravel_index(scoreline_counts.argmax(), scoreline_counts.shape)
                    st.caption(
                        f"Most likely simulated scoreline: **{team_a} {most_likely_idx[0]}–{most_likely_idx[1]} {team_b}** "
                        f"({scoreline_prob[most_likely_idx]:.1%} of simulations) · "
                        f"Actual result: **{team_a} {actual_goals_a}–{actual_goals_b} {team_b}**"
                    )

# ===========================================================================
# PAGE: LIVE PREDICTOR
# ===========================================================================

elif page == "🔮 Live Predictor":
    st.title("🔮 Live xG Predictor")
    st.caption("Set up a shot and get an instant probability from your best trained model.")

    if not best_xg_model_path:
        st.warning("⚠️ No `best_xg_model__*.joblib` found. Upload your results zip from the sidebar.")
    else:
        model = load_joblib_model(best_xg_model_path)
        model_label = os.path.basename(best_xg_model_path).replace("best_xg_model__", "").replace(".joblib", "").replace("_", " ")
        st.success(f"Loaded model: **{model_label}**")

        compare_mode = st.toggle("🆚 Compare two shot setups side by side", value=False)

        def shot_inputs(key_prefix, default_x=108.0, default_y=40.0):
            x = st.slider("Pitch position — x (120=goal line)", 60.0, 120.0, default_x, 0.5, key=f"{key_prefix}_x")
            y = st.slider("Pitch position — y (40=center)", 0.0, 80.0, default_y, 0.5, key=f"{key_prefix}_y")
            body_part = st.radio("👟 Body part", ["Right Foot", "Left Foot", "Head"], horizontal=True, key=f"{key_prefix}_bp")
            shot_type = st.radio("🎬 Shot type", ["Open Play", "Set Piece (FK/Corner)"], horizontal=True, key=f"{key_prefix}_st")
            under_pressure = st.toggle("😰 Under pressure", value=False, key=f"{key_prefix}_up")
            return x, y, body_part, shot_type, under_pressure

        def compute_features(x, y, body_part, shot_type, under_pressure):
            GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 7.32
            distance = float(np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2))
            angle = float(np.abs(np.arctan2(
                GOAL_WIDTH * (GOAL_X - x), (GOAL_X - x) ** 2 + (y - GOAL_Y) ** 2 - (GOAL_WIDTH / 2) ** 2,
            )))
            angle_deg = float(np.degrees(angle))
            row = pd.DataFrame([{
                "distance": distance, "angle_deg": angle_deg,
                "is_header": 1 if body_part == "Head" else 0,
                "is_left_foot": 1 if body_part == "Left Foot" else 0,
                "is_right_foot": 1 if body_part == "Right Foot" else 0,
                "is_open_play": 1 if shot_type == "Open Play" else 0,
                "is_set_piece": 1 if shot_type != "Open Play" else 0,
                "under_pressure": 1 if under_pressure else 0,
            }])[FEATURE_ORDER]
            return row, distance, angle_deg

        def gauge(value, title):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value * 100, number={"suffix": "%", "font": {"size": 40}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "crimson"},
                       "steps": [{"range": [0, 15], "color": "#f2f2f2"},
                                 {"range": [15, 40], "color": "#e0e0e0"},
                                 {"range": [40, 100], "color": "#c8f7c5"}]},
                title={"text": title},
            ))
            return style_fig(fig, 300)

        def mini_pitch(x, y):
            fig = go.Figure()
            for tr in pitch_lines_traces():
                fig.add_trace(tr)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                      marker=dict(size=20, color="crimson", line=dict(width=2, color="black")),
                                      showlegend=False))
            fig.update_layout(xaxis=dict(visible=False, range=[58, 122]), yaxis=dict(visible=False, range=[-2, 82]),
                               margin=dict(l=10, r=10, t=10, b=10))
            return style_fig(fig, 260)

        if not compare_mode:
            left, right = st.columns([2, 3])
            with left:
                st.subheader("⚙️ Shot setup")
                x, y, body_part, shot_type, under_pressure = shot_inputs("single")
                features, distance, angle_deg = compute_features(x, y, body_part, shot_type, under_pressure)
                st.caption(f"📏 Distance: **{distance:.1f}**  •  📐 Angle: **{angle_deg:.1f}°**")
                xg_value = float(model.predict_proba(features)[0, 1])
            with right:
                st.plotly_chart(gauge(xg_value, "Probability this shot is a goal"), width='stretch')
                st.plotly_chart(mini_pitch(x, y), width='stretch')
                if xg_value >= 0.4:
                    st.info("🔥 High-quality chance.")
                elif xg_value >= 0.15:
                    st.info("👍 Decent chance.")
                else:
                    st.info("🥶 Low-probability chance.")
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                st.subheader("🔵 Shot A")
                xa, ya, bpa, sta, upa = shot_inputs("a", 108.0, 34.0)
                feat_a, dist_a, ang_a = compute_features(xa, ya, bpa, sta, upa)
                xg_a = float(model.predict_proba(feat_a)[0, 1])
                st.plotly_chart(gauge(xg_a, "Shot A xG"), width='stretch')
            with col_y:
                st.subheader("🔴 Shot B")
                xb, yb, bpb, stb, upb = shot_inputs("b", 108.0, 46.0)
                feat_b, dist_b, ang_b = compute_features(xb, yb, bpb, stb, upb)
                xg_b = float(model.predict_proba(feat_b)[0, 1])
                st.plotly_chart(gauge(xg_b, "Shot B xG"), width='stretch')

            diff = xg_a - xg_b
            if abs(diff) < 0.01:
                st.success("🤝 Both shots are roughly equally dangerous.")
            elif diff > 0:
                st.success(f"🔵 Shot A is the better chance — {diff:+.1%} higher xG.")
            else:
                st.success(f"🔴 Shot B is the better chance — {-diff:+.1%} higher xG.")

            fig = go.Figure()
            for tr in pitch_lines_traces():
                fig.add_trace(tr)
            fig.add_trace(go.Scatter(x=[xa], y=[ya], mode="markers+text", text=["A"], textposition="top center",
                                      marker=dict(size=20, color="#1f77b4", line=dict(width=2, color="black")), showlegend=False))
            fig.add_trace(go.Scatter(x=[xb], y=[yb], mode="markers+text", text=["B"], textposition="top center",
                                      marker=dict(size=20, color="#d62728", line=dict(width=2, color="black")), showlegend=False))
            fig.update_layout(xaxis=dict(visible=False, range=[58, 122]), yaxis=dict(visible=False, range=[-2, 82]),
                               title="Both shot locations")
            st.plotly_chart(style_fig(fig, 320), width='stretch')

# ===========================================================================
# PAGE: CHART GALLERY
# ===========================================================================

elif page == "🖼️ Chart Gallery":
    st.title("🖼️ Chart Gallery")
    st.caption("Every image your notebook exported, straight from the results zip.")

    if not results_dir:
        st.warning("⚠️ No results loaded. Upload your results zip from the sidebar.")
    else:
        images = sorted(glob.glob(os.path.join(results_dir, "*.png")))
        if not images:
            st.info("No PNG images found in the results folder.")
        else:
            search = st.text_input("🔎 Filter by name", "")
            filtered_images = [p for p in images if search.lower() in os.path.basename(p).lower()]
            cols = st.columns(2)
            for i, img_path in enumerate(filtered_images):
                label = os.path.basename(img_path).replace(".png", "").replace("_", " ").title()
                with cols[i % 2]:
                    st.image(img_path, caption=f"📌 {label}", width='stretch')

# ===========================================================================
# PAGE: RAW DATA
# ===========================================================================

elif page == "📁 Raw Data":
    st.title("📁 Raw Data")

    tab_names, tab_data = [], []
    if results_df is not None:
        tab_names.append("📊 Model results"); tab_data.append(results_df)
    if preds_df is not None:
        tab_names.append("🔮 Test predictions"); tab_data.append(preds_df)
    if shots_df is not None:
        tab_names.append("🎯 Shots dataset"); tab_data.append(shots_df)

    if not tab_data:
        st.warning("⚠️ Nothing loaded yet — use the sidebar to upload your results zip and/or shots CSV.")
    else:
        tabs = st.tabs(tab_names)
        for tab, name, data in zip(tabs, tab_names, tab_data):
            with tab:
                search = st.text_input("🔎 Search (any column, text match)", "", key=f"search_{name}")
                view = data
                if search:
                    mask = data.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                    view = data[mask]
                st.caption(f"{len(view):,} of {len(data):,} rows shown")
                st.dataframe(view, width='stretch', height=500)
                st.download_button(
                    f"⬇️ Download {name.split(' ', 1)[1]} as CSV",
                    view.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name.split(' ', 1)[1].lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )
