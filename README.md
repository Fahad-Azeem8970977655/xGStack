# xG / PSxG Analytics Suite

## What's in this zip
- `xG_PSxG_ModelZoo_Colab.ipynb` — Colab notebook: downloads StatsBomb data (or loads a CSV you upload),
  trains the full model zoo (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM,
  SVM, KNN, Naive Bayes, MLP, Stacked Ensemble, Calibrated best model), and exports everything to
  `xg_project_results.zip`.
- `app.py` — advanced multi-page Streamlit dashboard (sidebar navigation, tabs, live predictor, chart gallery).
- `requirements.txt` — dependencies for the dashboard.

## 1. Run the notebook (if you haven't already)
Open `xG_PSxG_ModelZoo_Colab.ipynb` in Google Colab, run it top to bottom, and download the
`xg_project_results.zip` it produces at the end. You already have one of these if you're reading this.

## 2. Set up the dashboard folder

```
xg_dashboard/
├── app.py
├── requirements.txt
├── results/                  ← unzip xg_project_results.zip into here
│   ├── model_comparison_results.csv
│   ├── xg_test_predictions.csv
│   ├── best_xg_model__*.joblib
│   └── *.png
└── xg_psxg_shots.csv         ← optional, for the Shot Explorer page
```

If you'd rather not unzip anything, just run the app and use the **📦 Load / refresh data**
uploader in the sidebar — it accepts the results zip directly and extracts it in memory,
plus a separate uploader for your shots CSV.

## 3. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Pages in the dashboard
- 🏠 **Home** — headline metrics, top-5 leaderboard preview, and a one-click HTML summary report export
- 🗒️ **Auto Insights** — automatically mined, plain-English findings from whatever data is loaded:
  best/worst-calibrated model, biggest over/underperforming player, most dangerous pitch zone,
  first-half vs second-half conversion, most-correlated model pair, and more — no clicking required
- 📊 **Model Leaderboard** — every model ranked by ROC AUC / Brier / log loss, per task, with an
  interactive bar chart
- 🔬 **Model Deep-Dive** — interactive ROC curves, calibration curves, a confusion matrix with a
  live decision-threshold slider (or auto-pick the best-F1 threshold), precision/recall curves,
  and a bootstrap-resampling tab that shows a 95% confidence interval on ROC AUC instead of a
  single fragile number
- 🥊 **Model Battle** — pick any two models and see their predictions plotted against each other,
  their classification agreement rate, and a table of the shots where they disagree the most
- 🧬 **Ensemble Builder** — blend any combination of your trained models with custom weight
  sliders, see instantly whether the blend beats every individual model, and export the blended
  predictions
- 🧠 **Explainability** — live permutation feature importance, a partial-dependence view, and (if
  `shap` is installed) a **SHAP tab**: mean |SHAP value| per feature plus a waterfall breakdown of
  exactly how one individual shot's prediction was built, contribution by contribution
- 🛠️ **AutoML Lab** — train a brand-new model directly in the browser on whatever shots CSV you've
  loaded: pick features, pick an algorithm, optionally turn on **cross-validation** or
  **GridSearchCV hyperparameter tuning**, see its ROC curve and feature importance, and download
  the trained `.joblib` — no notebook required
- 📥 **Batch Predictor** — upload a spreadsheet of hypothetical or real shots (a template is
  provided) and score every row at once with your best trained model, then download the results
- 🗺️ **Shot Explorer** — filterable shot map (Plotly, hoverable), a **danger-zone heatmap** of
  goal rate by pitch location, player goals-vs-xG breakdown with automatic **over/under-performer
  alerts**, a **player radar comparison** (pick any two players, compare shot profiles on a radar
  chart), and distance/angle distributions
- 🏟️ **Match Center** — pick a match and watch cumulative xG build up minute by minute for each
  team, goals marked as stars, a full shot log, and a **Monte Carlo match simulator** that replays
  the game thousands of times from each shot's xG to get win/draw/loss probabilities and a full
  scoreline probability grid
- 🔮 **Live Predictor** — set a shot's pitch position, body part, shot type and pressure, and get
  an instant xG probability from your best trained model on a gauge + mini pitch map — or flip on
  **compare mode** to size up two shot setups side by side
- 🖼️ **Chart Gallery** — every PNG your notebook exported, browsable and searchable
- 📁 **Raw Data** — browse, search, and download every underlying table as CSV

## Chart appearance
Use the **🎨 Appearance** panel in the sidebar to switch every chart's Plotly theme
(`plotly`, `plotly_dark`, `ggplot2`, `seaborn`, `simple_white`, `presentation`).

## Optional: SHAP explainability
The Explainability page's SHAP tab needs the `shap` package (already in `requirements.txt`).
It's the heaviest dependency here — if you'd rather skip it, delete the `shap` line from
`requirements.txt` and that one tab will show an install hint instead of crashing the app.

## Deploying for free (optional)
Push this folder to a GitHub repo and deploy on [share.streamlit.io](https://share.streamlit.io)
pointing at `app.py`. Uploading the results zip through the sidebar works the same way there as locally.
