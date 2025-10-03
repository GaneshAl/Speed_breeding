
import os
import itertools
import numpy as np
import pandas as pd
from pyDOE2 import ccdesign
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Try to import shap if available
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

# For interactive dataframe display (works in this notebook environment)
try:
    from caas_jupyter_tools import display_dataframe_to_user
    HAVE_DISPLAY = True
except Exception:
    HAVE_DISPLAY = False

# ---------------------------
# Configuration
# ---------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
OUTPUT_DIR = r"D:\Speed breeding proposal - R1\Multi model resutls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CCD decode settings (same levels used in manuscript/code)
alpha = 1.682
decode_params = {
    "Photoperiod_h":        (14, 16, 18, 20, 22),
    "LightIntensity_PPFD":  (200, 300, 400, 500, 600),
    "DayTemp_C":            (18, 20, 22, 24, 26),
    "NightTemp_C":          (12, 14, 16, 18, 20),
    "Humidity_percent":     (40, 50, 55, 60, 65),
    "CO2_ppm":              (400, 500, 600, 700, 800)
}
FEATURE_NAMES = list(decode_params.keys())

# Helper: decode a single ccd row (values are -alpha, -1, 0, 1, alpha)
def decode_row(row):
    decoded = {}
    for k, v in zip(FEATURE_NAMES, row):
        mapping = {-alpha: decode_params[k][0], -1: decode_params[k][1], 0: decode_params[k][2], 1: decode_params[k][3], alpha: decode_params[k][4]}
        # Round the value for matching due to float representation
        key = float(np.round(v, 3))
        decoded[k] = mapping.get(key, decode_params[k][2])
    return decoded

# ---------------------------
# Synthetic models
# ---------------------------
def sim_A(P, L, Td, Tn, H, C, rng):
    # Baseline linear-like model (original)
    base = (P - 14) * 2.0 + (Td - 20) * 2.5 + (C - 400) * 0.05
    synergy = 0.01 * L * (H / 100.0)
    temp_diff_penalty = -1.5 * abs(Td - Tn - 6)
    noise = rng.normal(0, 5)
    return 50 + base + synergy + temp_diff_penalty + noise

def sim_B(P, L, Td, Tn, H, C, rng):
    # Saturation & thermal optimum
    gP = 10.0 / (1.0 + np.exp(-1.0 * (P - 18.0)))            # logistic saturation in photoperiod
    hC = 25.0 * (C) / (400.0 + C)                            # Michaelis-Menten-like for CO2
    sLH = 0.02 * (L * (H / 100.0)) / (1.0 + 0.003 * L)       # diminishing returns synergy
    bell = -0.6 * (Td - 24.0) ** 2                           # bell-shaped temp term (optimum ~24C)
    dT = -1.5 * abs(Td - Tn - 6.0)
    noise = rng.normal(0, 5.0)
    return 50.0 + gP + hC + sLH + bell + dT + noise

def sim_C(P, L, Td, Tn, H, C, rng):
    # Strong interactions & heteroscedastic noise
    x = np.array([P, L / 100.0, Td, Tn, H / 10.0, C / 100.0])
    w = np.array([1.6, 0.5, 1.2, -0.4, 0.3, 0.8])
    inter = 0.002 * L * (H / 100.0) + 0.03 * (P - 18.0) * (C / 100.0 - 6.0)
    dT = -1.8 * abs(Td - Tn - 6.0)
    mu = 50.0 + float(w.dot(x)) + inter + dT
    sd = max(1.0, 3.0 + 0.002 * (max(mu, 0.0) ** 2) ** 0.5)
    return mu + rng.normal(0.0, sd)

def sim_D(P, L, Td, Tn, H, C, rng):
    # Thresholds/plateaus & U-shaped humidity penalty
    if P < 16.0:
        fP = 0.0
    else:
        fP = min(P - 16.0, 4.0) * 2.2  # plateaus around 20h
    if C < 600.0:
        fC = 0.06 * (C - 400.0)
    else:
        fC = 0.06 * 200.0 + 0.02 * (C - 600.0)
    uH = -0.08 * (H - 57.5) ** 2
    sLH = 0.012 * L * (H / 100.0)
    dT = -1.5 * abs(Td - Tn - 6.0)
    noise = rng.normal(0.0, 5.0)
    return 50.0 + fP + fC + uH + sLH + dT + noise

def sim_E(P, L, Td, Tn, H, C, rng):
    # Non-separable + periodic photoperiod contribution
    qP = 4.0 * np.sin(np.pi * (P - 12.0) / 10.0)
    poly = 1.5 * (P - 18.0) + 0.0015 * L + 1.1 * (Td - 22.0) - 0.7 * (Tn - 16.0) + 0.02 * (H - 55.0) + 0.04 * (C - 600.0)
    inter = 0.0008 * L * (C - 600.0) + 0.02 * (P - 18.0) * (Td - 22.0) - 0.015 * (H - 55.0) * (Td - Tn - 6.0)
    dT = -1.6 * abs(Td - Tn - 6.0)
    noise = rng.normal(0.0, 5.0)
    return 50.0 + qP + poly + inter + dT + noise

SIM_FUNCS = {
    "Model_A_linear": sim_A,
    "Model_B_saturation_optimum": sim_B,
    "Model_C_interactions_heterosc": sim_C,
    "Model_D_threshold_plateau": sim_D,
    "Model_E_periodic_nonseparable": sim_E,
}

# ---------------------------
# Pipeline functions
# ---------------------------
def generate_ccd_df():
    # generate CCD with 6 factors, center repeats (4,4) face circumscribed
    ccd_array = ccdesign(6, center=(4, 4), face='circumscribed')
    rows = []
    for row in ccd_array:
        decoded = decode_row(row)
        rows.append(decoded)
    ccd_df = pd.DataFrame(rows)
    return ccd_df

def simulate_and_train(sim_func, model_name):
    # Generate CCD data and simulate yield using sim_func
    ccd_df = generate_ccd_df()
    # simulate yields
    y = []
    for _, r in ccd_df.iterrows():
        y.append(sim_func(r["Photoperiod_h"], r["LightIntensity_PPFD"], r["DayTemp_C"], r["NightTemp_C"], r["Humidity_percent"], r["CO2_ppm"], rng))
    ccd_df = ccd_df.copy()
    ccd_df["Yield"] = y

    # Train RF model
    X = ccd_df[FEATURE_NAMES].copy()
    y_series = ccd_df["Yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_series, test_size=0.2, random_state=RANDOM_SEED)
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # Evaluate
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)
    rmse = mean_squared_error(y_test, pred_test, squared=False)
    r2 = r2_score(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)

    # Full factorial predictions (5 levels each -> 15625 rows)
    levels = {k: list(v) for k, v in decode_params.items()}
    full_combinations = list(itertools.product(*levels.values()))
    full_df = pd.DataFrame(full_combinations, columns=levels.keys())
    full_df["Predicted_Yield"] = model.predict(full_df[FEATURE_NAMES])

    # Optimal combination
    optimal_idx = full_df["Predicted_Yield"].idxmax()
    optimal_row = full_df.loc[optimal_idx].to_dict()

    # Feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importances}).sort_values("Importance", ascending=False)

    # SHAP if available (TreeExplainer)
    shap_summary_saved = None
    shap_beeswarm_saved = None
    if HAVE_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # summary bar
            plt.figure(figsize=(6,4))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.tight_layout()
            shap_summary_saved = os.path.join(OUTPUT_DIR, f"{model_name}_shap_summary.png")
            plt.savefig(shap_summary_saved, dpi=150)
            plt.close()
            # beeswarm
            plt.figure(figsize=(6,6))
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            shap_beeswarm_saved = os.path.join(OUTPUT_DIR, f"{model_name}_shap_beeswarm.png")
            plt.savefig(shap_beeswarm_saved, dpi=150)
            plt.close()
        except Exception as e:
            print(f"SHAP failed for {model_name}: {e}")

    # Train/Test scatter plot
    ttsave = os.path.join(OUTPUT_DIR, f"{model_name}_train_vs_test.png")
    plt.figure(figsize=(6,4))
    plt.scatter(y_train, pred_train, label='Train', alpha=0.7)
    plt.scatter(y_test, pred_test, label='Test', alpha=0.7)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'{model_name}: Train vs Test')
    plt.legend()
    mn = min(min(y_train.min(), y_test.min()), min(pred_train.min(), pred_test.min()))
    mx = max(max(y_train.max(), y_test.max()), max(pred_train.max(), pred_test.max()))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.tight_layout()
    plt.savefig(ttsave, dpi=150)
    plt.close()

    # Residuals plot
    ressave = os.path.join(OUTPUT_DIR, f"{model_name}_residuals.png")
    residuals = y_test - pred_test
    plt.figure(figsize=(6,4))
    plt.scatter(pred_test, residuals, alpha=0.7)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals (Actual - Pred)')
    plt.title(f'{model_name}: Residuals')
    plt.tight_layout()
    plt.savefig(ressave, dpi=150)
    plt.close()

    # Save outputs
    ccd_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_ccd_data.csv"), index=False)
    full_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_full_grid_predictions.csv"), index=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_feature_importances.csv"), index=False)

    # Summary dict
    summary = {
        "model": model_name,
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "optimal_predicted_yield": float(full_df["Predicted_Yield"].max()),
        "optimal_params": {k: float(v) for k, v in optimal_row.items() if k != "Predicted_Yield"},
        "feature_importances": importance_df.to_dict(orient="records"),
        "ccd_rows": len(ccd_df),
        "shap_summary": shap_summary_saved,
        "shap_beeswarm": shap_beeswarm_saved,
        "train_test_plot": ttsave,
        "residual_plot": ressave
    }

    return summary

# ---------------------------
# Run for all models and collect summaries
# ---------------------------
summaries = []
for name, func in SIM_FUNCS.items():
    print(f"Running pipeline for {name} ...")
    s = simulate_and_train(func, name)
    summaries.append(s)
    print(f"Completed {name}: RMSE={s['rmse']:.2f}, R2={s['r2']:.2f}, MAE={s['mae']:.2f}")

# Create a summary dataframe
summary_rows = []
for s in summaries:
    row = {
        "model": s["model"],
        "rmse": s["rmse"],
        "r2": s["r2"],
        "mae": s["mae"],
        "optimal_predicted_yield": s["optimal_predicted_yield"],
    }
    # expand optimal params into columns
    for k, v in s["optimal_params"].items():
        row[f"opt_{k}"] = v
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
summary_csv = os.path.join(OUTPUT_DIR, "models_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# Save full summaries as JSON too
with open(os.path.join(OUTPUT_DIR, "models_summary_full.json"), "w") as fh:
    import json
    json.dump(summaries, fh, indent=2)

# Display summary to user (not required but helpful)
if HAVE_DISPLAY:
    display_dataframe_to_user("Multi-model summary", summary_df)
else:
    print("\nSummary table:\n", summary_df.to_string(index=False))

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Summary CSV: {summary_csv}")

