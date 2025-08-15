import os
import numpy as np
import pandas as pd
import itertools
from pyDOE2 import ccdesign
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Plot settings
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'Arial',
    'font.weight': 'bold'
})


# Set up reproducible random number generator
rng = np.random.default_rng(seed=42)

# Create output directory
output_dir = r"D:\Speed breeding proposal"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Central Composite Design
ccd_array = ccdesign(6, center=(4, 4), face='circumscribed')
ccd_df = pd.DataFrame(ccd_array, columns=[
    "Photoperiod_h", 
    "LightIntensity_PPFD", 
    "DayTemp_C", 
    "NightTemp_C", 
    "Humidity_percent", 
    "CO2_ppm"
])

# Step 2: Decode CCD
alpha = 1.682
decode_params = {
    "Photoperiod_h": (14, 16, 18, 20, 22),
    "LightIntensity_PPFD": (200, 300, 400, 500, 600),
    "DayTemp_C": (18, 20, 22, 24, 26),
    "NightTemp_C": (12, 14, 16, 18, 20),
    "Humidity_percent": (40, 50, 55, 60, 65),
    "CO2_ppm": (400, 500, 600, 700, 800)
}
for col in ccd_df.columns:
    extreme_low, low, center, high, extreme_high = decode_params[col]
    mapping = {-alpha: extreme_low, -1: low, 0: center, 1: high, alpha: extreme_high}
    ccd_df[col] = ccd_df[col].apply(lambda x: mapping.get(np.round(x, 3), center))

# Step 3: Simulate Yield (Deterministic)
def simulate_yield(photoperiod, light_intensity, day_temp, night_temp, humidity, co2):
    base = (photoperiod - 14) * 2.0 + (day_temp - 20) * 2.5 + (co2 - 400) * 0.05
    synergy = 0.01 * light_intensity * (humidity / 100)
    temp_diff_penalty = -abs(day_temp - night_temp - 6) * 1.5
    noise = rng.normal(0, 5)  # controlled randomness
    return 50 + base + synergy + temp_diff_penalty + noise

ccd_df["Yield"] = ccd_df.apply(
    lambda row: simulate_yield(
        row["Photoperiod_h"],
        row["LightIntensity_PPFD"],
        row["DayTemp_C"],
        row["NightTemp_C"],
        row["Humidity_percent"],
        row["CO2_ppm"]
    ),
    axis=1
)

# Step 4: Train Model
X = ccd_df.drop("Yield", axis=1)
y = ccd_df["Yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Test RMSE: {rmse:.2f}")
print(f"R² Score (Test): {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Step 5: Full Parameter Space Prediction
levels = {
    "Photoperiod_h": [14, 16, 18, 20, 22],
    "LightIntensity_PPFD": [200, 300, 400, 500, 600],
    "DayTemp_C": [18, 20, 22, 24, 26],
    "NightTemp_C": [12, 14, 16, 18, 20],
    "Humidity_percent": [40, 50, 55, 60, 65],
    "CO2_ppm": [400, 500, 600, 700, 800]
}
full_combinations = list(itertools.product(*levels.values()))
full_df = pd.DataFrame(full_combinations, columns=levels.keys())
full_df["Predicted_Yield"] = model.predict(full_df)

optimal = full_df.loc[full_df["Predicted_Yield"].idxmax()]
print("\nOptimal Parameter Combination:")
print(optimal)

# Step 6: Feature Importance
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP Summary Bar Plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_bar.svg'), format='svg')
plt.close()

# SHAP Beeswarm Plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_beeswarm.svg'), format='svg')
plt.close()

# Train vs Test Plot
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.scatter(y_train, train_pred, label='Train', alpha=0.7)
plt.scatter(y_test, test_pred, label='Test', alpha=0.7)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Train vs Test Predictions')
plt.legend()
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_vs_test.svg'), format='svg')
plt.show()

# Residual Plot
residuals = y_test - test_pred
plt.figure(figsize=(10,5))
sns.residplot(x=test_pred, y=residuals, lowess=True)
plt.xlabel('Predicted Yield')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.axhline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_plot.svg'), format='svg')
plt.show()

ccd_df.drop(columns="Yield").to_csv(os.path.join(output_dir, "ccd_combinations.csv"), index=False)

ccd_df.to_csv(os.path.join(output_dir, "synthetic_data_with_yield.csv"), index=False)

full_df.to_csv(os.path.join(output_dir, "full_parameter_space_predictions.csv"), index=False)
