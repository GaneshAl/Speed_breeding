# 🌱 Speed Breeding Optimization (DoE-ML)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](#-requirements)
[![GitHub Stars](https://img.shields.io/github/stars/GaneshAl/Speed_breeding?style=social)](https://github.com/GaneshAl/Speed_breeding/stargazers)

This repository provides a simulation + machine learning workflow for exploring optimal environmental conditions in speed breeding using synthetic data.

It integrates Central Composite Design (CCD) for experiment design, five synthetic yield models, Random Forest regression, and SHAP interpretability to study feature effects and interactions.

---

## 📌 Features
- **Experimental Design**:Central Composite Design (pyDOE2) with 6 environmental factors:
- Photoperiod (hours)
- Light Intensity (PPFD)
- Day Temperature (°C)
- Night Temperature (°C)
- Humidity (%)
- CO₂ concentration (ppm)
  
- **Synthetic Yield Simulation**: Five simulation functions with different structures:
- Model_A_linear
- Model_B_saturation_optimum
- Model_C_interactions_heterosc
- Model_D_threshold_plateau
- Model_E_periodic_nonseparable
  
- **Machine Learning**: Train/test split, Random Forest Regressor
- **Evaluation Metrics**: RMSE, R², MAE
- **Optimal Condition Search**: Predicts yield for the entire parameter space
- **Model Interpretability**: SHAP summary plots and feature importance
- **Visualizations**:
  - Train vs Test predictions
  - Residual plot
  - SHAP bar and beeswarm plots
  - Feature importance ranking

---

## 🛠 Requirements
Install the dependencies with:
```bash
pip install numpy pandas pyDOE2 scikit-learn shap matplotlib seaborn
```

---

## 🚀 How to Run
Clone the repository:
```bash
git clone https://github.com/GaneshAl/Speed_breeding.git
cd Speed_breeding
```

Run the script:
```bash
python main.py
```

---

## 📂 Outputs 
All results are saved to the OUTPUT_DIR you define in the script. For each model (A–E), the pipeline saves:
- *_ccd_data.csv — CCD dataset with simulated yields
- *_full_grid_predictions.csv — predictions for all parameter combinations
- *_feature_importances.csv — Random Forest feature importance
- *_train_vs_test.png — scatter plot of train/test predictions
- *_residuals.png — residual analysis
- *_shap_summary.png & *_shap_beeswarm.png — SHAP plots (if enabled)
---

## 📊 Example Output

**Feature Importance (SHAP-based)**:
```
      Feature   Importance
 Photoperiod_h      0.256
 DayTemp_C          0.211
 LightIntensity_PPFD 0.187
 ...
```

**Optimal Parameter Combination (example) **:
```
Photoperiod_h           20.0
LightIntensity_PPFD    600.0
DayTemp_C               26.0
NightTemp_C             18.0
Humidity_percent        60.0
CO2_ppm                800.0
Predicted_Yield         89.5
```

---

## 📜 License
This project is licensed under the MIT License — feel free to use and adapt.

---

## 🤝 Contributing
Pull requests and suggestions are welcome! For major changes, open an issue first to discuss.

---

## 💡 Notes
- The dataset is synthetically generated for demonstration purposes.
- The parameter ranges can be updated in the `decode_params` and `levels` dictionaries.
- To change the random seed, modify:
```python
rng = np.random.default_rng(seed=42)
```
