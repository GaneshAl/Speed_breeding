# 🌱 Speed Breeding Yield Optimization (Synthetic Data + ML Modeling)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](#-requirements)
[![GitHub Stars](https://img.shields.io/github/stars/GaneshAl/Speed_breeding?style=social)](https://github.com/GaneshAl/Speed_breeding/stargazers)

This repository contains a Python workflow for **designing experiments**, **generating synthetic crop yield data**, and **training a machine learning model** to identify optimal environmental conditions for speed breeding.  

It uses **Central Composite Design (CCD)** to explore parameter space, simulates yield with a deterministic model plus noise, fits a **Random Forest Regressor**, evaluates performance, and provides **SHAP feature importance analysis** for model interpretability.

---

## 📌 Features
- **Experimental Design**: Uses `pyDOE2` Central Composite Design with 6 environmental factors:
  - Photoperiod (hours)
  - Light Intensity (PPFD)
  - Day Temperature (°C)
  - Night Temperature (°C)
  - Humidity (%)
  - CO₂ concentration (ppm)
- **Synthetic Yield Simulation**: Deterministic formula + random noise for realism
- **Machine Learning**: Train/test split, Random Forest Regressor
- **Evaluation Metrics**: RMSE, R², MAE
- **Optimal Condition Search**: Predicts yield for the entire parameter space
- **Model Interpretability**: SHAP summary plots and feature importance
- **Visualizations**:
  - Train vs Test predictions
  - Residual plot
  - SHAP bar and beeswarm plots

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

## 📂 Outputs (saved in the `output_dir` path you set in the script)
- `synthetic_data_with_yield.csv` — full experimental dataset
- `ccd_combinations.csv` — CCD combinations without yield
- `full_parameter_space_predictions.csv` — predictions for all parameter combinations
- `shap_summary_bar.svg` — SHAP feature importance (bar)
- `shap_beeswarm.svg` — SHAP beeswarm plot
- `train_vs_test.svg` — Train vs Test scatter plot
- `residual_plot.svg` — Residual analysis

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

**Optimal Parameter Combination**:
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
