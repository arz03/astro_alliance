# Parikalp 2026 Hackathon - Astro Alliance
## Manned Exploration Shuttle "PARIKALP" Survival Challenge Context

This document provides a comprehensive, self-contained overview of the hackathon project context, dataset anomalies, mathematical solutions, and implementation guidelines. Coding agents should use this as their starting reference when beginning a session.

---

## 1. Core Problem Statement & Scenario
**The Crisis (Year 2084):**
The exploration shuttle *PARIKALP* is trapped in the gravitational orbit of an exoplanet. The engines must be fired to hit the exact **Escape Velocity ($v_e$)** to escape safely:
* **Too slow:** The shuttle crashes onto the surface.
* **Too fast:** The backup fuel tanks explode.

The escape velocity formula is:
$$v_e = \sqrt{\frac{2 G M}{R}}$$

**Where:**
* $G = 6.6743 \times 10^{-11} \text{ m}^3 \text{ kg}^{-1} \text{ s}^{-2}$ (Universal Gravitational Constant)
* $M$ = Planet Best Mass in kg = $\text{pl\_bmasse} \times 5.9722 \times 10^{24} \text{ kg}$
* $R$ = Planet Radius in meters = $\text{pl\_rade} \times 6.371 \times 10^6 \text{ m}$

**The Complication:**
Sensor streams are corrupted with heavy noise, missing values, and physical inconsistencies. Standard physics equations cannot be applied to raw data. The objective is to clean the dataset, resolve hidden unit inconsistencies, train machine learning models to reconstruct missing planetary features, predict the target variables (`pl_rade` and `pl_bmasse`), and accurately calculate the escape velocity.

---

## 2. Evaluation & Marking Scheme (Total: 100 Marks)
* **Round 1 (Online Submission):** Python Notebook (.ipynb) [70 Marks] + Project Report (PDF) [30 Marks].
* **Jupyter Notebook Evaluation Sections (70 Marks Total):**
  1. **Exploratory Data Analysis (EDA) [25 Marks]**
     * *1.1 Data Inspection (6 Marks):* Data types, descriptive statistics, missing value identification, and initial imputation.
     * *1.2 Statistical Inference (8 Marks):* Skewness assessment, normalization, outlier detection, noise analysis (e.g., `cosmic_noise_1`/`2`, `pl_orbper_noisy`, `pl_orbsmax_noisy`), scatter/correlation analysis, target variable distributions.
     * *1.3 Feature Selection, Dimensionality Reduction & Encoding (6 Marks):* Column dropping, target leakage checking, categorical encoding, feature selection, and PCA/dimensionality reduction.
     * *1.4 Visualization (5 Marks):* Visual identification of physics law violations and applying mathematical corrections.
  2. **Numerical Interpretation & Mathematical Analysis [14 Marks]**
     * *2.1 Feature Engineering (10 Marks):* Planetary, orbital, stellar, and photometric/spectral feature creation.
     * *2.2 Additional Features (4 Marks):* Novel engineered features, domain knowledge interactions, and statistical insights.
  3. **Radius and Mass Prediction [13 Marks]**
     * Multi-output regression models, cross-validation, hyperparameter tuning, model evaluation (MAE, RMSE, $R^2$), actual vs. predicted visualization, and final escape velocity calculation.
  4. **Model Interpretation [13 Marks]**
     * Bias-variance analysis, residual/error plots, comparison of at least 2 models.
  5. **Conclusion [5 Marks]**
     * Key findings, limitations, and future scope.

---

## 3. Data Dictionary Key Variables
* **Targets:**
  * `pl_rade`: Planet Radius [Earth Radii]
  * `pl_bmasse`: Planet Best Mass [Earth Masses]
* **Key Astronomical Constants:**
  * Earth Mean Radius ($1 \text{ R}_\oplus$): $6,371.0 \text{ km}$ ($6.3710 \times 10^6 \text{ m}$)
  * Earth Mass ($1 \text{ M}_\oplus$): $5,972.20 \text{ vottaarams}$ ($5.9722 \times 10^{24} \text{ kg}$) $\Rightarrow 1 \text{ vottaaram} = 10^{21} \text{ kg}$
  * Earth Escape Velocity: $11.186 \text{ km/s}$
  * Solar Radius ($1 \text{ R}_\odot$): $695,700 \text{ km}$ ($6.9570 \times 10^8 \text{ m}$)
  * Solar Mass ($1 \text{ M}_\odot$): $1,988,400 \text{ vottaarams}$ ($1.9884 \times 10^{30} \text{ kg}$)
* **Key Columns:**
  * `pl_orbper`: Orbital Period [Days]
  * `pl_orbsmax`: Orbit Semi-Major Axis [AU]
  * `pl_ratror`: Ratio of Planet Radius to Stellar Radius
  * `st_mass`: Stellar Mass [Solar Masses]
  * `st_rad`: Stellar Radius [Solar Radii]
  * `sy_dist`: System Distance [Parsecs]
  * `sy_plx`: System Parallax [milliarcseconds (mas)]
  * `orbital_size_score`: Engineered orbital score
  * `stellar_scale_score`: Engineered stellar score

---

## 4. Key Data Anomalies & Mathematical Corrections (Discovered)

### 4.1. Distance Unit Inconsistency
* **Issue:** Distance is logged in mixed units: Parsecs and Light-Years. Some values are also logged as negative numbers.
* **Physical Relation:** $D_{pc} = \frac{1000}{p_{mas}}$ (Distance in parsecs is the inverse of parallax in arcseconds).
* **Bifurcation Ratio:** Let $\text{ratio} = \frac{\text{abs}(sy\_dist)}{1000 / sy\_plx}$.
  * If $\text{ratio} \approx 1.0$, the unit is **Parsecs**.
  * If $\text{ratio} \approx 3.26$, the unit is **Light-Years** (since $1 \text{ pc} \approx 3.26156 \text{ ly}$).
* **Correction Formula:**
  ```python
  import numpy as np
  
  # Take absolute value to correct signs
  df['sy_dist_clean'] = df['sy_dist'].abs()
  
  # Check against physical distance
  expected_dist_pc = 1000.0 / df['sy_plx']
  ratio = df['sy_dist_clean'] / expected_dist_pc
  
  # If ratio > 2.0, the value is in Light-Years. Convert to Parsecs.
  is_ly = (ratio > 2.0) & (df['sy_plx'] > 0)
  df.loc[is_ly, 'sy_dist_clean'] = df.loc[is_ly, 'sy_dist_clean'] / 3.26156
  ```

### 4.2. Geometric Size Violation (Planet Radius Calibration)
* **Issue:** Many `pl_rade` values exceed $100 \text{ R}_\oplus$, and some are around $7,500$ to $12,000 \text{ R}_\oplus$, which is physically impossible for a planet (envelope radius boundary limit is $\sim 27.35$ Jupiter Radii $\approx 306.57 \text{ R}_\oplus$).
* **Physical Relation (Transit Signal Depth Calibration):**
  $$pl\_ratror = \frac{pl\_rade \times 6.371 \times 10^6}{st\_rad \times 6.957 \times 10^8} = \frac{pl\_rade}{st\_rad} \times 0.00915768$$
* **Correction Formula:**
  For transit-detected planets that violate the physical boundary limits (e.g. $pl\_rade > 100$), reconstruct the true planet radius from `pl_ratror` and `st_rad`:
  $$pl\_rade_{true} = pl\_ratror \times st\_rad \times \frac{695700.0}{6371.0} \approx pl\_ratror \times st\_rad \times 109.197928$$
  ```python
  # Identify violations
  is_violation = (df['pl_rade'] > 100) & (df['pl_ratror'] > 0) & (df['st_rad'] > 0)
  
  # Reconstruct
  df.loc[is_violation, 'pl_rade'] = df.loc[is_violation, 'pl_ratror'] * df.loc[is_violation, 'st_rad'] * (695700.0 / 6371.0)
  ```

### 4.3. Imputed Placeholders (Artificial Peaks)
* **Issue:** Approximately 30% of the `pl_rade` values (12,053 rows) are exactly `2.302000`. This is a default placeholder imputation value in the raw dataset. Modeling pipelines should note this artificial peak as it can bias predictions.

---

## 5. Engineered Scores (Formulas Decoded)
* **Orbital Size Score:** Fully redundant and linearly related to the orbital period (`pl_orbper`).
  $$\text{orbital\_size\_score} \approx 0.70 \times pl\_orbper + 0.1354$$
* **Stellar Scale Score:** Reconstructed stellar scale metric using primary parameters:
  $$\text{stellar\_scale\_score} \approx 0.814 \times st\_mass + 1.043 \times st\_rad + 0.142$$

---

## 6. Modeling Pipeline Requirements

### 6.1. Avoiding Target Leakage
* **Caution:** Do not use `pl_ratror`, `pl_ratrorerr1`, or `pl_ratrorerr2` directly as training features to predict `pl_rade`, since `pl_ratror` is a direct mathematical derivative of the target `pl_rade`. Doing so violates target leakage constraints and results in penalties. Drop them before modeling.

### 6.2. Multi-Output Regression Pipeline
1. **Preprocessing:**
   * Apply distance unit correction (`sy_dist`).
   * Apply radius boundary correction (`pl_rade`).
   * Encode categorical features (`discoverymethod`, `disc_facility`, `soltype`) using Target Encoding, One-Hot Encoding, or Ordinal Encoding.
   * Drop unique identifiers/timestamps (`rowid`, `pl_name`, `hostname`, `pl_letter`, `rowupdate`, `pl_pubdate`, `releasedate`, `disc_pubdate`).
   * Drop target leakage columns.
   * Split the dataset into train and test sets.
   * Use a robust imputer (e.g., `IterativeImputer` or `KNNImputer`) to fill missing features.
2. **Modeling:**
   * Train models to predict both `pl_rade` and `pl_bmasse`.
   * Baseline Model: Random Forest or Multi-Output Ridge Regression.
   * Advanced Model: Multi-output wrappers on Gradient Boosting regressors (e.g., `XGBRegressor`, `LGBMRegressor`, or `CatBoostRegressor`).
3. **Evaluation:**
   * Evaluate using MAE, RMSE, and $R^2$.
   * Plot residual errors and actual vs. predicted values.
4. **Escape Velocity Output:**
   * Calculate escape velocity on predicted outputs using SI conversion factors.
   * Validate that computed velocities align with the physical constraints (avoiding fuel tank explosion or surface crash).
