# PARIKALP 2026 Final Report Context

## Purpose
Use this document as the source brief for the final 30-mark report PDF. It should be concise, rubric-aligned, and based on the actual notebook results in `notebooks/data_inspection.ipynb`.

## Problem Framing
Astro Alliance's PARIKALP shuttle is trapped near an exoplanet. The goal is to predict planetary mass and radius from noisy observational data and then compute escape velocity:

$$v_e = \sqrt{\frac{2GM}{R}}$$

The report should explain how the notebook cleans corrupted measurements, avoids leakage, and builds a defensible predictive pipeline.

## Recommended Report Structure

### 1. Introduction
Summarize the challenge, the target variables, and the scientific objective.

### 2. Data Understanding and EDA
Cover the main anomalies and what they imply:
- data types, missingness, descriptive statistics
- skewness and outliers
- target distributions and difficulty
- noise columns and their lack of signal
- feature redundancy and multicollinearity
- physics-law violations visible in the raw data

### 3. Data Cleaning and Feature Engineering
Explain the corrections and why they are valid:
- distance unit correction for `sy_dist`
- radius reconstruction for invalid `pl_rade` rows
- placeholder radius handling for `2.302`
- leakage column removal
- categorical encoding
- engineered features such as log transforms, regime labels, color indices, stellar proxies, and interaction features

### 4. Modeling Strategy
Describe the leakage-safe modeling path:
- base HistGradientBoostingRegressor for mass prediction
- Random Forest baseline with imputation
- GridSearchCV-tuned HGB as the best single-model performer
- multi-output regression for joint radius/mass prediction
- two-stage pipeline: radius imputer followed by mass predictor
- why PCA stayed exploratory-only and was not used in the final mass model

### 5. Evaluation and Interpretation
Include the key metrics and comparisons:
- base HGB: R² 0.8839, MAE 0.2086, RMSE 0.3054
- tuned HGB: R² 0.9193, MAE 0.1675, RMSE 0.2546
- Random Forest: R² 0.8780, MAE 0.2091, RMSE 0.3130
- two-stage pipeline: R² 0.9228, MAE 0.1589, RMSE 0.2490
- cross-validation stability: HGB CV R² 0.8799 ± 0.0056

Explain the residual behavior, model comparison, feature importance, and regime-wise performance.

### 6. Escape Velocity Output
Show how the predicted mass and radius were converted into escape velocity and used for the final physical interpretation.

### 7. Conclusion
State the main findings, limitations, and future scope.

## Key Messages to Preserve
- Leakage control matters more than squeezing a small score gain from target-derived features.
- The tuned HGB model is the best single-model performer.
- The two-stage pipeline is the strongest end-to-end solution for rows with missing radius information.
- PCA should be described as exploratory-only unless it is rebuilt as a train-only preprocessing step.

## Suggested Figures for the Report
- data quality / anomaly summary table
- distance unit distribution plot
- radius placeholder histogram
- correlation / redundancy heatmap
- model comparison chart
- actual vs predicted plot
- residual plot
- feature importance chart
- regime-wise performance summary
- escape velocity histogram

## Writing Style Guidance
Keep the final report technical but concise. Emphasize scientific justification, preprocessing defensibility, and measurable model performance. Avoid claiming improvements that were not validated in the notebook.