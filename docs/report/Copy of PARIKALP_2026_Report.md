
**PARIKALP 2026**  
**ASTRO ALLIANCE \- DATA ANALYTICS CHALLENGE**

*Project Report*

*Reconstructing Corrupted Exoplanet Sensor Data to Predict Planetary Mass, Radius and Escape Velocity for the PARIKALP Shuttle*

![][image1]  
Section 1: EDA & Physics Validation | Section 2: Feature Engineering | Section 3-4: Modeling | Escape Velocity: Final Output

| Team Name | Team Vortex |
| :---- | :---- |
| **Event** | PARIKALP 2026 Data Analytics Round 1 |
| **Deliverables** | Solution Notebook (.ipynb) and this Project Report (.docx / PDF) |
| **Submission Section** | Round 1 \- Online Submission (Report: 30 Marks) |

# **Report Index**

**1\. Problem Understanding and Approach**

   1.1 The 2084 Crisis Scenario

   1.2 Objective and Constraints

   1.3 Our Approach: Pipeline Overview

**2\. Exploratory Data Analysis (EDA)** 

   2.1 Data Inspection and Overview

   2.2 Physics-Based Anomaly Detection and Correction

   2.3 Data Cleaning Pipeline

   2.4 Feature Selection, Dimensionality Reduction and Encoding 

   2.5 Feature Engineering and Additional Features

   2.6 Modeling Strategy: Leakage-Safe Two-Stage Pipeline

**3\. Predictions Results and Analysis**

   3.1 Model Performance Comparison

   3.2 Actual vs Predicted and Residual Analysis

   3.3 Regime-Wise Performance

   3.4 Feature Importance Analysis

   3.5 Bias-Variance and Cross-Validation Analysis

   3.6 Escape Velocity Calculation and Mission Risk

   3.7 Key Findings

   3.8 Limitations

   3.9 Future Scope

**4\. Conclusion**

# **1\. Problem Understanding and Approach**

## **1.1 The 2084 Crisis Scenario**

In the mission year 2084, an unexpected gravitational anomaly destroys the main engines of the human exploration shuttle PARIKALP, trapping it in the gravitational orbit of a recently found exoplanet. There are no second chances, and the crew only has enough fuel for one emergency launch attempt. The team must ignite the engines to reach the precise escape velocity in order to escape the planet's gravity.

The escape velocity is determined by the classical formula *v_e = sqrt(2GM/R)*, where G is the universal gravitational constant, M is the planet's mass, and R is its radius. There are two catastrophic failure possibilities for the mission: launching too slowly leads the shuttle to fall back and crash into the surface, while launching too quickly puts the already overworked backup fuel tanks under lethal stress and causes them to explode. 

Cosmic storms caused damage to the onboard sensor network, which provides mass and radius. The accessible historical dataset of 39,913 previously reported exoplanets is corrupted by significant noise, missing values, and physical abnormalities, including a serious unit-logging issue in one of the distance metrics. It is impossible to apply traditional physics equations to this data. The mission needs a machine learning pipeline that will accurately predict the planet's mass and radius, recreate the missing planetary characteristics, and clean up the distorted data in order to establish a reliable escape velocity..

## **1.2 Objective and Constraints**

Developing a dependable Python-based predictive pipeline that cleans the noisy sensor data, missing planetary features, estimates the two target variables, *pl\_rade* (Planet Radius, Earth radii) and *pl\_bmasse* (Planet Best Mass, Earth masses), and uses these predictions to calculate the escape velocity and consequent mission risk is our responsibility. Three limitations guided every design decision made for this project:  

* **No target leakage**: *pl\_ratror* and its uncertainty columns should never be used as a radius predictor because they are direct mathematical derivatives of *pl\_rade*. Each feature set was carefully reviewed in order to remove this and any other source of leakage.

* **Physical validity**: Each anomaly correction must be based on a true physical connection (parallax-distance geometry, signal depth geometry) rather than a blind statistical method like capping or naive outlier elimination.

* **Reproducibility**: Since the evaluation specifically favors documented rationale over black-box findings, every cleaning, imputation, and modeling decision required a clear, code-visible justification.

## 

## **1.3 Our Approach: Pipeline Overview**

From start to finish, the whole technique adheres to the official assessment structure: physics validation and exploratory data analysis, followed by feature engineering, a two-stage radius and mass prediction pipeline, and finally the computation of escape velocity and mission risk. 

The flowchart below summarizes the flow used in the attached solution notebook.

![][image2]  
Section 1: EDA & Physics Validation | Section 2: Feature Engineering | Section 3-4: Modeling | Escape Velocity: Final Output

*Figure 1.1 \- End-to-end analytics pipeline, from raw corrupted sensor data to the final mission risk classification.*

# **2\. Exploratory Data Analysis (EDA)** 

## **2.1 Data Inspection and Overview**

The dataset contains 39,913 rows and 96 columns describing planetary, orbital, stellar, system and photometric parameters, together with identifier and administrative metadata. Of these, 86 columns are numeric and 10 are categorical or identifier-type text columns. Missingness is concentrated in the uncertainty columns and in a small number of physical parameters, as summarised below.

| Column | Missing Values | Missing % |
| ----- | :---: | :---: |
| pl\_insol | 22,580 | 56.6% |
| st\_dens | 17,343 | 43.5% |
| pl\_orbsmax | 17,643 | 44.2% |
| st\_mass | 6,378 | 16.0% |
| pl\_orbper | 3,404 | 8.5% |
| st\_rad | 3,424 | 8.6% |
| sy\_dist | 906 | 2.3% |
| pl\_bmasse (target) | 231 | 0.6% |

Missing values are handled with a hybrid strategy. Columns that are ultimately dropped need no treatment. Categorical columns are filled with an explicit Unknown label before ordinal encoding, so missingness itself becomes a learnable signal. Numeric predictors are left as native NaN and handled internally by HistGradientBoostingRegressor's missingness-aware tree splits, which is more principled here than blanket mean or median imputation, particularly given the artificial placeholder spike described in Section 2.2.

## **2.2 Physics-Based Anomaly Detection and Correction**

Three independent physics-based checks were used to identify and correct data corruption, each visually confirmed before any correction was applied.

### **2.2.1 Distance Unit Inconsistency**

The physical relationship between distance in parsecs and parallax in milliarcseconds is *D = 1000 / p*. Analyzing the ratio of the logged distance to this predicted value revealed two separate clusters: one around 1.0 (correctly documented in parsecs) and another near 3.26 (incorrectly logged in light-years), as well as 1,005 rows with a negative sign mistake. 

![][image3]

*Figure 2.1 \- Distribution of the ratio abs(sy\_dist) / (1000 / sy\_plx), showing the parsec and light-year peaks.*

| Diagnostic | Row Count |
| ----- | :---: |
| Correctly logged in parsecs (ratio 0.5 \- 1.5) | 24,979 |
| Mistakenly logged in light-years (ratio 2.0 \- 5.0) | 10,963 |
| Negative sign error | 1,005 |

Correction rule: Take the absolute value of *sy\_dist*, compute the ratio against the distance determined by parallax, and divide the result by 3.26156 to convert light-years to parsecs if the ratio exceeds 2.0. This rule was further validated using a physical distance barrier of 8,500 parsecs, the realistic line-of-sight limit for the Galactic Bulge, which contains the majority of microlensing detections. Prior to correction, 210 rows exceeded this limit; after repair, no rows did.

### **2.2.2 Planetary Geometric Size Violation**

A planet's radius surpassed 100 Earth radii in 1,212 rows, with the greatest raw result reaching over 5 trillion Earth radii, which is physically impossible for a planet. The real radius may be calculated using independently observed parameters and the transit signal depth ratio:

*pl_ratror = (pl_rade x 6371.0) / (st_rad x 695700.0)*

![][image4]

*Figure 2.2 \- Raw pl\_rade distribution showing the physics-violating tail (left) versus the reconstructed, physically plausible radii for the affected rows (right).*

Of the 1,212 violating rows, 664 had both *pl\_ratror* and *st\_rad* available and were reconstructed using 

*pl_rade_true = pl_ratror x st_rad x (695700.0 / 6371.0)*

the remaining 569 rows, which included a second, less noticeable placeholder cluster of 333 rows positioned at precisely 7,518.508389 Earth radii and a lengthy tail up to around 34,800 Earth radii, could not be reconstructed in this manner because they lacked *pl\_ratror* or *st\_rad*. The exact 2.302 placeholder (Section 2.2.3) and any value that remained greater than 100 Earth radii after reconstruction were set to missing and excluded from direct radius-target training, then completed through the staged radius model instead of being left to quietly distort the training target. 

### **2.2.3 Artificial Placeholder Spike**

The single most common value in *pl\_rade* is precisely 2.302000, appearing in 12,053 rows. Using a tight numeric tolerance around 2.302 flags 12,068 rows before the cleaning pass and 12,070 rows after geometric reconstruction, or about 30.2% of the dataset. Any radius model would be biased toward an artificial mode if it were trained directly on this default imputation placeholder instead of an actual physical measurement. 

![][image5]

*Figure 2.3 \- Distribution of pl\_rade zoomed to 0-10 Earth radii, showing the artificial spike at 2.302.*

### **2.2.4 Secondary Consistency Checks**

Stellar density was cross-checked against the physical relation:

*rho_star = 1.4095 x st_mass / st_rad^3.*

Density is a noisier secondary signal, as evidenced by the fact that 25.0% of rows diverged by more than a factor of two, despite the median ratio of logged to computed density being near to 1.0. *st\_mass* and *st\_rad* were kept as the main, more dependable predictors instead of re-deriving density. The supplied engineered scores were also deciphered: *stellar\_scale\_score* is a strong linear combination of *st\_mass* and *st\_rad* (R-squared of 0.996) and *orbital\_size\_score* is a nearly perfect linear function of *pl\_orbper* alone (R-squared of 1.000), indicating that both are redundant with current primary features and were only retained as inexpensive diagnostic fallbacks. 

## **2.3 Data Cleaning Pipeline**

Before any feature engineering or modeling, the aforementioned adjustments were combined into a single ordered cleaning pipeline and applied to the working dataset once.

| Step | Action | Rows / Columns Affected |
| ----- | :---: | :---: |
| 1 | Reconstruct radius from transit ratio for geometric violations | 664 rows corrected |
| 2 | Neutralise the exact 2.302 placeholder to missing | 12,070 rows |
| 2b | Neutralise any value still above 100 Earth radii after step 1 | 569 rows |
| 3 | Correct light-year to parsec unit errors and take absolute value | 11,346 rows |
| 4 | Drop identifier, noise and target-leakage columns | 15 columns |
| 5 | Ordinally encode categorical features | 3 columns |

The columns removed in step 4 fall into three groups: pure identifiers and timestamps that carry no physical signal (*rowid, pl\_name, hostname, pl\_letter, rowupdate, pl\_pubdate, releasedate, disc\_pubdate*); synthetic noise columns confirmed to carry no independent signal (*cosmic\_noise\_1, cosmic\_noise\_2, pl\_orbper\_noisy, pl\_orbsmax\_noisy*, with correlations to the mass target below 0.006 in magnitude); and the target-leakage columns *pl\_ratror*, *pl\_ratrorerr1* and *pl\_ratrorerr2*, which are direct mathematical derivatives of *pl\_rade*. After cleaning, the dataset shape is 39,913 rows by 81 columns, with 12,639 rows carrying a missing *pl\_rade* (the two neutralised placeholder families combined) and 231 rows missing *pl\_bmasse*.

## **2.4 Feature Selection, Dimensionality Reduction and Encoding** 

The three categorical columns, *discoverymethod*, *disc\_facility* and *soltype*, were ordinally encoded rather than one-hot or target encoded. This choice was made deliberately: the modeling algorithms are tree-based (HistGradientBoostingRegressor and Random Forest), which split on ordinal-encoded categories natively without needing a linear or ordered relationship, so ordinal encoding avoids the dimensionality explosion that one-hot encoding would cause on *disc\_facility*'s 74 categories, and it avoids the leakage risk inherent in target encoding, which implicitly uses the target variable during the encoding step itself.

The eleven photometric magnitude columns are highly collinear. Principal Component Analysis on this block, fit purely for diagnostic purposes on the full dataset, shows the first principal component alone explains 86.95% of the variance, with the first two components together explaining 95.32%.

![][image6]

*Figure 2.4 \- Scree plot of the photometric magnitude PCA (left) and the systems projected into the first two principal components, coloured by mass (right).*

This diagnostic PCA is explicitly excluded from the modeling feature set. *Mag\_pca\_1* and *mag\_pca\_2* are never given to any prediction model since using PCA on the entire dataset prior to the train and test split would leak information about the test set into the training components. Instead, two derived color indices, B minus V and J minus K, are retained as raw features along with a limited and diverse subset of magnitude bands. .

## **2.5 Feature Engineering and Additional Features**

Seventeen physics-informed features were engineered across four categories, bringing the total dataset to 98 columns before feature selection. These features encode known astrophysical relationships explicitly, which both speeds up learning for the tree-based models and makes the partial dependence and importance analyses in Section 3 more interpretable.

| Category | Engineered Features | Basis |
| ----- | :---: | :---: |
| Planetary / Orbital | *log\_orbper, log\_orbsmax, log\_insol, log\_eqt, log\_rade, radius\_cubed, planet\_regime, mass\_density\_proxy* | Log-transforms of heavily skewed quantities; radius-cubed and regime bucket for density behaviour |
| Stellar | *st\_luminosity, kepler\_proxy, log\_st\_dens* | **Stefan-Boltzmann law; Kepler's third law** |
| Photometric | *color\_BV, color\_JK* | Standard astronomical colour indices |
| Additional / Novel (Section 2.2 marks) | *has\_rv, total\_spectra, system\_multiplicity\_index, transit\_depth\_proxy* | Follow-up effort and detection-context signals |

A critical leakage boundary was enforced here as well: *radius\_cubed*, *planet\_regime*, *mass\_density\_proxy*, *log\_rade* and *transit\_depth\_proxy* are all direct functions of *pl\_rade*. These are excluded from the base feature set used to predict mass, and are only reintroduced downstream through a leakage-safe predicted radius, as described in Section 2.6.

## **2.6 Modeling Strategy: Leakage-Safe Two-Stage Pipeline**

The competition explicitly calls for multi-output regression. An exploratory joint multi-output HistGradientBoostingRegressor was trained for rubric coverage, but it has a practical limitation: it can only be trained and evaluated on the subset of rows with a real, non-placeholder radius, which excludes a large share of the dataset. The final submission path instead uses a leakage-safe two-stage pipeline, illustrated below.

![][image7]

*Figure 2.5 \- The two-stage architecture: Stage 1 predicts radius from leakage-safe features using out-of-fold predictions on the training set; Stage 2 predicts mass using the predicted radius as an additional feature.*

Ground-truth *pl\_rade* is never used to predict mass; the mass model only ever sees a leakage-safe, predicted radius.

In Stage 1, a dedicated radius model is trained on the leakage-safe feature set using a log1p target transform to handle the right-skewed radius distribution. Its predictions on the training set are generated out-of-fold, using five-fold *cross\_val\_predict*, so that no row ever sees a radius prediction derived from a model that was trained on that same row. In Stage 2, these out-of-fold radius predictions, together with derived features such as *log-radius*, *radius-cubed*, *planet regime* and a *transit-depth proxy*, are added back into the feature set, and a final HistGradientBoostingRegressor is trained to predict *log10(pl\_bmasse)*. At no point does the mass model see the ground-truth radius; it only ever sees a leakage-safe, model-generated estimate. This final pipeline, referred to as *stage2\_mass\_model*, is used consistently for every downstream evaluation in Section 3, including residual analysis, regime-wise metrics, feature importance and the escape velocity calculation.

# **3\. Predictions, Results and Analysis**

## **3.1 Model Performance Comparison**

Four modeling stages were compared on the held-out test set (7,937 rows, log10 mass scale unless noted). Cross-validation and hyperparameter tuning were carried out with GridSearchCV across 16 parameter combinations on 3 folds, selecting *learning\_rate* \= 0.1, *max\_depth* \= 10, *max\_iter* \= 500 and *min\_samples\_leaf* \= 10 as the best configuration, with a best cross-validated R-squared of 0.9035.

| Model | R-squared | MAE | RMSE |
| ----- | :---: | :---: | :---: |
| HistGradientBoosting (default params) | 0.8839 | 0.2086 | 0.3054 |
| Random Forest (baseline) | 0.8780 | 0.2091 | 0.3130 |
| HistGradientBoosting (tuned, GridSearchCV) | 0.9193 | 0.1675 | 0.2546 |
| Two-Stage Pipeline (final submission model) | 0.9228 | 0.1589 | 0.2490 |

![][image8]

*Figure 3.1 \- Model comparison across R-squared, MAE and RMSE for all four candidate mass-prediction models.*

On its own held-out test rows, the customized Stage 1 radius model has an R2 of 0.8100 and an MAE of 0.6640 Earth radii. With a HistGradientBoosting score of 0.8799  0.0056 R2 across folds and a Random Forest score of 0.8789   0.0063, five-fold cross-validation confirms that the mass model is stable and not overfit to one split. 

## 

## 

## 

## 

## 

## 

## 

## 

## **3.2 Actual vs Predicted and Residual Analysis**

![][image9]

*Figure 3.2 \- Actual versus predicted log10(mass) for the final two-stage pipeline (left) and the Random Forest baseline (right).*

The diagonal line of perfect prediction is where both models cluster tightly together. For high-mass gas giants, the gap widens, indicating that the mass-radius relationship becomes corrupted in that state: if the planet is huge enough that its composition, not just size, drives mass, a small mistake in anticipated radius equates to a large error in inferred mass. 

![][image10]

*Figure 3.3 \- Residuals versus predicted value (left), residual distribution (centre), and residual spread split by whether the row originally had a real or placeholder radius (right).*

There is no consistent bias in either direction, as shown by the residual mean of \-0.0009 and median of \-0.0013, both of which are around zero. The slightly heavier right tail characteristic of mass prediction in log space is reflected in the residual standard deviation of 0.2490 with a moderate positive skew of 0.5162. As expected given the limited information available for those rows, rows with a placeholder radius (now neutralized to absent) have a noticeably greater residual spread than rows with a true measured radius. 

## 

## **3.3 Regime-Wise Performance**

A helpful context for interpreting the escape velocity results in Section 3.6 is provided by splitting the test set according to the projected planetary regime, which reveals distinct error profiles across planet types. 

| Regime | Test Rows | R-squared | MAE | RMSE |
| ----- | :---: | :---: | :---: | :---: |
| Rocky (radius below 1.5 Earth radii) | 1,072 | 0.8283 | 0.1427 | 0.2022 |
| Super-Earth (1.5 to 4.0 Earth radii) | 4,955 | 0.7853 | 0.1346 | 0.2091 |
| Sub-Giant (4.0 to 10.0 Earth radii) | 872 | 0.8577 | 0.2357 | 0.3297 |
| Gas Giant (10.0 Earth radii or larger) | 1,038 | 0.7461 | 0.2270 | 0.3629 |

![][image11]

*Figure 3.4 \- Partial dependence of predicted mass on predicted radius, stellar radius and stellar mass, all monotonically increasing as expected from the underlying physics.*

Super-Earths, the most numerous class in the dataset, show the tightest mean absolute error. Gas giants show both the lowest R-squared and the highest error, consistent with their more variable core-to-envelope composition at a fixed radius. The partial dependence curves confirm the model has learned a physically sensible, monotonically increasing relationship between mass and each predicted radius, stellar radius and stellar mass.

## **3.4 Feature Importance Analysis**

Permutation importance was computed directly on the final two-stage pipeline model and its corresponding feature set, so that the ranking below reflects exactly what drives the submitted predictions rather than an earlier intermediate model.

![][image12]

*Figure 3.5 \- Top 15 features by permutation importance for the final mass-prediction pipeline.*

| Rank | Feature | Importance (mean drop in R-squared) |
| ----- | :---: | :---: |
| 1 | *pl\_rade\_pred* (predicted radius) | 0.6260 |
| 2 | *pl\_orbper* (orbital period) | 0.0531 |
| 3 | *disc\_year* (discovery year) | 0.0459 |
| 4 | *discoverymethod* | 0.0404 |
| 5 | *transit\_depth\_proxy\_pred* | 0.0284 |
| 6 | *sy\_pnum* (planets in system) | 0.0258 |
| 7 | *color\_JK* | 0.0232 |
| 8 | *glat* (galactic latitude) | 0.0203 |
| 9 | *sy\_pmdec* (proper motion) | 0.0200 |
| 10 | *mass\_density\_proxy\_pred* | 0.0200 |

The predicted radius dominates all other features by an order of magnitude, exactly as physical intuition would predict from the mass-radius relation. Beyond that, the orbital period and several detection-context features (discovery year, discovery method, number of planets in the system) contribute meaningfully. This is worth stating plainly: part of the model's remaining signal comes from proxies for which survey found the planet, rather than from pure physics. Different surveys, radial velocity, transit, microlensing, preferentially detect different planet populations, so these columns partly encode selection effects rather than causal physical relationships. This is treated as an honest limitation rather than hidden, and is discussed further in Section 3.8.

## 

## **3.5 Bias-Variance and Cross-Validation Analysis**

![][image13]

*Figure 3.6 \- Learning curve for the tuned HistGradientBoosting model, showing training and validation R-squared as training set size increases.*

The validation R2 converging above 0.87 shows a slight bias, indicating that the model architecture is expressive enough to reflect the underlying mass-radius-orbit relationships without adding complexity. This exhibits minimal variance, indicating that the model generalizes well rather than overfitting the training data.

## **3.6 Escape Velocity Calculation and Mission Risk**

The escape velocity is estimated as Ve \=2GMR, where G \= 6.6743e-11,  M \= plbmass 5.9722e24 kg, and R \= plrade 6.371e6 m, based on the predicted radius and mass for each test-set exoplanet. To ensure that the unit conversions are correct, this formula approximates 11.186 km/s to three decimal places when applied to exactly one Earth mass and one Earth radius. 

With a mean of 31.07 km/s and a median of 18.85 km/s, the estimated escape velocity for the 7,937 test-set planets varies from 0.76 to 535.67 km/s. An escape velocity exceeding 200 km/s is produced by a small subset of 78 rows (0.98%), which is astrophysically unlikely for this planet population and suggests compounded Stage 1 to Stage 2 prediction error rather than a true physical event. Instead of being silently included, these rows are reported individually and are not included in the headline mission-risk percentages below.

![][image14]

*Figure 3.7 \- Distribution of predicted escape velocity across the test set, with the Earth-calibrated engine setting and safe launch window marked.*

The mission physics specify a set, Earth-calibrated speed of 11.186 km/s for the shuttle's engines. If a certain exoplanet requires a higher escape velocity than this preset value provides, the shuttle is underpowered in comparison to what is needed and falls back to the surface. If the required escape velocity is less than the preset figure, the shuttle exceeds its limitations and the added strain destroys the fuel tanks. The tolerance range around the Earth baseline of plus or minus 2.0 km/s is known as the safe launch window. 

| Risk Zone | Condition | Planet Count | Share |
| ----- | :---: | :---: | :---: |
| Zone A: Surface Crash Risk | Required *v\_e* above 13.19 km/s (engines under-powered) | 7,025 | 89.4% |
| Zone B: Safe Escape Window | Required *v\_e* between 9.19 and 13.19 km/s | 659 | 8.4% |
| Zone C: Fuel Tank Explosion Risk | Required *v\_e* below 9.19 km/s (engines over-powered) | 175 | 2.2% |

A fixed Earth-calibrated engine setting is consistently underpowered for the majority of the 7,859 planets in the physically plausible range because the cataloged exoplanet population skews toward higher mass and tighter orbits than Earth. This is evident directly from the dataset. An accurate, planet-specific mass and radius prediction is mission-critical rather than a nice-to-have because only a small range of planets fall within the safe launch window. In the vast majority of cases examined here, assuming Earth-like values for an unknown exoplanet would be incorrect.

## **3.7 Key Findings**

* The resulting two-stage mass prediction pipeline surpasses both the Random Forest baseline (R2 0.8780) and the modified single-stage model (R2  of 0.9193) on the held-out test set, with R2 of 0.9228 and MAE of 0.1589 in log10 space.

* The dedicated radius model removes target leakage by design, resulting in anR2 of 0.8100 and an MAE of 0.66 Earth radii. Its out-of-fold forecasts benefit the mass model while never exposing the ground-truth radius.

* Three different physics violations were found and fixed: a mixed parsec and light-year distance unit error, a geometric radius violation that could be reconstructed from the transit signal depth ratio, and two independent false placeholder value spikes in the radius target.

* Permutation significance suggests a secondary contribution from detection-survey context characteristics, while verifying that anticipated radius is by far the dominant driver of expected mass, consistent with the accepted astrophysical mass-radius relationship.

* The Earth escape velocity sanity check reproduces 11.186 km/s exactly, validating the physics implementation before it is applied to predicted, noisy planetary parameters.

* Applying the pipeline to the test-set exoplanet population shows 89.4% of physically plausible planets would place the PARIKALP shuttle in surface crash risk under a fixed Earth-calibrated engine setting, underscoring the mission necessity of planet-specific prediction.

## **3.8 Limitations**

* Rows that originally had a placeholder or missing radius still show a wider residual spread in the final mass predictions than rows with a genuine measured radius, reflecting the reduced information available for those rows even after Stage 1 reconstruction.

* Gas giant mass prediction carries higher variance than other regimes, because mass at a fixed radius becomes increasingly dependent on internal composition rather than size alone in that regime.

* Part of the model's predictive signal is attributable to detection-method and survey-context features, which partly encode which planets a given technique is capable of finding rather than a purely physical relationship, and should be interpreted with that caveat.

* 78 test rows (0.98%) produced escape velocities above 200 km/s due to compounded error across the two prediction stages; these were excluded from the headline mission-risk percentages rather than allowed to distort them.

## **3.9 Future Scope**

* Bayesian mass-radius relation frameworks (in the style of Forecaster or Spright) to obtain calibrated uncertainty intervals around each escape velocity estimate rather than a single point estimate.

* An ensemble of regime-specific regressors, trained separately for rocky, super-Earth, sub-giant and gas-giant populations, to address the accuracy gap observed in Section 3.3.

* To further clarify composition-driven mass variation at a constant radius, models of star age and metallicity evolution are incorporated.

*  Investigating deep learning techniques directly on light-curve data for planets that lack a pre-derived radius ratio and just have transit photometry. 

# **4\. Conclusion**

This project set out to turn a corrupted, 96-column exoplanet archive into a reliable basis for one irreversible mission decision. The starting dataset of 39,913 rows carried three separate physics violations: a mixed parsec and light-year distance error, a geometric radius error large enough to put some rows at over 5 trillion Earth radii, and two distinct artificial placeholder spikes standing in for missing measurements. None of these were treated as generic outliers to be capped or dropped. Each one was traced back to a specific, verifiable physical relationship, the parallax-distance formula for the unit error, the transit signal depth ratio for the radius error, and cross-referenced value counts for the placeholders, and corrected on that basis. That discipline mattered more than it might sound: 664 rows were successfully reconstructed through the transit-ratio formula alone, and a further 569 rows that could not be reconstructed were neutralised rather than left in the training data to quietly distort the radius model.

That same discipline carried through into modeling. Every feature used to predict mass was checked against the target it was meant to predict, so radius-derived quantities such as radius cubed and the transit-depth proxy never leak directly into the mass model. Instead, they come back in only through a leakage-safe, out-of-fold predicted radius, generated by a separate Stage 1 model and never touched by ground truth during training. The result is a two-stage pipeline that predicts planetary mass with an R-squared of 0.9228 and reconstructs planetary radius with an R-squared of 0.8100 on held-out data. The mass model also holds up under five-fold cross-validation rather than depending on a lucky split.

The escape velocity calculation built on top of these predictions passes its Earth-baseline sanity check exactly, reproducing 11.186 km/s to three decimal places before it is ever applied to a real, noisy prediction. Applied to the historical test population, the analysis shows that a fixed, Earth-calibrated engine setting would put 89.4% of physically plausible catalogued exoplanets in the crash-risk zone, with only 8.4% falling inside the safe launch window and 2.2% at risk of an explosion. That imbalance is not a modeling artifact; but it reflects a real property of the dataset, since catalogued exoplanets skew toward higher mass and tighter orbits than Earth. It is also the clearest argument this project can make for why the PARIKALP mission depends on predicting each planet's mass and radius individually. Assuming Earth-like values for an unknown world would have been wrong for the vast majority of the planets evaluated here, and for a mission with exactly one shot at the correct engine burn, that margin of error was never acceptable to begin with.
