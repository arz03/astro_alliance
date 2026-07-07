user: explain the core problem statement and what are the possible methodologies that we can use to approach it. what could be the steps? data preprocessing, viasulization, modelling ets (iterative process). check and explain.

AI:
**Core Problem Statement**
The challenge is set in 2084, where a manned exploration shuttle named PARIKALP is trapped in the gravitational pull of an exoplanet. The crew must fire their engines to reach the exact **Escape Velocity ($v_{e}=\sqrt{\frac{2GM}{R}}$)** to break free. If the velocity is too slow, the shuttle crashes; if it is too fast, the backup fuel tanks will explode.

The core issue is that the sensor data required to calculate this velocity—specifically the planet's mass, radius, and gravity—is corrupted with heavy noise, errors, and missing values. Because of this, standard physics equations cannot be used directly on the raw data. Additionally, a miscommunication resulted in one of the distance parameters being logged in two different units, creating a hidden inconsistency that must be resolved. 

Your objective as the lead Space Data Scientist is to build a reliable Python-based predictive machine learning pipeline using historical exoplanet data to clean the data, reconstruct missing planetary features, and accurately predict the mass and radius in order to calculate the exact escape velocity.

**Methodologies and Iterative Steps**
To approach this problem, your methodology should follow a structured, end-to-end data analytics pipeline. Based on the evaluation criteria, the steps include:

**1. Data Preprocessing & Exploratory Data Analysis (EDA)**
*   **Data Inspection:** Identify numerical and categorical features, determine data types, and generate descriptive statistics. 
*   **Handling Missing Values & Noise:** The data contains observational anomalies. You must detect noise (such as the provided `cosmic_noise_1`, `cosmic_noise_2`, `pl_orbper_noisy`, and `pl_orbsmax_noisy` features) and analyze its impact. Use appropriate imputation techniques to handle missing values.
*   **Statistical Inference:** Plot distributions to assess skewness, normalize the data if required, and handle outliers. 

**2. Visualization**
*   Use scatter plots and correlation matrices to explore relationships between features.
*   Analyze the distribution of your target variables (Planet Radius `pl_rade` and Planet Mass `pl_bmasse`).
*   Critically, you must **identify physics law violations** through your visualizations and apply mathematical relationships to correct them.

**3. Feature Engineering & Dimensionality Reduction**
*   **Data Cleaning:** Drop unnecessary columns, check for data leakage, and encode categorical features with proper justification. Correct the hidden unit inconsistency in the distance parameter.
*   **Feature Creation:** Engineer new planetary, orbital, stellar, or photometric features. You can utilize mathematical relationships for calibration, such as calculating the dimensionless transit signal depth ratio ($pl\_ratror = \frac{pl\_rade \times 6.3710 \times 10^6}{st\_rad \times 6.9570 \times 10^8}$).
*   Apply and interpret dimensionality reduction techniques.

**4. Modeling**
*   **Radius and Mass Prediction:** Implement **Multi-output Regression Models** to predict the planet's mass and radius.
*   **Optimization:** Perform cross-validation and hyperparameter tuning to ensure model robustness.
*   **Escape Velocity Calculation:** Once mass and radius are predicted, calculate the final escape velocity using the formula $v_{e}=\sqrt{\frac{2GM}{R}}$ (where $G=6.6743\times10^{-11}$, and $M$ and $R$ are converted into standard SI units using Earth's mass and radius baselines).

**5. Model Interpretation and Evaluation**
*   Evaluate your models using regression metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^{2}$.
*   Compare the performance of at least two different models and conduct a bias-variance analysis.
*   Visualize predictions (Actual vs. Predicted) and generate error/residual plots to prove mathematical viability.

**Key Rules to Keep in Mind:**
*   Your pipeline must be built entirely in **Python** (preferably Google Colab or Jupyter Notebook).
*   Any form of target leakage will result in penalties.


user: break it down further for me. why do we need to predict radius and mass? how do we do it? do we have enough training data for it? then how do we predict escape velocity once we have rad and mass?

AI:
The dataset contains intentional anomalies and observational errors that you must identify as physics-based inconsistencies through visualization. Based on the mission parameters and astrophysical constraints, you should specifically look for the following physics law violations and data inconsistencies:

**1. The Hidden Distance Unit Inconsistency**
You must find a critical logging error where one of the distance parameters was recorded in two different measurement units. Because distance is a fundamental variable in calculating orbital mechanics and gravitational forces, having mixed units in a single feature is a severe physical inconsistency. If this is not detected and mathematically corrected, any prediction you make will be entirely invalid and potentially fatal to the mission. 

**2. Geometric Size Validation (Transit Signal Depth Calibration)**
You need to verify the physical relationship between the planet's radius (`pl_rade`), the host star's radius (`st_rad`), and the dimensionless transit signal depth ratio (`pl_ratror`). The dataset documentation provides a specific equation to establish geometric size validation bounds between the planet disks and stellar radii:
$pl\_ratror = \frac{pl\_rade \times 6.3710 \times 10^6}{st\_rad \times 6.9570 \times 10^8}$.
A physics law violation occurs if the recorded data points do not satisfy this mathematical equation. By plotting these variables against one another, you can visually identify planets whose recorded sizes geometrically contradict the size of their host stars.

**3. Planetary Geometric Boundary Exceedances**
You should look for planets that violate the maximum measured metric boundaries for their physical size. The dataset benchmarks note that the geometric maximum boundary for a planetary envelope radius—specifically for highly inflated gas envelopes like close-in irradiated Hot Jupiters—is approximately 27.35 Jupiter Radii. Any planetary radius in the dataset that significantly exceeds this limit violates established astrophysical constraints. 

**How to approach finding them:**
To secure maximum points in the Exploratory Data Analysis (EDA) and Visualization sections, you cannot just filter these out using code; you must **prove their existence visually**. You should use scatter plots to expose the data points that break the `pl_ratror` geometric relationship, and plot the distributions of the distance parameters to reveal the bifurcated data caused by the mixed units. Once identified visually, you must formulate and apply the correct mathematical relationships to impute or correct these violations.

user: what should be step 0 and step 1?

AI:
**Step 0: Mission Setup and Rule Compliance**
Before writing any code, your foundational step is to set up your environment and strictly align with the competition's administrative and evaluation protocols. 
*   **Environment Setup:** You must strictly use Python, and deploying your pipeline in standard frameworks using Google Colab or Jupyter Notebook (.ipynb) is highly preferred. 
*   **Understand the Target:** Your ultimate goal is to calculate the escape velocity using $v_{e}=\sqrt{\frac{2GM}{R}}$, which means your machine learning model first needs to predict two specific target variables: `pl_rade` (Planet Radius) and `pl_bmasse` (Planet Best Mass).
*   **Rule Strictness:** Ensure all code is entirely your original work, as the use of LLMs or AI-assisted tools is strictly prohibited and will be checked via AI detection. Furthermore, you must maintain clean practices, as any form of target or data leakage will result in penalties.

**Step 1: Exploratory Data Analysis (EDA)**
According to the official marking scheme, Step 1 is the Exploratory Data Analysis phase, which carries a total of 25 marks. You need to break this down into four specific sub-sections:

*   **1.1 Data Inspection (6 Marks):** Start by inspecting the dataset (which contains up to 96 distinct features) to determine the data types of all features and identify the numerical and categorical columns. You must generate and interpret descriptive statistics, pinpoint features with missing values, and handle those missing values using appropriate imputation techniques.
*   **1.2 Statistical Inference (8 Marks):** In this step, plot the distribution of numerical features to evaluate data skewness and apply normalization if necessary. You must detect and handle outliers, and perform targeted noise detection to analyze its impact. Specifically, you should investigate features like `cosmic_noise_1`, `cosmic_noise_2`, `pl_orbper_noisy`, and `pl_orbsmax_noisy`. Use scatter plots or correlation matrices to explore relationships between variables and thoroughly analyze the distribution and potential prediction challenges for your target variables (`pl_rade` and `pl_bmasse`).
*   **1.3 Feature Selection, Dimensionality Reduction & Encoding (6 Marks):** Clean your dataset by dropping unnecessary columns and double-checking to ensure there are no data leakage columns. You must encode your categorical features while providing clear justification, select the most relevant features, and apply and interpret dimensionality reduction techniques.
*   **1.4 Visualization (5 Marks):** This is a crucial analytical step where you must visually identify physics law violations. You are expected to find anomalies, such as the miscommunicated distance parameter that was logged in two different units, and geometric bounds violations (like a planetary envelope radius exceeding 27.35 Jupiter Radii). Once identified visually, you must formulate and apply the correct mathematical relationships—such as the transit signal depth calibration formula ($pl\_ratror = \frac{pl\_rade \times 6.3710 \times 10^6}{st\_rad \times 6.9570 \times 10^8}$)—to correct and impute these inconsistencies.