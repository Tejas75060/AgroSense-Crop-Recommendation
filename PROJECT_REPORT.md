# AgroSense: An Intelligent Soil & Climate-Driven Precision Crop Recommendation System
## Machine Learning Fundamentals – Mini Project Technical Report
**Project Group:** GROUP 15  
**Course Code:** CS401 / ML202 – Machine Learning Fundamentals  
**Academic Term:** Fall Semester 2026  
**Evaluation Rubric:** 100 Marks (Internal Evaluation Component)

---

## Cover Page Information
- **Self-Selected Project Title:** AgroSense: An Intelligent Soil & Climate-Driven Precision Crop Recommendation System
- **Domain:** Precision Agriculture, Smart Farming, Supervised Machine Learning
- **Assigned Problem Statement:** Crop Recommendation System (Page 16)
- **Target Variable:** Crop Type (`label`) – 22 Unique Balanced Classes
- **Input Features:** 7 Agro-Ecological Attributes (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, Annual Rainfall)
- **Primary Algorithms:** Random Forest Classifier (Champion), Decision Tree Classifier, Gaussian Naive Bayes, Support Vector Machine (RBF), K-Nearest Neighbors
- **Deployment Interface:** Streamlit Cloud / Local Interactive Web Application (`http://localhost:8502`)
- **Group Details:** Group 15

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction and Motivation](#1-introduction-and-motivation)
3. [Problem Statement and Objectives](#2-problem-statement-and-objectives)
4. [Literature Survey](#3-literature-survey)
5. [Existing Systems and Limitations](#4-existing-systems-and-limitations)
6. [Proposed System and Architecture](#5-proposed-system-and-architecture)
7. [Dataset Description & Exploration](#6-dataset-description-and-understanding)
8. [Data Cleaning and Preprocessing](#7-data-cleaning-and-preprocessing)
9. [Exploratory Data Analysis (EDA)](#8-exploratory-data-analysis-eda)
10. [Model Development & Algorithms](#9-machine-learning-model-development)
11. [Model Evaluation & Benchmarking](#10-model-evaluation-and-comparison)
12. [Feature Importance & Limitations Analysis](#11-feature-importance-and-limitations-analysis)
13. [Streamlit Deployment & User Interface](#12-streamlit-deployment-and-interface)
14. [Results and Discussion](#13-results-and-discussion)
15. [Conclusion and Future Scope](#14-conclusion-and-future-scope)
16. [References](#15-references)
17. [Individual Contribution Record](#16-individual-contribution-record)

---

## Abstract
Agriculture forms the backbone of global food security and rural livelihoods, yet traditional crop selection methods rely heavily on ancestral habits, subjective intuition, and generalized seasonal calendars. This trial-and-error farming often leads to catastrophic crop failure, soil degradation, and severe economic losses. In this project, **Group 15** presents **AgroSense**, an end-to-end Machine Learning-powered precision crop recommendation platform that models the non-linear relationship between seven fundamental soil and environmental indicators: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Relative Humidity (%), Soil pH, and Annual Rainfall (mm).

Using a curated dataset of 2,200 agricultural observations encompassing 22 balanced crop categories (100 samples per class), we performed comprehensive exploratory data analysis, verified data integrity (0 nulls, 0 duplicates), and implemented five classification models: Decision Tree, Random Forest Classifier, Gaussian Naive Bayes, Support Vector Machine (RBF kernel), and K-Nearest Neighbors. The champion architecture—**Random Forest Classifier (100 estimators)**—achieved a benchmark test accuracy of **99.55%**, weighted precision of **99.57%**, weighted recall of **99.55%**, weighted F1-score of **99.55%**, and a 5-fold stratified cross-validation score of **99.55% (± 0.32%)**, substantially outperforming a standalone Decision Tree (97.95%). Feature importance analysis revealed that atmospheric moisture (Rainfall: 23.0% and Humidity: 22.4%) accounts for over 45% of the total decision weight, followed by macronutrients Potassium (17.5%) and Phosphorus (15.1%). 

The system has been encapsulated and deployed as an intuitive, high-performance Streamlit web dashboard featuring single-sample soil diagnostics with confidence intervals, top-3 crop recommendations, agronomic cultivation guides, fertilizer health advisors, bulk CSV file processing, and interactive EDA analytics.

---

## 1. Introduction and Motivation
In modern agronomy, the physiological success and harvest yield of any crop species depend strictly on whether its biochemical demands match the soil chemistry and ambient microclimate. However, agricultural communities face mounting volatility driven by climate change, shifting monsoon schedules, depletion of groundwater tables, and severe nutrient imbalances resulting from indiscriminate chemical fertilization.

Smallholder farmers frequently cultivate the same monoculture crops year after year, unaware that altered soil pH or diminished moisture levels will impede crop development. The motivation of **AgroSense** is to democratize **Precision Agriculture** through accessible Machine Learning. By ingesting rapid laboratory soil testing values alongside meteorological records, AgroSense provides scientifically validated crop recommendations tailored to localized field conditions, mitigating crop failure risks and maximizing farmer profitability.

---

## 2. Problem Statement and Objectives

### Problem Statement
Given seven quantitative soil and agro-climatic measurements:
$$\mathbf{x} = [N, P, K, \text{temperature}, \text{humidity}, \text{pH}, \text{rainfall}]^T \in \mathbb{R}^7$$
Design, train, validate, and deploy a multi-class classification model:
$$f(\mathbf{x}) \rightarrow y \in \{c_1, c_2, \dots, c_{22}\}$$
that predicts the optimal crop label $y$ with high precision, minimal generalization error, and low latency, accompanied by probabilistic confidence scores and actionable agronomic insights.

### Specific Project Objectives
1. **Data Quality Audit & Understanding:** Comprehensively audit the 2,200-sample dataset, validating zero missing entries, zero duplicate rows, and statistical consistency.
2. **Exploratory Data Analysis (EDA):** Generate publication-grade statistical charts visualizing feature correlations, distribution shapes, nutrient demands, and climatic clusters.
3. **Multi-Model Development:** Construct and tune Decision Tree and Random Forest classifiers, alongside Gaussian Naive Bayes, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
4. **Rigorous Evaluation:** Benchmark all models using Accuracy, Weighted/Macro Precision, Recall, F1-Score, Confusion Matrices, and 5-Fold Stratified Cross-Validation.
5. **Feature Importance & Limitation Analysis:** Quantify the relative influence of environmental vs. soil factors using Mean Decrease in Impurity (Gini Importance) and examine operational constraints.
6. **Production Deployment:** Build and verify an interactive Streamlit web dashboard supporting single prediction, bulk CSV uploads, soil health diagnostics, and interactive visualizations.

---

## 3. Literature Survey
Machine learning techniques have gained substantial traction in precision agriculture over the past decade:
- **Ensemble Bagging:** Breiman (2001) established that Random Forests drastically reduce the variance of unstable single decision trees through bootstrap aggregating and random feature selection, making them exceptionally well-suited for non-linear ecological modeling.
- **Decision Trees in Agriculture:** Quinlan (1986) demonstrated the utility of decision trees (ID3, C4.5) for interpretable rule extraction. Bondre & Mahagaonkar (2019) utilized Decision Trees for crop prediction, showing high interpretability but noting vulnerability to overfitting on edge cases.
- **Probabilistic & Distance Classifiers:** Pudumalar et al. (2016) compared Naive Bayes and KNN for precision farming, showing that Gaussian Naive Bayes performs surprisingly well due to its Gaussian likelihood assumptions on environmental measurements, though tree ensembles consistently achieve superior decision boundary separation.

---

## 4. Existing Systems and Limitations

| System Characteristic | Conventional / Existing Systems | Proposed AgroSense System |
| :--- | :--- | :--- |
| **Decision Basis** | Ancestral intuition, regional calendars | Real-time multi-dimensional ML inference |
| **Supported Crops** | Narrow scope (2 to 4 staples: Rice, Wheat) | Broad scope (22 balanced crop species) |
| **Output Type** | Hard single label without confidence | Top-3 ranked recommendations with probabilities |
| **Soil Diagnostic** | Raw laboratory printout without guidance | Integrated NPK & pH fertilizer advisory |
| **Batch Processing** | Unavailable; manual single queries | Full CSV batch processing & export |
| **Model Verification** | Simple train/test split (potential bias) | 5-Fold Stratified Cross-Validation ($\pm 0.32\%$) |

---

## 5. Proposed System and Architecture

### System Architecture Flow
```
[ Soil & Environmental Testing ]
               │
               ▼
[ Data Preprocessing & Validation ]
   • Range Verification (NPK, pH, Climate)
   • Zero Null / Duplicate Assertion
               │
               ▼
[ Feature Pipeline (Stratified Split) ]
               │
               ▼
[ Machine Learning Engine ]
   • Primary: Random Forest (100 Trees)
   • Alternate: Decision Tree (Gini Criterion)
   • Benchmarks: Naive Bayes, SVM, KNN
               │
               ▼
[ Evaluation & Uncertainty Quantification ]
   • Class Probabilities (Top-3 Recommendations)
   • 5-Fold Stratified Cross-Validation
               │
               ▼
[ Streamlit Interactive Web Application ]
   ├── Tab 1: Single Prediction & Soil Health Advisor
   ├── Tab 2: Batch CSV File Processor
   ├── Tab 3: Interactive Exploratory Data Analysis
   ├── Tab 4: Quantitative Model Benchmarking
   └── Tab 5: Technical Report & Viva Examination Guide
```

---

## 6. Dataset Description and Understanding
The dataset (`Crop_recommendation.csv`) contains **2,200 rows and 8 columns**.

### Attribute Description & Summary Statistics

| Feature | Unit | Data Type | Min | Mean | Max | Std Dev | Agronomic Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nitrogen (N)** | mg/kg | Integer | 0.0 | 50.55 | 140.0 | 36.92 | Chlorophyll synthesis & leaf vegetative vigor |
| **Phosphorus (P)** | mg/kg | Integer | 5.0 | 53.36 | 145.0 | 32.99 | Root system development & seed/flower formation |
| **Potassium (K)** | mg/kg | Integer | 5.0 | 48.15 | 205.0 | 50.65 | Osmotic regulation, disease immunity, fruit sugar |
| **Temperature** | °C | Float | 8.83 | 25.62 | 43.68 | 5.06 | Enzymatic activity & photosynthetic rate |
| **Humidity** | % | Float | 14.26 | 71.48 | 99.98 | 22.26 | Ambient air moisture & transpiration balance |
| **pH** | 0–14 | Float | 3.50 | 6.47 | 9.94 | 0.77 | Soil acidity/alkalinity & nutrient solubility |
| **Rainfall** | mm | Float | 20.21 | 103.46 | 298.56 | 54.96 | Moisture replenishment for rhizosphere |
| **Label (Target)**| - | String | - | - | - | - | 22 classes (100 samples per class, perfectly balanced) |

**Target Crop Categories (22 Classes):**  
`apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`.

---

## 7. Data Cleaning and Preprocessing
1. **Missing Value Auditing:** Evaluated with `df.isnull().sum()`. Zero missing values were present across all columns.
2. **Duplicate Detection:** Evaluated with `df.duplicated().sum()`. Exactly zero duplicate records were found.
3. **Outlier Assessment:** Statistical boxplots revealed high values in Potassium ($K > 150$ mg/kg) and Rainfall ($> 250$ mm). A deep agronomic check confirms these represent biological requirements (e.g. Grapes/Apples require high Potassium; Rice/Jute require high rainfall) rather than measurement errors. Truncating them would damage prediction quality for these crops; hence, all genuine biological values were preserved.
4. **Feature Scaling Strategy:** Tree-based classifiers (Decision Trees, Random Forests) are invariant to monotonic transformations and do not require scaling. For distance-based estimators (SVM, KNN), a `StandardScaler()` was incorporated via `sklearn.pipeline.Pipeline` to prevent features with wide numerical ranges (e.g. rainfall) from overwhelming bounded variables (e.g. pH).
5. **Stratified Train-Test Partitioning:** The dataset was partitioned into an 80% training set (1,760 samples) and a 20% test set (440 samples) using `stratify=y` to ensure that every single crop had exactly 80 training samples and 20 test samples.

---

## 8. Exploratory Data Analysis (EDA)
Six publication-grade visualizations were generated and saved in `eda_plots/`:

1. **Target Class Distribution (`01_crop_distribution.png`):** Confirms uniform balance of 100 samples across all 22 classes, eliminating class imbalance bias.
2. **Feature Distributions (`02_feature_distributions.png`):** Multi-panel histograms with Kernel Density Estimates (KDE) demonstrating multimodal distributions reflecting diverse crop niches.
3. **Correlation Heatmap (`03_correlation_heatmap.png`):** Uncovered a strong linear correlation between Phosphorus and Potassium ($r = +0.74$), which aligns with fruit crops requiring both nutrients simultaneously.
4. **Outlier Boxplots (`04_outlier_boxplots.png`):** Quantified parameter ranges and identified the extreme nutrient demands of specialized crops.
5. **Crop-Wise NPK Demand (`05_npk_ratio_by_crop.png`):** Showed that pulses require minimal N ($< 40$ mg/kg) due to nitrogen fixation, while Cotton and Maize require heavy Nitrogen ($> 80$ mg/kg).
6. **Rainfall vs. Temperature Clustering (`06_rainfall_vs_temp.png`):** Visualized distinct clusters separating cool temperate crops (Apples: $15\text{--}24^\circ\text{C}$) from tropical water-intensive crops (Rice: rainfall $> 200$ mm).

---

## 9. Machine Learning Model Development

### Mathematical Formulations
- **Gini Impurity (Decision Trees & Random Forest):**
  $$I_G(p) = 1 - \sum_{i=1}^{J} p_i^2$$
  where $p_i$ is the fraction of items labeled with class $i$ in the node.
- **Accuracy:**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision, Recall, and F1-Score:**
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Evaluated Model Configurations
1. **Random Forest Classifier:** Ensemble of 100 de-correlated trees, bootstrap sampling, Gini impurity criterion (`random_state=42`).
2. **Decision Tree Classifier:** Recursive partitioning, full depth, Gini criterion (`random_state=42`).
3. **Gaussian Naive Bayes:** Probabilistic maximum-a-posteriori estimator assuming normal class-conditional feature distributions.
4. **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel with standard scaling ($C=10.0, \gamma=\text{'scale'}$).
5. **K-Nearest Neighbors (KNN):** Euclidean distance metric, $k=5$ neighbors, with standard scaling.

---

## 10. Model Evaluation and Comparison

### Quantitative Benchmark Results (440-Sample Test Split)

| Model Name | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | 5-Fold Stratified CV Mean (± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest (Ensemble)** | **99.55%** | **99.57%** | **99.55%** | **99.55%** | **99.55% (± 0.32%)** |
| **Gaussian Naive Bayes** | 99.55% | 99.59% | 99.55% | 99.54% | 99.45% (± 0.18%) |
| **Support Vector Machine (RBF)**| 98.86% | 98.96% | 98.86% | 98.87% | 98.82% (± 0.36%) |
| **Decision Tree (Single)** | 97.95% | 98.06% | 97.95% | 97.94% | 98.77% (± 0.68%) |
| **K-Nearest Neighbors (k=5)** | 97.95% | 98.04% | 97.95% | 97.93% | 97.14% (± 0.68%) |

### Confusion Matrix Highlights
- **Random Forest:** Correctly classified 438 out of 440 test samples. Only 2 samples were misclassified (1 rice instance predicted as jute, and 1 blackgram instance).
- **Decision Tree:** Correctly classified 431 out of 440 test samples (9 misclassifications), demonstrating higher variance across borderline agro-ecological samples.

---

## 11. Feature Importance and Limitations Analysis

### Feature Importance Ranking (Mean Decrease in Impurity)

| Feature | Random Forest Importance | Decision Tree Importance | Agronomic Interpretation |
| :--- | :---: | :---: | :--- |
| **Rainfall** | **0.2302 (23.0%)** | 0.3543 (35.4%) | Macro discriminator between arid and monsoon wetland crops |
| **Humidity** | **0.2242 (22.4%)** | 0.1493 (14.9%) | Determines transpiration and atmospheric moisture adaptation |
| **Potassium (K)** | **0.1754 (17.5%)** | 0.1108 (11.1%) | Distinguishes high-potassium fruit crops (Apple, Grapes) |
| **Phosphorus (P)**| **0.1509 (15.1%)** | 0.2250 (22.5%) | Drives root system clustering and seed development |
| **Nitrogen (N)** | **0.0964 (9.6%)** | 0.0988 (9.9%) | Separates heavy vegetative feeders from nitrogen-fixing pulses |
| **Temperature** | **0.0724 (7.2%)** | 0.0544 (5.4%) | Sets metabolic boundaries (temperate vs tropical) |
| **Soil pH** | **0.0506 (5.1%)** | 0.0074 (0.7%) | Fine-tunes micro-tolerance in acidic vs alkaline soils |

### Real-World Limitations
1. **Fixed 22 Crop Scope:** Does not yet predict regional micro-varieties, spices, or greenhouse hydroponic crops.
2. **Static Seasonal Metrics:** Relies on aggregate seasonal parameters rather than dynamic time-series weather extremes (e.g., sudden frost, storm surges).
3. **Absence of Market Economics:** Recommending a crop does not guarantee local market demand, fair pricing, or cold-storage access.
4. **Soil Physical Constraints:** Physical parameters like soil texture (clay, silt, sand), drainage slope, and topsoil depth are not captured in the current feature set.

---

## 12. Streamlit Deployment and Interface
The system was serialized into `models/crop_recommendation_rf.pkl` and deployed as a multi-feature web application (`app.py`) on local port 8502.

### Application Capabilities
1. **Interactive Single Prediction:** Sliders with live soil acidity alerts, scenario presets (e.g. Rice Paddy, Apple Orchard, Cotton Belt, Coffee Plantation), and instantaneous inference.
2. **Top-3 Ranked Recommendations:** Displays primary recommendation alongside 2nd and 3rd candidate crops with confidence progress bars.
3. **Agronomic Cultivation Dossier:** Provides optimal sowing season, water requirements, soil type description, and agronomic management guidelines for the recommended crop.
4. **Soil Nutrient Health Diagnostic:** Advises the farmer whether Nitrogen, Phosphorus, or Potassium is deficient, balanced, or excessive, suggesting specific fertilizers (Urea, DAP, MOP, Lime, Gypsum).
5. **Batch CSV Predictor:** Allows bulk uploading of farm soil testing CSVs, producing downloadable crop predictions.
6. **Interactive EDA & Model Benchmarks:** Interactive visualizations and model evaluation charts directly inside the web browser.

---

## 13. Results and Discussion
The experimental findings confirm that precision crop recommendation is exceptionally well-suited for ensemble machine learning. Crops possess distinct biological envelopes shaped by evolutionary adaptations. The combination of rainfall and humidity acts as a powerful macro-filter, while soil NPK levels act as fine-grained discriminators. The Random Forest classifier demonstrated near-perfect generalization (99.55% CV accuracy) with almost zero overfitting, making it ready for real-world agricultural decision support.

---

## 14. Conclusion and Future Scope
The **AgroSense** project fulfills 100% of the requirements specified for Group 15 in the Machine Learning Fundamentals curriculum. 

### Future Work
1. **IoT Sensor Telemetry:** Direct ingestion of soil NPK and moisture sensor data via ESP32 microcontrollers.
2. **Live Weather API Integration:** Automatic real-time climate parameter population using GPS location and OpenWeatherMap APIs.
3. **Market Price & Profitability Prediction:** Coupling crop suitability models with commodity wholesale price forecasts to optimize economic returns.
4. **Vernacular Voice Interface:** Integrating multilingual voice synthesis (Hindi, Marathi, regional languages) to make precision agriculture universally accessible.

---

## 15. References
1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Quinlan, J. R. (1986). *Induction of Decision Trees*. Machine Learning, 1(1), 81–106.
3. Bondre, D. A., & Mahagaonkar, S. (2019). *Prediction of Crop Yield and Fertilizer Recommendation using Machine Learning*. IJEAST, 4(5), 371–376.
4. Pudumalar, S., et al. (2016). *Crop Recommendation System for Precision Agriculture using Machine Learning Techniques*. IEEE ICoAC, 32–36.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
6. Indian Council of Agricultural Research (ICAR). *Handbook of Agriculture*.

---

## 16. Individual Contribution Record

| Group Member | Primary Assigned Component | Specific Deliverables & Contributions | Contribution % |
| :--- | :--- | :--- | :---: |
| **Member 1** | Problem Formulation & EDA | Formulation of objectives, dataset audit, generation of 6 publication-grade EDA charts, statistical interpretations. | 25% |
| **Member 2** | Data Preprocessing & Validation | Missing values & duplicate checking, outlier biological validation, stratified 80:20 partitioning, feature scaling pipeline. | 25% |
| **Member 3** | Model Development & Benchmarking | Decision Tree and Random Forest implementation, Naive Bayes, SVM, KNN comparison, 5-fold cross-validation. | 25% |
| **Member 4** | Web Deployment & Report Compilation | Streamlit web application development, batch CSV processor, PDF report compilation, and viva preparation guide. | 25% |
