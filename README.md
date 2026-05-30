# Apple Stock Prediction & Sentiment Analysis Pipeline

This repository provides an end-to-end machine learning workflow that analyzes Apple Inc. stock prices (`AAPL`) alongside GPT-generated news sentiment scores. 

By combining traditional technical indicators with advanced natural language sentiment analysis, this pipeline performs automated data preprocessing, multi-model regression, sentiment-based classification, and hyperparameter tuning to find the top 5 optimal predictive combinations.

---

## 📑 Table of Contents

- [Phase 1: Preprocessing](#phase-1-preprocessing)
- [Phase 2: Regression Analysis](#phase-2-regression-analysis)
- [Phase 3: Classification & Sentiment Comparison](#phase-3-classification--sentiment-comparison)
- [Phase 4: Top 5 Combination Search](#phase-4-top-5-combination-search)

---

## 🛠️ Phase 1: Preprocessing

**Raw Data:** A dataset compiled from major news articles regarding Apple's flagship product, the iPhone, from 2007 onwards.

### Step 1: Data Initialization
1. Sort data chronologically and standardize timestamps to `UTC`.
2. Filter data to include only records from **2020 onwards**, eliminating historical noise from pre-2019 data.
3. Print basic dataset info (shape, columns, data types, and descriptive statistics).

### Step 2: Dirty Data Cleaning
- **Deduplication:** Remove exact duplicate rows and partial duplicates based on the article `Title`.
- **OHLC Logical Validation:** Ensure price logic (e.g., `High` must be $\ge$ `Low`).
- **Negative Value Check:** Verify that prices and volumes strictly contain non-negative values.
- **Remove Sparse/Irrelevant Columns:** Drop sparsely populated columns (e.g., `Dividends`, `Stock Splits`) and text columns already quantified by GPT (`Title`, `Link`, `Source`, `gpt_summary`).

### Step 3: GPT Sentiment Feature Refinement
Consolidate positive and negative GPT scores into a single `signed_sentiment_score` to clearly indicate sentiment direction with +/- signs.

### Step 4: Outlier Detection
We analyze outliers across key metrics:
`Open`, `High`, `Low`, `Close`, `Volume`, `ATR`, `ADX`, `MACD_Signal`.
*Note: Outliers in financial markets often reflect genuine macroeconomic events rather than system errors. Therefore, they are retained.*

### Step 5: Rare Category Consolidation
To prevent the curse of dimensionality during One-Hot Encoding, categorical labels in `gpt_event_type` with a frequency of less than `10` are grouped into a unified `Other` category.

### Step 6: Feature Engineering
Generate domain-specific technical indicators to help the model learn market patterns:
1. `Daily_Volatility`: `(High - Low) / Open`
2. `Buy_Pressure` & `Sell_Pressure`: Measures intraday buying/selling strength based on shadows (wicks).
3. `Close_Change_Rate`: Daily percentage return.
4. `BB_Position`: Current price position relative to the Bollinger Bands `(Close - Lower) / (Upper - Lower)`.
5. `Volume_Change_Rate`: Daily trading volume percentage change.

### Step 7: Target Label & Market Status (Bull/Bear) Creation
- **Target (1/0):** `1` if tomorrow's closing price is strictly greater than today's; otherwise `0`.
- **Market Status (MA_20 & Bull_Bear):** Calculates the 20-day moving average (`MA_20`). If the current close is above `MA_20`, `Bull_Bear` is set to `1` (Bull Market), else `0` (Bear Market). 

### Step 8: Missing Values
Drop any rows containing `NaN` (Not a Number) across the defined features.

### Step 9: Data Visualization
Automatically generates and saves three diagnostic plots:
- **Time Series Plot:** Historical AAPL close prices.
- **Target Distribution Bar Chart:** Class balance check for Up/Down days.
- **Correlation Heatmap:** Visualizes correlations across all numerical features, including prices, technical indicators, and GPT sentiment scores.

### Step 10 & 11: Feature Selection & One-Hot Encoding
Select optimal continuous features (Moving averages, Momentum, Volatility, GPT Sentiments) and apply Boolean-to-Integer One-Hot Encoding on categorical features (`gpt_event_type`, `gpt_sentiment_direction`). Drop duplicate rows based strictly on features to prevent data leakage.

### Step 12 & 13: Train/Test Split & Scaling
- **Split:** The most recent 3 months of data are strictly reserved as the `Test` set.
- **Scaling:** Date columns and binary targets are excluded from scaling. We generate three datasets:
  1. `Raw Data` (No scaling)
  2. `Standard Scaler` (Sensitive to outliers; assumes normal distribution)
  3. `Robust Scaler` (Resistant to extreme spikes/crashes using IQR)

### Step 14 & 15: Final Export & Quality Assurance
The pipeline saves 7 distinct `.csv` files locally (Raw, Standard-scaled, Robust-scaled for Train/Test + Baseline). A final sanity check ensures no `NaN`s, duplicates, and verifies target proportions.

---

## Phase 2: Regression Analysis

The regression module attempts to predict the exact numerical 'Closing Price' for the next day, and subsequently infers the price direction.

- **Datasets Used:** `Raw Data`, `Robust Scaled`, `Standard Scaled`.
- **Algorithms Evaluated:** 
  - Linear Regression
  - Random Forest (Base & max_depth=3)
  - Decision Tree
- **Diagnostic Visualizations (Pop-up):**
  - **Prediction Error (RMSE) Bar Chart:** Lower bars indicate less deviation from actual prices.
  - **Directional Accuracy Bar Chart:** Evaluates if the model correctly predicted the Up/Down direction (>50% boundary).

---

## Phase 3: Classification & Sentiment Comparison

Unlike regression, the classification module is strictly optimized to predict market direction: `Up(1)` or `Down(0)`. This phase specifically tests whether **GPT News Sentiment** improves model performance.

- **A/B Testing & Evaluation Strategy:**
  - **Models:** Evaluates `Version A` (Technical only) vs. `Version B` (Technical + Sentiment).
  - **Market Segmentation:** Models are trained **once** on the full dataset. The generated predictions are then dynamically segmented to evaluate performance across Full Period, Bull Market, and Bear Market conditions.
- **Cross Validation:** Uses `GridSearchCV` (optimized for `f1_weighted`) with `StratifiedKFold` to find the absolute best hyperparameters for a `DecisionTreeClassifier` while preventing overfitting.
- **Diagnostic Visualizations (Pop-up):**
  1. **Performance Metrics:** Bar charts comparing Accuracy, Precision(w), Recall(Up), and F1-Score(w) across Full, Bull, and Bear market periods.
  2. **Confusion Matrices:** Heatmaps showing False Positives and False Negatives.
  3. **ROC Curves:** Evaluates classification robustness. Larger Area Under the Curve (AUC) signifies a healthier model.

---

## Phase 4: Top 5 Combination Search

The grand finale. The pipeline dynamically cross-evaluates every combination of techniques used previously to find the optimal deployment strategy.

- **Search Space:**
  - `Features`: [Base (All Technical) vs. Full (Technical + Sentiment)]
  - `Scalers`: [Standard vs. Robust vs. None]
  - `Algorithms`: [DecisionTree vs. RandomForest]
  - `Hyperparameters`: Depth, splits, estimators. (Optimized for `f1_weighted`)
- **Output:** The terminal will display the **Top 5 Best Combinations** strictly ranked by `Accuracy`. These top configurations provide ready-to-deploy specifications for your trading strategies or further analysis.
---

*End of Document. To run the full pipeline, execute `python single_top_level_func.py` in your terminal and interact with the sequential graphical outputs.*
