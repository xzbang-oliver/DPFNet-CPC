# ==========================================
# MAC-Mobility V1.0
# Author: Oliver Xu
# Description: Adaptive Model Tournament for 
# Intercity Population Flow Calibration
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chinese_calendar as calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.unicode_minus': False})

VERSION = "V1.0"
output_dir = f"MAC_Mobility_{VERSION}"
if not os.path.exists(output_dir): os.makedirs(output_dir)

SPRING_FESTIVAL_DATES = {
    2020: (pd.Timestamp('2020-01-10'), pd.Timestamp('2020-02-18')),
    2021: (pd.Timestamp('2021-01-28'), pd.Timestamp('2021-03-08')),
    2022: (pd.Timestamp('2022-01-17'), pd.Timestamp('2022-02-25')),
    2023: (pd.Timestamp('2023-01-07'), pd.Timestamp('2023-02-15')),
    2024: (pd.Timestamp('2024-01-26'), pd.Timestamp('2024-03-05')),
    2025: (pd.Timestamp('2025-01-14'), pd.Timestamp('2025-02-22'))
}

def get_dist_to_sf(date):
    year = date.year
    if year not in SPRING_FESTIVAL_DATES: return 100
    start, end = SPRING_FESTIVAL_DATES[year]
    if start <= date <= end: return 0
    return min(abs((date - start).days), abs((date - end).days))

# =================================================================
# Load the training registry: LBS_Anchor_Aligned_Training_Data.csv
# The input file must contain the following columns for reproducibility:
# 1. date: Calendar date (YYYY-MM-DD) for holiday and seasonal feature extraction.
# 2. BMI_adj: Adjusted Baidu Migration Index (Independent Variable/Feature).
# 3. person_times_10k: Actual observed or anchor flow volumes (Target, Unit: 10,000).
# 4. data_fidelity: Sample weights for dual-fidelity training 
#    (e.g., Daily Ground-truth = 2.0, Monthly Anchors = 1.0).
# =================================================================
df = pd.read_csv('LBS_Anchor_Aligned_Training_Data.csv')

# --- Feature Engineering ---
df['date'] = pd.to_datetime(df['date'])
df['is_festive'] = df['date'].apply(lambda x: 1 if calendar.is_holiday(x) else 0)
df['dist_to_SF'] = df['date'].apply(get_dist_to_sf)
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

X = df[['BMI_adj', 'is_festive', 'dist_to_SF', 'month', 'day_of_week']]
y = df['person_times_10k']
weights = df['data_fidelity'].values

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# --- Candidate Model Ensemble ---
models = {
    "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, verbose=0, random_seed=42),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42),
    "Ridge": Ridge(alpha=1.0)
}

results = {}
best_model = None
best_r2 = -np.inf

# --- Adaptive Model Tournament ---
for name, model in models.items():
    # Use log-transformation for target to handle long-tail distribution
    model.fit(X_train, np.log1p(y_train), sample_weight=w_train)
    y_pred = np.expm1(model.predict(X_test))
    
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results[name] = {'R2': r2, 'MAPE': mape}
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"MAC-Mobility {VERSION} Champion Model: {best_name} (R2: {best_r2:.4f})")

joblib.dump(best_model, os.path.join(output_dir, f'MAC_Mobility_Core_{VERSION}.pkl'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

y_pred_final = np.expm1(best_model.predict(X_test))
sns.regplot(x=y_test, y=y_pred_final, ax=ax1, scatter_kws={'alpha':0.4, 's':20}, line_kws={'color':'red'})
ax1.set_title(f'Technical Validation: Predicted vs Actual\n(Model: {best_name})')
ax1.set_xlabel('Actual Flow (10k)')
ax1.set_ylabel('Predicted Flow (10k)')

importances = best_model.get_feature_importance() if best_name == "CatBoost" else best_model.feature_importances_
sns.barplot(x=importances, y=X.columns, ax=ax2, palette='viridis')
ax2.set_title('Feature Importance Analysis')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'Validation_Report_{VERSION}.png'), dpi=300)
plt.show()