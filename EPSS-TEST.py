# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ------------------------------------------------------------------------------
# Step 1: Data Loading from EPSS API
# ------------------------------------------------------------------------------
# Use the EPSS API endpoint to fetch data for the first 100 CVEs.
# The envelope=true&pretty=true parameters ensure nicely formatted JSON.
EPSS_API_URL = 'https://api.first.org/data/v1/epss?envelope=true&pretty=true'

try:
    response = requests.get(EPSS_API_URL)
    response.raise_for_status()  # Raise an error for bad responses
except requests.RequestException as e:
    raise SystemExit(f"Error fetching data from EPSS API: {e}")

# Parse the JSON response.
result = response.json()
if 'data' not in result:
    raise ValueError("The API response does not contain a 'data' key.")

# Convert the list of CVE dictionaries into a Pandas DataFrame.
data_list = result['data']
df = pd.DataFrame(data_list)

print("Data loaded from EPSS API:")
print(df.head())

# ------------------------------------------------------------------------------
# Step 2: Data Preprocessing and Feature Engineering
# ------------------------------------------------------------------------------
# Convert 'epss' and 'percentile' values to floats.
df['epss'] = df['epss'].astype(float)
if 'percentile' in df.columns:
    df['percentile'] = df['percentile'].astype(float)

# Extract numerical features from the 'cve' string.
def extract_year(cve):
    try:
        return int(cve.split('-')[1])
    except (IndexError, ValueError):
        return np.nan

def extract_id_number(cve):
    try:
        return int(cve.split('-')[2])
    except (IndexError, ValueError):
        return np.nan

df['year'] = df['cve'].apply(extract_year)
df['id_number'] = df['cve'].apply(extract_id_number)

# Remove rows where feature extraction failed.
df.dropna(subset=['year', 'id_number'], inplace=True)

# For regression, set the target as the actual EPSS score.
target = 'epss'

# Select features. Here we're using the extracted year, numeric ID, and percentile (if available).
feature_cols = ['year', 'id_number']
if 'percentile' in df.columns:
    feature_cols.append('percentile')

X = df[feature_cols]
y = df[target]

print("\nFeature DataFrame preview:")
print(X.head())
print("\nTarget preview:")
print(y.head())

# ------------------------------------------------------------------------------
# Step 3: Data Splitting
# ------------------------------------------------------------------------------
# Split data into training (70%) and a temporary set (30%), then split the temporary set into validation and test sets.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ------------------------------------------------------------------------------
# Step 4: Feature Scaling
# ------------------------------------------------------------------------------
# Standardize the feature values.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------------------------
# Step 5: Baseline Model Training using XGBoostRegressor
# ------------------------------------------------------------------------------
# Initialize and train a baseline XGBoost regressor. We use the 'reg:squarederror' objective.
baseline_model = XGBRegressor(objective='reg:squarederror', random_state=42)
baseline_model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------------------------
# Step 6: Baseline Model Evaluation
# ------------------------------------------------------------------------------
# Make predictions on the validation and test sets.
y_val_pred = baseline_model.predict(X_val_scaled)
y_test_pred = baseline_model.predict(X_test_scaled)

# Calculate performance metrics.
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'\nBaseline XGBoost Model Validation MSE: {mse_val:.6f}, R2: {r2_val:.3f}')
print(f'Baseline XGBoost Model Test MSE: {mse_test:.6f}, R2: {r2_test:.3f}')

# ------------------------------------------------------------------------------
# Step 7: Optional Hyperparameter Tuning with GridSearchCV for XGBoostRegressor
# ------------------------------------------------------------------------------
# Define a grid of hyperparameters to search over.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# Fit the grid search to the training data.
grid_search.fit(X_train_scaled, y_train)
print("\nBest Hyperparameters found for XGBoostRegressor:")
print(grid_search.best_params_)

# Retrieve and evaluate the best model.
best_model = grid_search.best_estimator_
y_val_pred_tuned = best_model.predict(X_val_scaled)
y_test_pred_tuned = best_model.predict(X_test_scaled)
mse_val_tuned = mean_squared_error(y_val, y_val_pred_tuned)
mse_test_tuned = mean_squared_error(y_test, y_test_pred_tuned)
r2_val_tuned = r2_score(y_val, y_val_pred_tuned)
r2_test_tuned = r2_score(y_test, y_test_pred_tuned)

print(f'Tuned XGBoost Model Validation MSE: {mse_val_tuned:.6f}, R2: {r2_val_tuned:.3f}')
print(f'Tuned XGBoost Model Test MSE: {mse_test_tuned:.6f}, R2: {r2_test_tuned:.3f}')
