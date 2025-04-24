# compare_epss_predictions.py

import pandas as pd
import joblib

# 1) Load model, scaler, and test data
model   = joblib.load('epss_model.joblib')
scaler  = joblib.load('scaler.joblib')
test_df = pd.read_csv('test_data.csv')

# 2) Prepare features & predict
feature_cols = [c for c in ['year','id_number','percentile'] if c in test_df.columns]
X_test = test_df[feature_cols]
X_test_s = scaler.transform(X_test)

test_df['predicted_epss'] = model.predict(X_test_s)

# 3) Compute absolute error & show top 10
test_df['error'] = (test_df['predicted_epss'] - test_df['actual_epss']).abs()
out = test_df[['cve','actual_epss','predicted_epss','error']].sort_values('error', ascending=False)

print("Top 10 largest errors:")
print(out.head(10).to_string(index=False))
