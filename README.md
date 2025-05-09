A machine learning project that predicts dynamic baseball player ratings (0–10) based on raw and normalized Baseball Savant metrics.

Overview
This project combines multiple CSV files of player data, cleans and normalizes features, dynamically adjusts weights based on player performance rules, and trains a Gradient Boosting Regressor model to predict player "ratings."

It includes:

Data preparation (merging, cleaning, normalizing)

Custom dynamic rating calculation

Train/test data split

Model training with Gradient Boosting

Performance evaluation (R² score, RMSE)

Features
✅ Upload and combine multiple Baseball Savant CSVs

✅ Normalize and engineer features

✅ Rule-based adjustments to account for player nuances

✅ Machine learning modeling and evaluation

✅ Output top-rated players based on predictions

Installation
You can run this directly in Google Colab with no local setup.
Or, locally:

bash
Copy
Edit
pip install pandas scikit-learn numpy
Usage
Upload your Baseball Savant CSV(s).

Merge datasets and normalize key stats.

Run the rating calculator function.

Train the Gradient Boosting Regressor.

Evaluate model performance and view player ratings.

Example of model training:

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)
Example evaluation:

python
Copy
Edit
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
How Ratings Are Calculated
Normalize player stats (e.g., launch speed, BABIP, ISO)

Dynamically adjust feature weights based on thresholds

Calculate a final rating scaled between 0–10

Example adjustment:

python
Copy
Edit
if row.get('launch_speed', 0) > 92:
    weights['launch_speed_norm'] += 0.10
elif row.get('launch_speed', 0) < 87:
    weights['launch_speed_norm'] -= 0.05
Project Structure
Data Upload & Merge

Feature Engineering & Normalization

Dynamic Rating Calculation

Model Training

Model Evaluation

Top Player Prediction Output

Future Improvements
Support year-to-year trend modeling (2023, 2024, 2025 data)

Include defensive and baserunning metrics

Hyperparameter tuning (Grid Search or Random Search)

Deploy a web app version for live input and prediction
