import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = 'MRNA.csv'
absolute_path = os.path.abspath(file_path)
stock_data = open(absolute_path, 'r')
# Load the historical stock data
data = pd.read_csv(stock_data)

# Prepare the data for machine learning
# Create features and labels
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Adj Close']

# Scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor()

# Define the parameter grid
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}

# Create a Grid SearchCV object
grid_search = GridSearchCV(rf_regressor, param_grid, cv=5)

# Fit the Grid SearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Use the best estimator to make predictions on the test data
predictions = grid_search.best_estimator_.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error: {mse}")



# ...

# Use the best estimator to make predictions on the test data
predictions = grid_search.best_estimator_.predict(X_test)


# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(y_test, predictions)
print(f"Mean absolute error: {mae}")

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print(f"R-squared score: {r2}")

# Calculate Precision
precision = precision_score(y_test, predictions)
print(f"Precision: {precision}")

# Calculate Recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# Calculate F1-score
f1 = f1_score(y_test, predictions)
print(f"F1-score: {f1}")
