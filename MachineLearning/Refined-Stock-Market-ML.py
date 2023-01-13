import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
This code is a machine learning program that uses a Random Forest Regressor model to predict the 
"Adj Close" column of a stock market data set and then it splits the data into training and test sets. 
It then defines a parameter grid to be used in a Grid SearchCV object, which is then used to find the 
best parameters for the Random Forest Regressor model. After that, it makes predictions on the test data
and calculates the mean squared error, mean absolute error, and R-squared score of the predictions. 
The final lines of the code create a dataframe of the predictions and actual prices, creates a column 
that calculates the difference between the predictions and actual prices, sets a threshold for the difference,
and creates lists of possible prices to buy and sell at based on the difference.
"""
# Load the historical stock data
file_path = 'MRNA.csv'
absolute_path = os.path.abspath(file_path)
data = pd.read_csv(absolute_path)

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

# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(y_test, predictions)
print(f"Mean absolute error: {mae}")

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print(f"R-squared score: {r2}")


# Use the best estimator to make predictions on the test data
predictions = grid_search.best_estimator_.predict(X_test)

# Create a dataframe to store the predictions and actual prices
result = pd.DataFrame({'predictions': predictions, 'actual': y_test})

# Create a column that calculates the difference between the predictions and actual prices
result['difference'] = result['predictions'] - result['actual']

# Set a threshold for the difference
threshold = 0.1

# Create a list of possible prices to buy at
buy_prices = result[result['difference'] > threshold]['actual'].tolist()

# Create a list of possible prices to sell at
sell_prices = result[result['difference'] < -threshold]['actual'].tolist()

# Print the lists in a more easy to understand format
print("Possible prices to buy at:")
for price in buy_prices:
    print(price)

print("Possible prices to sell at:")
for price in sell_prices:
    print(price)
