"""
Random Forest Vs XGBoost

Here we'll look at the relative performances of Random Forest and XGBoost regression model.
We will use them to predict house prices using the California Housing Dataset.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing # Sklearn dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Loading the California data set
data = fetch_california_housing()
X,y = data.data, data.target

df = pd.DataFrame(data=data.data, columns=data.feature_names) # converting the data to a pandas df to have a look
df['MedhouseVal'] = y # also just to check data, real value is values x 100,000

# Checking the data out
n_observations, n_features = X.shape
print('Number of Observations: ' + str(n_observations))
print('Number of Features: ' + str(n_features))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Model time.
We'll set the number of base estimators, or individual trees to be used ini eac and then initialise. 
'''

n_estimators = 100
randomForest = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# Now fit the models
start_time_rf = time.time()
randomForest.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf # measuring the time to train

# Measure training time for XGBoost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

# Now we'll measure prediction for each
# Measure prediction time for Random Forest
start_time_rf = time.time()
y_pred_rf = randomForest.predict(X_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf

# Measure prediciton time for XGBoost
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb

'''
Calculate MSE and R^2 for each
'''
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)


'''
Now time to print results 
'''
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')

print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')

# We can see a massive difference between the times betweenRandom forest and XGBoost, the latter being substantially quicker

std_y = np.std(y_test) # std deviation for plotting

plt.figure(figsize=(14, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()


# XGBoost plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()
plt.tight_layout()
plt.show()


'''
Lets predict a house price with our models. We need the following features:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
'''

sample_features = np.array([[8.0,   # MedInc (Median income)
                             25.0,  # HouseAge
                             6.0,   # AveRooms
                             1.0,   # AveBedrms
                             1000,  # Population
                             3.0,   # AveOccup
                             34.0,  # Latitude
                            -118.0  # Longitude
                            ]])

# Predicting with both models
rf_prediction = randomForest.predict(sample_features)
xgb_prediction = xgb.predict(sample_features)

# Multiply by 100,000 to get real house price
print("Random Forest prediction: ${:,.2f}".format(rf_prediction[0] * 100000))
print("XGBoost prediction:       ${:,.2f}".format(xgb_prediction[0] * 100000))


# Show actual and predicted prices for a few test cases
for i in range(5):
    actual = y_test[i] * 100000
    pred_rf = y_pred_rf[i] * 100000
    pred_xgb = y_pred_xgb[i] * 100000
    print(f"Actual: ${actual:,.2f} | RF: ${pred_rf:,.2f} | XGB: ${pred_xgb:,.2f}")


from sklearn.metrics import mean_absolute_error

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"Random Forest MAE: ${mae_rf * 100000:,.2f}")
print(f"XGBoost MAE:       ${mae_xgb * 100000:,.2f}")

'''
So we can see our estimates are about 30,000-32,000 off. Feature selection or hyperparameter tuning may help us here
'''
