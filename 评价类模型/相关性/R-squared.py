# Train/test split for regression
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

# Create training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the regressor to the training data
reg_all=LinearRegression()
reg_all.fit(X_train,y_train)

# Predict on the test data: y_prediction
y_prediction=reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2:{}".format(reg_all.score(X_test,y_test)))

rmse=np.sqrt(mean_squared_error(y_test,y_prediction))
print("Root Mean Squared Error:{}".format(rmse))