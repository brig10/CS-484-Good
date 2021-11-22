# Importing Libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix

# Creating Training & Testing set from the given data
X_train = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
y_train = np.array([0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1])
X_test = np.array([0, 1, 2, 3, 4])
y_test = np.array([0,0,1,1,1])

# Logistic model with specified parameters
lgr_model = LogisticRegression(fit_intercept=True, class_weight=1, tol=1e-8, 
                               solver='newton-cg', max_iter=100)

# Making the feature set a 2D Array
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Fit the model on train data
lgr_model.fit(X_train, y_train)

# Coefficient & Intercept
coef = lgr_model.coef_
intercept = lgr_model.intercept_

# Prediction on test data
y_pred = lgr_model.predict(X_test)

# Calculating misclassification rate using MSE
print('MSE:', mean_squared_error(y_test, y_pred))
# Misclassification rate using Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)

tn = matrix[0][0]
fp = matrix[0][1]
fn = matrix[1][0]
tp = matrix[1][1]

misclass_rate = (fp + fn) / np.sum(matrix)
error = (fp + fn) / (tn + fp + fn + tp)
print('poop' , error)