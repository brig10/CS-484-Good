import numpy
import pandas
import statsmodels.api as stats

X_train = pandas.DataFrame({'x': [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]})
y_train = pandas.DataFrame({'y': [0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1]}).astype('category')

# Train the multinomial logistic regression model
X_train = stats.add_constant(X_train, prepend=True)
logit = stats.MNLogit(y_train, X_train)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

# Apply the model to the test data and calculate the predicted probabilities
X_test = pandas.DataFrame({'x': [0,1,2,3,4]})
X_test = stats.add_constant(X_test, prepend=True)
y_predProb = thisFit.predict(X_test)

# Calculate the predicted y category using the predicted probability for y = 1
y_predictClass = numpy.where(y_predProb[1] >= 0.3, 1, 0)

# Identify the misclassified observations
y_test = pandas.DataFrame({'y': [0,0,1,1,1]})
y_misClass = numpy.where(y_predictClass == y_test['y'], 0, 1)

# Calcyulate the misclassification rate
misClass_Rate = numpy.mean(y_misClass)

print (misClass_Rate)