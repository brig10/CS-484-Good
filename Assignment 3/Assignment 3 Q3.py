import numpy
import pandas
import scipy

import statsmodels.api as stats

data = pandas.read_csv('C:\\Users\\galla\\CS-484-Good-2\\Assignment 3\\sample_v10.csv',
                       delimiter=',')
print(data.y.value_counts())


# Specify y as a categorical variable
y = data['y'].astype('category')
y_category = y.cat.categories

# Backward Selection
# Consider Model 0 is y = Intercept + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
X = data[['x1']]
X = X.join(data[['x2']])
X = X.join(data[['x3']])
X = X.join(data[['x4']])
X = X.join(data[['x5']])
X = X.join(data[['x6']])
X = X.join(data[['x7']])
X = X.join(data[['x8']])
X = X.join(data[['x9']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)

# Consider Model 1 is y = Intercept + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x10
X = data[['x1']]
X = X.join(data[['x2']])
X = X.join(data[['x3']])
X = X.join(data[['x4']])
X = X.join(data[['x5']])
X = X.join(data[['x6']])
X = X.join(data[['x7']])
X = X.join(data[['x8']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)

# Consider Model 2 is y = Intercept + x1 + x2 + x3 + x4 + x5 + x6 + x8 + x10
X = data[['x1']]
X = X.join(data[['x2']])
X = X.join(data[['x3']])
X = X.join(data[['x4']])
X = X.join(data[['x5']])
X = X.join(data[['x6']])
X = X.join(data[['x8']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = DF0
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 3 is y = Intercept + x1 + x3 + x4 + x5 + x6 + x8 + x10
X = data[['x1']]
X = X.join(data[['x3']])
X = X.join(data[['x4']])
X = X.join(data[['x5']])
X = X.join(data[['x6']])
X = X.join(data[['x8']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = DF0
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 4 is y = Intercept + x1 + x4 + x5 + x6 + x8 + x10
X = data[['x1']]
X = X.join(data[['x4']])
X = X.join(data[['x5']])
X = X.join(data[['x6']])
X = X.join(data[['x8']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = DF0
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Consider Model 5 is y = Intercept + x1 + x4 + x6 + x8 + x10
X = data[['x1']]
X = X.join(data[['x4']])
X = X.join(data[['x6']])
X = X.join(data[['x8']])
X = X.join(data[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = DF0
DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
