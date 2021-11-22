import numpy
import pandas
import scipy.stats as stats

from sklearn import preprocessing, naive_bayes

# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'BOTH'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   X2 = stats.chi2_contingency(countTable, correction = False)[0]
   print("X2:   ", X2)
   n = numpy.sum(numpy.sum(countTable))
   print('n:   ', n)
   minDim = min(countTable.shape)-1
   print('minDim:   ', minDim)
   V = numpy.sqrt((X2/n) / minDim)
   print("Cramer's V:  ", V)

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

inputData = pandas.read_csv('Assignment 4\\Purchase_Likelihood.csv',
                         delimiter=',')

# insurance -> group_size, homeowner, married_couple
subData = inputData[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()

catinsurance = subData['insurance'].unique()
catgroup_size = subData['group_size'].unique()
cathomeowner = subData['homeowner'].unique()
catmarried_couple = subData['married_couple'].unique()

print('Unique Values of insurance: \n', catinsurance)
print('Unique Values of group_size: \n', catgroup_size)
print('Unique Values of homeowner: \n', cathomeowner)
print('Unique Values of married_couple: \n', catmarried_couple)



RowWithColumn(rowVar = subData['insurance'], columnVar = subData['group_size'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['homeowner'], show = 'ROW')
RowWithColumn(rowVar = subData['insurance'], columnVar = subData['married_couple'], show = 'ROW')

subData = subData.astype('category')
xTrain = pandas.get_dummies(subData[['group_size', 'homeowner', 'married_couple']])

yTrain = numpy.where(subData['insurance'] == 0, 1, 2)

# Incorrectly Use sklearn.naive_bayes.BernoulliNB

#_objNB = naive_bayes.BernoulliNB(alpha = 1e-10)
#thisModel = _objNB.fit(xTrain, yTrain)
#
#print('Probability of each class:')
#print(numpy.exp(_objNB.class_log_prior_))
#print('\n')
#
#print('Empirical probability of features given a class, P(x_i|y)')
#print(xTrain.columns)
#print(numpy.exp(_objNB.feature_log_prob_))
#print('\n')
#
#print('Number of samples encountered for each class during fitting')
#print(_objNB.class_count_)
#print('\n')
#
#print('Number of samples encountered for each (class, feature) during fitting')
#print(_objNB.feature_count_)
#print('\n')
#
#xTest = pandas.DataFrame(numpy.zeros((1, xTrain.shape[1])), columns = xTrain.columns)
#
#xTest[['CreditCard_American Express', 'Gender_Female', 'JobCategory_Professional']] = [1,1,1]
#y_predProb = thisModel.predict_proba(xTest)
#
#print(y_predProb)

# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = ['group_size', 'homeowner', 'married_couple']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['insurance'])
yLabel = labelEnc.inverse_transform([0, 1])

ugroup_size = numpy.unique(subData['group_size'])
uhomeowner = numpy.unique(subData['homeowner'])
umarried_couple = numpy.unique(subData['married_couple'])

featureCategory = [ugroup_size, uhomeowner, umarried_couple]
print(featureCategory)

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['group_size', 'homeowner', 'married_couple']])

_objNB = naive_bayes.CategoricalNB(alpha = 0)
thisModel = _objNB.fit(xTrain, yTrain)

print('Number of samples encountered for each class during fitting')
print(yLabel)
print(_objNB.class_count_)
print('\n')

print('Probability of each class:')
print(yLabel)
print(numpy.exp(_objNB.class_log_prior_))
print('\n')

print('Number of samples encountered for each (class, feature) during fitting')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(_objNB.category_count_[i])
   print('\n')

print('Empirical probability of features given a class, P(x_i|y)')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(numpy.exp(_objNB.feature_log_prob_[i]))
   print('\n')

# CreditCard = American Express, Gender = Female, JobCategory = Professional
#xTest = featureEnc.transform([['American Express', 'Female', 'Professional']])
#
#y_predProb = thisModel.predict_proba(xTest)
#print('Predicted Probability: ', yLabel, y_predProb)