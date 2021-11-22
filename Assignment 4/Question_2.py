import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neighbors as kNN
import sklearn.svm as svm
import sklearn.tree as tree
import statsmodels.api as sm

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)  
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 13)
numpy.set_printoptions(precision = 13)

trainData = pandas.read_csv('Assignment 4\\SpiralWithCluster.csv',
                              usecols = ['SpectralCluster', 'x', 'y'])

y_threshold = trainData['SpectralCluster'].mean()



# Build Support Vector Machine classifier
xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20211111, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['id'] = y_predictClass

svm_Mean = trainData.groupby('id').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
w = thisFit.coef_[0]
bSlope = -w[0] / w[1]
xx = numpy.linspace(-3, 3)
aIntercept = (thisFit.intercept_[0]) / w[1]
yy = aIntercept + bSlope * xx

# plot the parallels to the separating hyperplane that pass through the
# support vectors
supV = thisFit.support_vectors_
wVect = thisFit.coef_
crit = thisFit.intercept_[0] + numpy.dot(supV, numpy.transpose(thisFit.coef_))

b = thisFit.support_vectors_[0]
yy_down = (b[1] - bSlope * b[0]) + bSlope * xx

b = thisFit.support_vectors_[-1]
yy_up = (b[1] - bSlope * b[0]) + bSlope * xx

cc = thisFit.support_vectors_

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(dpi=200)
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(xx, yy, color = 'black', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

trainData['radius'] = numpy.sqrt(trainData['x']**2 + trainData['y']**2)
trainData['theta'] = numpy.arctan2(trainData['y'], trainData['x'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

trainData['theta'] = trainData['theta'].apply(customArcTan)

carray = ['red', 'blue']
plt.figure(dpi=100)
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

group = []
for i in range(100):
    if ((trainData['theta'][i] > 6) and (trainData['radius'][i] < 1.5)):
        group.append(0)
    elif (((trainData['theta'][i] >= 3) and (trainData['radius'][i] <= 2.5)) or ((trainData['theta'][i] >= 5) and (trainData['radius'][i] < 3))):
        group.append(1)
    elif (((trainData['theta'][i] >= 0) and (trainData['radius'][i] <= 2.5)) or ((trainData['theta'][i] >= 2) and (trainData['radius'][i] <= 3)) or ((trainData['theta'][i] >= 3) and (trainData['radius'][i] <= 4))):
        group.append(2)
    else:
        group.append(3)

print(group)

trainData['Group'] = group

xTrain = trainData[['x','y']]
yTrain = trainData['Group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20211111, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

carray = ['red', 'blue', 'green', 'black']
plt.figure(dpi=200)
for i in range(4):
    subData = trainData[trainData['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.legend(title = 'Predicted Class', loc = 'upper center', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

data01 = trainData[trainData.Group == 0].append(trainData[trainData.Group == 1])
data12 = trainData[trainData.Group == 1].append(trainData[trainData.Group == 2])
data23 = trainData[trainData.Group == 2].append(trainData[trainData.Group == 3])

xTrain = data01[['x','y']]
yTrain = data01['Group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20211111, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Group 0 vs Group 1')

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
data01['id'] = y_predictClass

svm_Mean = trainData.groupby('id').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


xTrain = data12[['x','y']]
yTrain = data12['Group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20211111, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Group 1 vs Group 2')

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
data12['id'] = y_predictClass

svm_Mean = trainData.groupby('id').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


xTrain = data23[['x','y']]
yTrain = data23['Group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20211111, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Group 2 vs Group 3')

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
data23['id'] = y_predictClass

svm_Mean = trainData.groupby('id').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

x = numpy.linspace(0, 5)
y1 = 4146.459604*x + 1.0000367
y2 = 12652.03674*x - 0.9999503
y3 = 130.1646923*x - 1.9298516

carray = ['red', 'blue', 'green', 'black']
plt.figure(dpi=200)
for i in range(4):
    subData = trainData[trainData['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.plot(x, y1, color = 'black', linestyle = '-')
plt.plot(x, y2, color = 'black', linestyle = '-')
plt.plot(x, y3, color = 'black', linestyle = '-')
plt.title('Support Vector Machines')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.legend(title = 'Predicted Class', loc = 'upper center', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


carray = ['red', 'blue']
plt.figure(dpi=200)
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.plot(x, y1, color = 'black', linestyle = '-')
plt.plot(x, y2, color = 'black', linestyle = '-')
plt.plot(x, y3, color = 'black', linestyle = '-')
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'upper center', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()




