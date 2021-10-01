
import pandas
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier

#Question 1 A
val = input('Enter NormalSameple Path: ')
inData = pandas.read_csv(val)
X = inData['x']
print(X.describe())

#Question 1 B
h = 2*(X.quantile(0.75)-X.quantile(0.25))*X.count()**(-1/3)
print('Recomomended Bin Width:  ' , h)

#Question 1 C
def calcCD (X, delta):
   maxX = np.max(X)
   minX = np.min(X)
   meanX = np.mean(X)
   middleX = delta * np.round(meanX / delta)
   nBinRight = np.ceil((maxX - middleX) / delta)
   nBinLeft = np.ceil((middleX - minX) / delta)
   lowX = middleX - nBinLeft * delta
   m = nBinLeft + nBinRight
   BIN_INDEX = 0
   boundaryX = lowX

   for iBin in np.arange(m):
      boundaryX = boundaryX + delta
      BIN_INDEX = np.where(X > boundaryX, iBin+1, BIN_INDEX)
   uBin, binFreq = np.unique(BIN_INDEX, return_counts = True)
   meanBinFreq = np.sum(binFreq) / m
   ssDevBinFreq = np.sum((binFreq - meanBinFreq)**2) / m
   CDelta = (2.0 * meanBinFreq - ssDevBinFreq) / (delta * delta)
   return(m, middleX, lowX, CDelta)

result = pandas.DataFrame()
deltaList = [1, 2, 2.5, 5, 10, 20, 25, 50]

for d in deltaList:
   nBin, middleX, lowX, CDelta = calcCD(X,d)
   highX = lowX + nBin * d
   result = result.append([[d, CDelta, lowX, middleX, highX, nBin]], ignore_index = True)

result = result.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin'})
print(result)

#Question 1 D
fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot of X')
ax1.boxplot(X, labels = ['X'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()

#Question 2 A
X_Groups = inData.groupby('group')
print(X_Groups.describe())

#Question 2 B
plotData = [X, X_Groups.get_group(0)['x'], X_Groups.get_group(1)['x']]
fig1, ax1 = plt.subplots()
ax1.set_title('Box plot of X and its Groups')
ax1.boxplot(plotData, labels = ['All of X', 'Group 0', 'Group 1'], vert=False)
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()

#Question 3 A
val2 = input('Enter Fraud Path: ')
fData = pandas.read_csv(val2)
goop = fData.groupby('FRAUD')
perc = goop.get_group(1)['CASE_ID'].count()/goop.get_group(0)['CASE_ID'].count()
print(perc*100, '%_fraud')

#Question 3 B
x = np.matrix(fData.drop(columns = ['CASE_ID', 'FRAUD']))
xtx = x.transpose() * x
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
evals_1 = evals[evals > 1.0]
evecs_1 = evecs[:,evals > 1.0]
dvals = 1.0 / np.sqrt(evals_1)
transf = evecs_1 * np.diagflat(dvals)
print("Transformation Matrix = \n", transf)
transf_x = x * transf
print("The Transformed x = \n", transf_x)
xtx = transf_x.transpose() * transf_x
print("Expect an Identity Matrix = \n", xtx)

#Question 3 C
F = fData['FRAUD']
nbrs = KNeighborsClassifier(n_neighbors=5).fit(transf_x, F)
print('SCORE: ', nbrs.score(transf_x, F))


#Question 3 D
in_vals = [7500, 15, 3, 127, 2, 2] * transf
print('\nInput Variables(Transformed):  ', in_vals)
neigh = nbrs.kneighbors(in_vals, return_distance=False)
print('\nNeighbors:   ', neigh)

#Question 3 E
print('\nPredicted Value:', nbrs.predict(in_vals))
print('\nProbability:', nbrs.predict_proba(in_vals))


#Question 4 A
val3 = input('Enter Flights path: ')
Flights = pandas.read_csv(val3)
Ap2 = Flights['Airport 2']
Ap3 = Flights['Airport 3']
fig1, ax1 = plt.subplots()
ax1.set_title('Airport 3 Versus Airport 2')
plt.xlabel('Airport 2')
plt.ylabel('Airport 3')
plt.plot(Ap2, Ap3, 'o')
plt.show()


#Question 4 B
dta = Ap2.append(Ap3)
print(dta.value_counts())


#Question 4 C
keys = dta.value_counts().keys()
count_vectors = np.zeros((15, 15), dtype=int)
data_array = np.array(Flights.drop(columns=['Flight', 'Carrier 1', 'Carrier 2', 'Airport 1', 'Airport 4']))
for i in range(14):
   for k in range(14):
      if data_array[i].__contains__(keys[k]):
         count_vectors[i][k] += 1
print('Word Counts: \n' ,count_vectors)
probe =[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def CosineD(x,y):
   normX = np.sqrt(np.dot(x, x))
   normY = np.sqrt(np.dot(y, y))
   if (normX > 0.0 and normY > 0.0):
      outDistance = 1.0 - np.dot(x, y) / normX / normY
   else:
      outDistance = np.NaN
   return outDistance

Dist_arr = [0]*15
for i in range(15):
   Dist_arr[i] = CosineD(count_vectors[i, :], probe)
print('Cos Distances' ,Dist_arr)
print('Prope: ' ,probe)


