# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy
import pandas

# Define a function to visualize the percent of a particular target category by an interval predictor
def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   dataTable = inData
   dataTable['LE_Split'] = (dataTable.iloc[:,0] <= split)

   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * numpy.log2(proportion)
      print('Row = ', iRow, 'Entropy =', rowEntropy)
      print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)

cars = pandas.read_csv('C:\\Users\\galla\\CS-484-Good-2\\Assignment 3\\claim_history.csv', delimiter=',')
key = [len(cars)]
j = 0
for i in range(len(cars)):
    if (cars['EDUCATION'][i] == 'Below High School'):
        cars['EDUCATION'][i] = 1
    elif (cars['EDUCATION'][i] == 'High School'):
        cars['EDUCATION'][i] = 2
    elif (cars['EDUCATION'][i] == 'Bachelors'):
        cars['EDUCATION'][i] = 3
    elif (cars['EDUCATION'][i] == 'Masters'):
        cars['EDUCATION'][i] = 4
    else:
        cars['EDUCATION'][i] = 5

print(cars['EDUCATION'])
print(key)


inData2 = cars[['EDUCATION', 'CAR_USE']].dropna()

# Horizontal frequency bar chart of Cylinders
#inData2.groupby('EDUCATION').size().plot(kind='barh')

# Horizontal frequency bar chart of Cylinders
#inData2.groupby('CAR_USE').size().plot(kind='barh')

'''crossTable = pandas.crosstab(index = inData2['EDUCATION'], columns = inData2['CAR_USE'],
                             margins = True, dropna = True)   
print(crossTable)'''

# Split (None), (3, 4, 5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 0.5)
print('Split Entropy = ', EV)

# Split (3), (4, 5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 1.5)
print('Split Entropy = ', EV)

# Split (3, 4), (5, 6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 2.5)
print('Split Entropy = ', EV)

# Split (3, 4, 5), (6, 8, 10, 12)
EV = EntropyIntervalSplit(inData2, 3.5)
print('Split Entropy = ', EV)

# Split (3, 4, 5, 6), (8, 10, 12)
EV = EntropyIntervalSplit(inData2, 4.5)
print('Split Entropy = ', EV)


RightBranch = inData2[inData2['EDUCATION'] > 1.5]
print('RIGHT')
# Split (8), (10, 12)
EV = EntropyIntervalSplit(RightBranch, 2.5)
print('Split Entropy = ', EV)

# Split (8, 10), (12)
EV = EntropyIntervalSplit(RightBranch, 3.5)
print('Split Entropy = ', EV)

EV = EntropyIntervalSplit(RightBranch, 4.5)
print('Split Entropy = ', EV)

# Load the TREE library from SKLEARN
'''from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
cars_DT = classTree.fit(inData2[['EDUCATION']], inData2['CAR_USE'])
DT_accuracy = classTree.score(inData2[['EDUCATION']], inData2['CAR_USE'])

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(DT_accuracy))'''

#import graphviz
#dot_data = tree.export_graphviz(cars_DT,
#                                out_file=None,
#                                impurity = True, filled = True,
#                                feature_names = ['EDUCATION'],
#                                class_names = ['Asia', 'Europe', 'USA'])

#graph = graphviz.Source(dot_data)
#graph

#graph.render('C:\\IIT\\Machine Learning\\Job\\cars_DT_output')
