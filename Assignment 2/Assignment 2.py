# Load the necessarty libraries
from mlxtend.preprocessing import TransactionEncoder
import pandas

Imaginary_Store = pandas.read_csv('C:\\Users\\galla\\CS484\\CS-484\\Assignment 2\\Groceries.csv',
                                  delimiter=',')

# Convert the Sale Receipt data to the Item List format
ListItem = Imaginary_Store.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

####################
####   Part A   ####
####################

# Calculate the frequency table of number of customers per item
nCustomerPurchase = Imaginary_Store.groupby('Item').size()
freqTable = pandas.Series.sort_index(pandas.Series.value_counts(nCustomerPurchase))
#print('Frequency of Number of Customers Purchase Item')
#print(freqTable)

# Calculate the frequency table of number of items purchase
nItemPurchase = Imaginary_Store.groupby('Customer').size()
freqTable = pandas.Series.sort_index(pandas.Series.value_counts(nItemPurchase))
print('Frequency of Number of Items Purchase')
print(freqTable)

####################
####   Part B   ####
####################

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.01, max_len = 25, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("List of Association Rules")
print(assoc_rules)

####################
####   Part C   ####
####################

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

####################
####   Part D   ####
####################

assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.58)
ar = assoc_rules.drop(columns=['antecedent support', 'consequent support', 'leverage', 'conviction'])
print('Association Rules with Confidence >= 58%')
print(ar)


'''# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.1, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.8)

assoc_rules['lift'].describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()'''