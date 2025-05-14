"""
####################################################################################################
Decision Trees.
####################################################################################################
"""

# DTs make predictions by following a tree like structure where each internal node asks a
# question about a feature and each leaf node provides a prediction or value. Classification and regression


# We have a data set which is called DT data represents every time there was a comedy show in town and some information about the comedian,
# along with whether they went or not.
import pandas
from pyexpat import features

import pandas as pd

df = pandas.read_csv("DataSets/DTdata.csv")

print(df.head())

'''For a decision tree all data has to be numerical. We will use pandas map() method to map strings to values'''

d = {'UK': 0, 'USA': 1, 'N':2} # this converts the respected sting to the associated numerical value.
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
print(df.head())

# Now we import the modules we need:
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

features = ['Age', 'Experience', 'Rank', 'Nationality'] # our features

X = df[features] # we want these features to determine our y
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

print(dtree.predict([[40,10,7,1]])) # lets predict a 40 y/o with 10 years experience ranked 7 and is from USA

# a cleaner version of this input without the warning would be like this.
input_data = pd.DataFrame([[40,10,7,1]], columns=['Age', 'Experience', 'Rank', 'Nationality'])
print(dtree.predict(input_data))

'''Note that this will give you different answers over the same data... this is as it is based on probability. important to remember that DT will not give us a 100% certain answer'''

plot_tree(dtree) # this is to print the decision tree
plt.show()