"""
KNN stands for K Nearest Neighbors. This is a basic algorithm which makes predictions based on the distance
between data points (samples) used for both classification and regression problems.

- For classification: it predicts the class by majority vote among the k neighbors.
- For regression: it predicts the target by averaging the values of the k neighbors.

Important:
- This works best when the data is standardised. We can use StandardScaler().
- Model complexity is controlled by the choice of 'k':
    - Low k = more flexible, more variance (risk of overfitting)
    - High k = smoother decision boundary, more bias (risk of underfitting)

Looks at every X_test data point and finds the K nearest neighbours in X_train.
Then predicts the target by taking a majority vote (Classification) or average of y_train values (Regression).

Simple example with K = 3:

We look at every x test point,
find the 3 nearest x train values,
look at their y train labels, if 2/3 are true,
then that x test point is true.
"""


"""
Here we will use a Telecomms provider data set where they have divided their customer base by service usage patters, into 4 groups.
This data is from IBM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
# print(df.head())

print(df['custcat'].value_counts()) # The amount of each category

### We can say that we have records of
# 281 customers who opt for Plus Services,
# 266 for Basic-services,
# 236 for Total Services,
# 217 for E-Services.
# It is seen that the data set is mostly balanced between the different classes and requires no special means of accounting for class bias.

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()

# As we can see from the matrix, some features have better correlation than others.
# The most important for us is the correlation between 'custcat' with the other features.

# This shows us the correlation between custcat and others. Note we drop need custcat here.
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)

### Get our data ready ###
X = df.drop('custcat', axis=1) # features
y = df['custcat'] # target

### Normalise data ### This is intuitive, If data points are greatly seperated, numerically, results can be very inaccurate.
X_norm = StandardScaler().fit_transform(X)

### Train Test Split ###
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size =0.2, random_state=4)

### KNN Classification ###
k = 4
# Train model and predict
knn_classifer = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifer.fit(X_train, y_train)

'Predicting'
yhat = knn_model.predict(X_test)

'Accuracy evaluation'
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

"""
Now We need to find out what the most suitable 'K' is
"""

Ks = 100 # look through k=1 to k=100
acc = np.zeros((Ks)) # creates an empty array to store the accuracy of each K
std_acc = np.zeros((Ks)) # creates an empty array to store the std deviation (error) of each K
for n in range(1, Ks + 1):
    # Now train the model and predict
    knn_model_n = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


"""
Plot the model accuracy for a different number of neighbors.
Now, you can plot the model accuracy and the standard deviation to identify the model with the most suited value of k.
"""

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) # argmax returns the index of the max value in acc