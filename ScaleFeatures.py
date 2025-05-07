'''
####################################################################################################
Scale Features.
####################################################################################################
'''

# Simply put, this is how we scale data into new values so they are easier to compare. For example, what is a kg compared to a meter?

# Standardisation Formula : z = (x-u) / s.  where: x is original value, u is mean and s is standard deviation.

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# let's use the same data as before but the volume as liters instead of cm^3.
df = pandas.read_csv('DataSets/data.csv')
df_liters = df.copy()
df_liters['Volume'] = df_liters['Volume'] / 1000
# print(df_liters.head()) # prints the first 5 rows

X = df_liters[['Weight', 'Volume']]
y = df_liters['CO2']

scaledX = scale.fit_transform(X)
print(scaledX)

# Note the first two values. Following out forumla, (790 - 1292.23) / 238.74 = -2.1 etc.

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)
