'''
####################################################################################################
Train/Test. Model Evaluation. In ML we have models to predict outcomes of certain events and want to measure if they are good enough.
####################################################################################################
'''

# Train/Test is a method to measure the accuracy of models.
# Creating a data set representing customers in a shop with their habits.

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2) # Keeps the random values we obtained

x = numpy.random.normal(3, 1, 100) # representing number of minutes before making a purchase
y = numpy.random.normal(150, 40, 100) / x # Money spent on the purchase (mean of 150 and std of 40)

plt.scatter(x, y)
plt.show()

# Now we split it into a training and test set, typically 80:20
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show() # plot of the training data. Notice it will look similar, just slightly less entry points.

"""From looking at the graph it seems polynomial regression would be the best fit"""

# this creates a poly function (poly1d) and passes through the coefficients from polyfit.
# the function is to the 4th degree (line goes down,up,down,up) for more complex curves which is like this : f(4) = a(4)^4 + b(4)^3 + c(4)^2 + d(4) + e
# the above finds the average spend at 4 minutes, we plug 4 into each degree and e is the intercept
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0,6,100) # create my line starting at 0, stopping at 6 and bring 100 points, for smoother curve

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

"""We can see from our graph the line extends up from 6, this shows people that stay 6 minutes spend over 200? this is a sign of overfitting"""
# Here we can now use r2_score to see how well this fits.

from sklearn.metrics import r2_score
print(r2_score(train_y, mymodel(train_x))) # 0.79, which shows that there is an OK relationship
# above uses target data y (what really happened) against our predicted data (from our model)

"""Now is our time to test"""
print(r2_score(test_y, mymodel(test_x))) # 0.80 shows the model fits the testing set also. So we are confident we can use this to predict future values.


print(mymodel(5)) # predict the spend when a customer spends 5 minutes in the shop