# %%
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
#The function np.random.rand() in NumPy is specifically designed to generate random numbers that are uniformly distributed between 0 (inclusive) and 1 (exclusive)

X = 6 * np.random.rand(100, 1) - 3
y =0.5 * X**2 + 1.5*X + 2 + np.random.randn(100, 1)
# quadratic equation used- y=0.5x^2+1.5x+2+outliers
plt.scatter(X,y,color='g')
plt.xlabel('X Dataset')
plt.ylabel('y Dataset')


# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

# %%
from sklearn.linear_model import LinearRegression
regression_1 = LinearRegression()

# %%
regression_1.fit(X_train, y_train)

# %%
from sklearn.metrics import r2_score
score=r2_score(y_test,regression_1.predict(X_test))
print(score)



# %%
#Lets visualize this model
plt.plot(X_train,regression_1.predict(X_train),color='r')
plt.scatter(X_train,y_train)
plt.xlabel("X Dataset")
plt.ylabel("Y")
'''
Too much scattered and doesnt fit in Simple Linear Regression
'''

# %%
#Lets apply polynomial transformation
from sklearn.preprocessing import PolynomialFeatures

# %%
poly=PolynomialFeatures(degree=2,include_bias=True)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)


# %%
from sklearn.metrics import r2_score
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score=r2_score(y_test,y_pred)
print(score)

# %%
plt.scatter(X_train,regression.predict(X_train_poly))
plt.scatter(X_train,y_train)

# %%
#Prediction of new data set

''' 
np.linspace() is a NumPy function that generates an array of evenly spaced values
.reshape(200, 1) reshapes the 1D array generated by np.linspace() into a 2D array with 200 rows and 1 column.
This type of array is often used in regression models or plotting, where you might want to predict or plot values over a specific range (e.g., to visualize a model's predictions across a range of inputs).
'''

X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
X_new_poly

# %%
y_new = regression.predict(X_new_poly)
plt.plot(X_new, y_new, "r-", linewidth=2, label=" New Predictions")
plt.plot(X_train, y_train, "b.",label='Training points')
plt.plot(X_test, y_test, "g.",label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# %% [markdown]
# Pipeline Concepts

# %% [markdown]
# 

# %%
''' 
code defines a function poly_regression(degree) that performs polynomial regression on a given dataset, fits a model, and plots the resulting regression line along with the training and test data points.
'''
from sklearn.pipeline import Pipeline
def poly_regression(degree):
    X_new = np.linspace(-3, 3, 200).reshape(200, 1)
    
    poly_features=PolynomialFeatures(degree=degree,include_bias=True)
    lin_reg=LinearRegression()
    poly_regression=Pipeline([
        ("poly_features",poly_features),
        ("lin_reg",lin_reg)
    ])
    poly_regression.fit(X_train,y_train) ## ploynomial and fit of linear reression
    y_pred_new=poly_regression.predict(X_new)
    #plotting prediction line
    plt.plot(X_new, y_pred_new,'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-4,4, 0, 10])
    plt.show()

# %%
poly_regression(1)

# %%
poly_regression(2)

# %%
poly_regression(3)

# %%
poly_regression(4)

# %%
poly_regression(5)

# %%
poly_regression(6)

# %%



