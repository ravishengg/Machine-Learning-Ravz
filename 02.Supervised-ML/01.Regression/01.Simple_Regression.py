# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# %%
df=pd.read_csv("DataSets/SOCR-HeightWeight.csv")

# %%
df.drop(columns=['Index'],inplace=True)
df.head()

# %%
plt.scatter(df['Weight(Pounds)'],df['Height(Inches)'])
plt.xlabel('Weight(Pounds)')
plt.ylabel('Height(Inches)')

# %%
#Correlation
df.corr()

# %%
import seaborn as sns
sns.pairplot(df)

# %%
#Independent and dependent Features
x=df[['Weight(Pounds)']]  #Independent feature Should be a dataframe or 2D-Array
y=df['Height(Inches)']  #Dependendent feature can be series or 1D-Array

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

# %%
#Standardization
from sklearn.preprocessing import StandardScaler

# %%
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)

# %%
x_test = scaler.transform(x_test)

# %%
#Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression

# %%
regression=LinearRegression(n_jobs=-1)
regression.fit(x_train,y_train)

# %%
print("Coefficient or slope: ",regression.coef_)
print("Intercept:",regression.intercept_)

# %%
##Plot Training Data Best Fit Line
plt.scatter(x_train,y_train)
plt.plot(x_train,regression.predict(x_train))

# %%
'''Prediction of Test Data
1. Predicted Height = intercept + Coef_(Weight)
2. y_pred_test = 68.20216961538462 + 1.11028361(x_test)
'''

# %%
#Prediction for test data
y_pred=regression.predict(x_test)

# %%
#Performance Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,a

# %%
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
R2Scr=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("mean_absolute_error:",mae)
print("Root Mean Squared Error:",rmse)
print("R2Square:",R2Scr)
'''Adjusted R Square
1 - [(1 - R2)(n - 1) / (n - p - 1)]
'''
Adj_R2Scr = 1 - (1 - R2Scr) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
print("Adjusted R Square:",Adj_R2Scr)

# %%
##OLS Linear Regression
import statsmodels.api as sm

# %%
model=sm.OLS(y_train,x_train).fit()

# %%
prediction=model.predict(x_test)

# %%
print(model.summary())

# %%
#Prediction for new data
height=regression.predict(scaler.transform([[142]]))/12
print(height)
print(height*12)



