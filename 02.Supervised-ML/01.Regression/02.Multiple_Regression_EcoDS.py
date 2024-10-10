# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# %%
df_index=pd.read_csv("DataSets/economic_index.csv")

# %%
df_index.head()

# %%
#Drop unwanted columns
df_index.drop(columns=["Unnamed: 0","year","month"],axis=1,inplace=True)
df_index.head()

# %%
#Check NULL values
df_index.isnull().sum()

# %%
#Lets do some data visualization
import seaborn as sns
sns.pairplot(df_index)

# %%
df_index.corr()

# %%
#Visualize datapoints more closely
plt.plot(df_index['interest_rate'],df_index['unemployment_rate'],color='r',marker='o')

# %%
#independent and dependant features
x=df_index.iloc[:,:-1] #All exceot last columns
y=df_index.iloc[:,-1] #Last Column

# %%
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

# %%
import seaborn as sns
sns.regplot(x=df_index['interest_rate'],y=df_index['index_price'])

# %%
sns.regplot(x=df_index['interest_rate'],y=df_index['unemployment_rate'])

# %%
sns.regplot(x=df_index['index_price'],y=df_index['unemployment_rate'])

# %%
#Standardization
from sklearn.preprocessing import StandardScaler

# %%
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

# %%
#Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression

# %%
regression=LinearRegression(n_jobs=-1)
regression.fit(x_train,y_train)

# %%
'''
Evaluate a score by cross-validation
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
https://scikit-learn.org/stable/modules/model_evaluation.html
'''

from sklearn.model_selection import cross_val_score
validation_score=cross_val_score(regression,x_train,y_train,scoring='neg_mean_squared_error',cv=3)

# %%
np.mean(validation_score)

# %%
#prediction
#Prediction for test data
y_pred=regression.predict(x_test)

# %%
#Performance Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

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

# %% [markdown]
# Assumptions

# %%
#If there is a linear relationship, that basically means your model has performed well.
plt.scatter(y_test,y_pred)


# %%
#Plot the residuals, if the plot shows normal distribution, than model is good
residuals=y_test-y_pred
sns.displot(residuals,kind='kde')

# %%
#Scatter plot with respect to predictions and residual
plt.scatter(y_pred,residuals)

# %%
##OLS Linear Regression
import statsmodels.api as sm
model=sm.OLS(y_train,x_train).fit()

prediction=model.predict(x_test)
print(model.summary())

# %%
print(regression.coef_)

# %%
#Prediction for new data
pred_index=regression.predict(scaler.transform([[0.8,11]]))
print(pred_index)


