# %% [markdown]
# ## Algerian Forest Fires Dataset 
# Data Set Information:
# 
# The dataset includes 244 instances that regroup a data of two regions of Algeria,namely the Bejaia region located in the northeast of Algeria and the Sidi Bel-abbes region located in the northwest of Algeria.
# 
# 122 instances for each region.
# 
# The period from June 2012 to September 2012.
# The dataset includes 11 attribues and 1 output attribue (class)
# The 244 instances have been classified into fire(138 classes) and not fire (106 classes) classes.

# %% [markdown]
# Attribute Information:
# 
# 1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
# Weather data observations
# 2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 3. RH : Relative Humidity in %: 21 to 90
# 4. Ws :Wind speed in km/h: 6 to 29
# 5. Rain: total day in mm: 0 to 16.8
# FWI Components
# 6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 8. Drought Code (DC) index from the FWI system: 7 to 220.4
# 9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 11. Fire Weather Index (FWI) Index: 0 to 31.1
# 12. Classes: two classes, namely Fire and not Fire

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

# %%
dataset=pd.read_csv('DataSets/Algerian_forest_fires_dataset_UPDATE.csv' ,header=1)

# %%
dataset.head()

# %%
dataset.info()

# %% [markdown]
# ## Data Cleaning

# %%
## missing values
dataset[dataset.isnull().any(axis=1)]

# %% [markdown]
# The dataset is converted into two sets based on Region from 122th index, we can make a new column based on the Region
# 
# 1 : "Bejaia Region Dataset"
# 
# 2 : "Sidi-Bel Abbes Region Dataset"
# 
# Add new column with region

# %%
dataset.loc[:122,"Region"]=0
dataset.loc[122:,"Region"]=1
df=dataset

# %%
df.info()

# %%
df[['Region']]=df[['Region']].astype(int)

# %%
df.head()

# %%
df.isnull().sum()

# %%
## Removing the null values
df=df.dropna().reset_index(drop=True)


# %%
df.head()

# %%
df.isnull().sum()

# %%
df.iloc[[122]]

# %%
##remove the 122nd row
df=df.drop(122).reset_index(drop=True)

# %%
df.iloc[[122]]

# %%
df.columns

# %%
## fix spaces in columns names
df.columns=df.columns.str.strip()
df.columns

# %%
df.info()

# %% [markdown]
# #### Changes the required columns as integer data type

# %%
df.columns

# %%
df[['month','day','year','Temperature','RH','Ws']]=df[['month','day','year','Temperature','RH','Ws']].astype(int)

# %%
df.info()

# %% [markdown]
# #### Changing the other columns to float data datatype
# 

# %%
objects=[features for features in df.columns if df[features].dtypes=='O']

# %%
for i in objects:
    if i!='Classes':
        df[i]=df[i].astype(float)

# %%
df.info()

# %%
objects

# %%
df.describe()

# %%
df.head()

# %%
## Let save the cleaned dataset
df.to_csv('Algerian_forest_fires_cleaned_dataset.csv',index=False)

# %% [markdown]
# ##  Exploratory Data Analysis

# %%
## drop day,month and year
df_copy=df.drop(['day','month','year'],axis=1)

# %%
df_copy.head()

# %%
## categories in classes
df_copy['Classes'].value_counts()

# %%
## Encoding of the categories in classes
df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)

# %%
df_copy.head()

# %%
df_copy.tail()

# %%
df_copy['Classes'].value_counts()

# %%
## Plot desnity plot for all features
plt.style.use('seaborn')
df_copy.hist(bins=50,figsize=(20,15))
plt.show()

# %%
## Percentage for Pie Chart
percentage=df_copy['Classes'].value_counts(normalize=True)*100

# %%
# plotting piechart
classlabels=["Fire","Not Fire"]
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
plt.title("Pie Chart of Classes")
plt.show()

# %%
## Correlation

# %%
df_copy.corr()

# %%
sns.heatmap(df.corr())

# %%
## Box Plots
sns.boxplot(df['FWI'],color='green')

# %%
df.head()

# %%
df['Classes']=np.where(df['Classes'].str.contains('not fire'),'not fire','fire')

# %%
## Monthly Fire Analysis
dftemp=df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel('Number of Fires',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title("Fire Analysis of Sidi- Bel Regions",weight='bold')

# %%
## Monthly Fire Analysis
dftemp=df.loc[df['Region']==0]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel('Number of Fires',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title("Fire Analysis of Brjaia Regions",weight='bold')

# %% [markdown]
# Its observed that August and September had the most number of forest fires for both regions. And from the above plot of months, we can understand few things
# 
# Most of the fires happened in August and very high Fires happened in only 3 months - June, July and August.
# 
# Less Fires was on September

# %%


# %%



