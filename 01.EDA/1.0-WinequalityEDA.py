# %% [markdown]
# ## EDA With Red Wine Data
# 
# Data Set Information:
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# 
# Attribute Information:
# 
# Input variables (based on physicochemical tests):
# - 1 - fixed acidity
# - 2 - volatile acidity
# - 3 - citric acid
# - 4 - residual sugar
# - 5 - chlorides
# - 6 - free sulfur dioxide
# - 7 - total sulfur dioxide
# - 8 - density
# - 9 - pH
# - 10 - sulphates
# - 11 - alcohol
# 
# Output variable (based on sensory data):
# - 12 - quality (score between 0 and 10)

# %%
import pandas as pd
df=pd.read_csv('DataSets/winequality-red.csv')
df.head()

# %%
## Summary of the dataset
df.info()

# %%
## descriptive summary of the dataset
df.describe()

# %%
df.shape

# %%
## List down all the columns names
df.columns

# %%
df['quality'].unique()

# %%
## Missing values in the dataset

df.isnull().sum()

# %%
## Duplicate records
df[df.duplicated()]

# %%
## Remove the duplicates
df.drop_duplicates(inplace=True)

# %%
df.shape

# %%
## Correlation
df.corr()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)

# %%
## Visualization
#conclusion- It is an imbalanced dataset
df.quality.value_counts().plot(kind='bar')
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.show()

# %%
df.head()

# %%
for column in df.columns:
    sns.histplot(df[column],kde=True)

# %%
sns.histplot(df['alcohol'])

# %%
#univariate,bivariate,multivariate analysis
sns.pairplot(df)

# %%
##categorical Plot
sns.catplot(x='quality', y='alcohol', data=df, kind="box")

# %%
df.head()

# %%
sns.scatterplot(x='alcohol',y='pH',hue='quality',data=df)

# %%


# %%


# %%


# %%


# %%



