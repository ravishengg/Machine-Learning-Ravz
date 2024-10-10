# %% [markdown]
# ## EDA And Feature Engineering Of Google Play Store Dataset
# 
# 1) Problem statement.
# Today, 1.85 million different apps are available for users to download. Android users have even more from which to choose, with 2.56 million available through the Google Play Store. These apps have come to play a huge role in the way we live our lives today. Our Objective is to find the Most Popular Category, find the App with largest number of installs , the App with largest size etc.
# 2) Data Collection.
# 
# The data consists of 20 column and 10841 rows.

# %% [markdown]
# ### Steps We Are Going to Follow
# 1. Data Clearning
# 2. Exploratory Data Analysis
# 3. Featur eEngineering

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

# %%
df=pd.read_csv('DataSets/googleplaystore.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
##summary of the dataset
df.describe()

# %%
##Missing Values
df.isnull().sum()

# %% [markdown]
# ## Insights and observation
# The dataset has msising values

# %%
df.head(2)

# %% [markdown]
# ## Data Cleaning

# %%
df['Reviews'].unique()

# %%
#Conversion of Reviews column to int gives error
#df['Reviews'].astype(int)

# %%
df['Reviews'].str.isnumeric().sum()

# %%
df[~df['Reviews'].str.isnumeric()]

# %%
df_copy=df.copy()

# %%
df_copy=df_copy.drop(df_copy.index[10472])

# %%
df_copy[~df_copy['Reviews'].str.isnumeric()]

# %%
## Convert Review Datatype to int
df_copy['Reviews']=df_copy['Reviews'].astype(int)

# %%
df_copy.info()

# %%
df_copy['Size'].unique()

# %%
19000K==19M

# %%
df_copy['Size'].isnull().sum()

# %%
df_copy['Size']=df_copy['Size'].str.replace('M','000')
df_copy['Size']=df_copy['Size'].str.replace('k','')
df_copy['Size']=df_copy['Size'].replace('Varies with device',np.nan)
df_copy['Size']=df_copy['Size'].astype(float)

# %%
df_copy['Size']

# %%
df_copy['Installs'].unique()

# %%
df_copy['Price'].unique()

# %%
chars_to_remove=['+',',','$']
cols_to_clean=['Installs','Price']
for item in chars_to_remove:
    for cols in cols_to_clean:
        df_copy[cols]=df_copy[cols].str.replace(item,'')

# %%
df_copy['Price'].unique()

# %%
df_copy['Installs'].unique()

# %%
df_copy['Installs']=df_copy['Installs'].astype('int')
df_copy['Price']=df_copy['Price'].astype('float')

# %%
df_copy.info()

# %%
## Handlling Last update feature
df_copy['Last Updated'].unique()

# %%
df_copy['Last Updated']=pd.to_datetime(df_copy['Last Updated'])
df_copy['Day']=df_copy['Last Updated'].dt.day
df_copy['Month']=df_copy['Last Updated'].dt.month
df_copy['Year']=df_copy['Last Updated'].dt.year

# %%
df_copy.info()

# %%
df_copy.head()

# %%
df_copy.to_csv('DataSets/google_cleaned.csv')

# %% [markdown]
# ## EDA
# 

# %%
df_copy.head()

# %%
df_copy[df_copy.duplicated('App')].shape

# %% [markdown]
# ## Observation
# The dataset has duplicate records

# %%
df_copy=df_copy.drop_duplicates(subset=['App'],keep='first')

# %%
df_copy.shape

# %% [markdown]
# ## Explore Data

# %%
numeric_features = [feature for feature in df_copy.columns if df_copy[feature].dtype != 'O']
categorical_features = [feature for feature in df_copy.columns if df_copy[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

# %% [markdown]
# ## 3.2 Feature Information
# 1. App :- Name of the App
# 2. Category :- Category under which the App falls.
# 3. Rating :- Application's rating on playstore
# 4. Reviews :- Number of reviews of the App.
# 5. Size :- Size of the App.
# 6. Install :- Number of Installs of the App
# 7. Type :- If the App is free/paid
# 8. Price :- Price of the app (0 if it is Free)
# 9. Content Rating :- Appropiate Target Audience of the App.
# 10. Genres:- Genre under which the App falls.
# 11. Last Updated :- Date when the App was last updated
# 12. Current Ver :- Current Version of the Application
# 13. Android Ver :- Minimum Android Version required to run the App

# %%
## Proportion of count data on categorical columns
for col in categorical_features:
    print(df[col].value_counts(normalize=True)*100)
    print('---------------------------')

# %%
## Proportion of count data on numerical columns
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df_copy[numeric_features[i]],shade=True, color='r')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()

# %% [markdown]
# ## Observations
# - Rating and Year is left skewed while Reviews,Size,Installs and Price are right skewed

# %%
# categorical columns
plt.figure(figsize=(20, 15))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
category = [ 'Type', 'Content Rating']
for i in range(0, len(category)):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=df[category[i]],palette="Set2")
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout() 

# %% [markdown]
# ## Which is the most popular app category?

# %%
df_copy.head(2)

# %%
df_copy['Category'].value_counts().plot.pie(y=df_copy['Category'],figsize=(15,16),autopct='%1.1f')

# %% [markdown]
# ## Observations
# 
# 1. There are more kinds of apps in playstore which are under category of family, games & tools
# 2. Beatuty,comics,arts and weather kinds of apps are very less in playstore

# %%
## Top 10 App Categories
category = pd.DataFrame(df_copy['Category'].value_counts())        #Dataframe of apps on the basis of category
category.rename(columns = {'Category':'Count'},inplace=True)

# %%
category

# %%
## top 10 app
plt.figure(figsize=(15,6))
sns.barplot(x=category.index[:10], y ='Count',data = category[:10],palette='hls')
plt.title('Top 10 App categories')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ## Insights
# 1. Family category has the most number of apps with 18% of apps belonging to it, followed by Games category which has 11% of the apps.
# 2. Least number of apps belong to the Beauty category with less than 1% of the total apps belonging to it.

# %% [markdown]
# ## Internal Assignments
# 1. Which Category has largest number of installations??
# 2. What are the Top 5 most installed Apps in Each popular Categories ??
# 3. How many apps are there on Google Play Store which get 5 ratings??

# %% [markdown]
# ## Which Category has largest number of installations??

# %%
df_cat_installs = df_copy.groupby(['Category'])['Installs'].sum().sort_values(ascending = False).reset_index()
df_cat_installs.Installs = df_cat_installs.Installs/1000000000# converting into billions
df2 = df_cat_installs.head(10)
plt.figure(figsize = (14,10))
sns.set_context("talk")
sns.set_style("darkgrid")

ax = sns.barplot(x = 'Installs' , y = 'Category' , data = df2 )
ax.set_xlabel('No. of Installations in Billions')
ax.set_ylabel('')
ax.set_title("Most Popular Categories in Play Store", size = 20)

# %% [markdown]
# ## Insights
# 1. Out of all the categories "GAME" has the most number of Installations.
# 2. With almost 35 Billion Installations GAME is the most popular Category in Google App store

# %% [markdown]
# ## What are the Top 5 most installed Apps in Each popular Categories ??

# %%
dfa = df_copy.groupby(['Category' ,'App'])['Installs'].sum().reset_index()
dfa = dfa.sort_values('Installs', ascending = False)
apps = ['GAME', 'COMMUNICATION', 'PRODUCTIVITY', 'SOCIAL' ]
sns.set_context("poster")
sns.set_style("darkgrid")

plt.figure(figsize=(40,30))

for i,app in enumerate(apps):
    df2 = dfa[dfa.Category == app]
    df3 = df2.head(5)
    plt.subplot(4,2,i+1)
    sns.barplot(data= df3,x= 'Installs' ,y='App' )
    plt.xlabel('Installation in Millions')
    plt.ylabel('')
    plt.title(app,size = 20)
    
plt.tight_layout()
plt.subplots_adjust(hspace= .3)
plt.show()

# %% [markdown]
# ## Insights
# - Most popular game is Subway Surfers.
# - Most popular communication app is Hangouts.
# - Most popular productivity app is Google Drive.
# - Most popular social app is Instagram.

# %% [markdown]
# ## How many apps are there on Google Play Store which get 5 ratings??

# %%
rating = df_copy.groupby(['Category','Installs', 'App'])['Rating'].sum().sort_values(ascending = False).reset_index()

toprating_apps = rating[rating.Rating == 5.0]
print("Number of 5 rated apps",toprating_apps.shape[0])
toprating_apps.head(1)

# %% [markdown]
# ## Result
# - There are 271 five rated apps on Google Play store
# - Top most is 'CT Brain Interpretation' from 'Family' Category

# %%
df_copy.head()

# %%



