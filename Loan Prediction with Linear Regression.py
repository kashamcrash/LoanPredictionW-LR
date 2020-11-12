#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Agenda
# A banking institution requires actionable insights from the perspective of Mortgage-Backed Securities, 
# Geographic Business Investment and Real Estate Analysis for USA

# Task in hand
# Identify potential monthly mortgage expenses for each region based on monthly family income and rental of the real estate
# Create a model to predict the potential demand in dollars amount of loan for each of the region in the USA

# Credentials
# kasham1991@gmail.com | Karan Sharma


# In[2]:


# Importing the basic libraries

import pandas as pd
import numpy as np
import time
import random
from math import *
import operator


# In[3]:


# Loading the datasets

train = pd.read_csv("C:\\Datasets\\loan_train.csv")
test = pd.read_csv("C:\\Datasets\\loan_test.csv")


# In[4]:


# Looking at the basics of the data

print("Columns of train data")
print("---------------------")
print(list(train.columns))
print("---------------------")
print("Columns of test data")
print("---------------------")
print(list(test.columns))

#len(train)
#len(test)


# In[5]:


# Both the datasets have significant Nan values and categorical values
# All values in blockid are NaN or 0
# UID is unique identification number

train.head()
train.info()
train.describe()
#test.head()
#test.info
#test.describe()


# In[6]:


# Indexing on the basis of UID as it is the unique identification number
# Setting the DataFrame index using existing columns

train.set_index(keys = ['UID'], inplace = True)
test.set_index(keys = ['UID'], inplace = True)

train.head()
#test.head()


# In[7]:


# Looking for missisng values/NaN
# Deriving a percentage value for missing values, creating a new column for easy understanding
# As seen below, blockid has only NaN; must be dropped

missing_list_train = train.isnull().sum() *100/len(train)
missing_values_train = pd.DataFrame(missing_list_train, columns=['Percantage of missing values'])
missing_values_train.sort_values(by = ['Percantage of missing values'], inplace = True, ascending=False)
missing_values_train[missing_values_train['Percantage of missing values'] >0][:10]


# In[8]:


# Applying the same for test dataset

missing_list_test = test.isnull().sum() *100/len(test)
missing_values_test = pd.DataFrame(missing_list_test, columns = ['Percantage of missing values'])
missing_values_test.sort_values(by = ['Percantage of missing values'], inplace = True, ascending = False)
missing_values_test[missing_values_test['Percantage of missing values'] >0][:10]


# In[9]:


# Dropping blockid and sumlevel
# Sum level holds no statistical value

train.drop(columns = ['BLOCKID','SUMLEVEL'], inplace = True)
test.drop(columns = ['BLOCKID', 'SUMLEVEL'], inplace = True)


# In[10]:


# Imputing the missing values with mean
# Creating an empyt list full of missing values

missing_train_cols = []
for col in train.columns:
    if train[col].isna().sum() !=0:
         missing_train_cols.append(col)
print(missing_train_cols)


# In[11]:


# Applying the same in test set

missing_test_cols = []
for col in test.columns:
    if test[col].isna().sum() !=0:
         missing_test_cols.append(col)
print(missing_test_cols)


# In[12]:


# Replacing with mean for both train and test dataset
# Mean is good statistical substitue 

for col in train.columns:
    if col in (missing_train_cols):
        train[col].replace(np.nan, train[col].mean(), inplace = True)

for col in test.columns:
    if col in (missing_test_cols):
        test[col].replace(np.nan, test[col].mean(), inplace = True)


# In[13]:


# Checking the new count

print(train.isna().sum().sum())
print(test.isna().sum().sum())


# In[14]:


# Time for debt analysis
# Debt analysis determines the proportion of loan accounts for a particular household

# Importing the required libraries for plotting

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
sns.set(style = "white", color_codes = True)
sns.set(font_scale = 1)


# In[15]:


# Exploring the top 2,500 locations where the percentage of households with 
# a second mortgage is the highest and percent ownership is above 10 percent
# Keeping the upper limit for the percent of households with a second mortgage to 50 percent
# We are utilizing SQL on pandas dataframe with pandasql

from pandasql import sqldf
q1 = "select place,pct_own,second_mortgage,lat,lng from train where pct_own >0.10 and second_mortgage <0.5 order by second_mortgage DESC LIMIT 2500;"
pysqldf = lambda q: sqldf(q, globals())
train_location_mort_pct = pysqldf(q1)

train_location_mort_pct.head()


# In[16]:


# Plotting the figure for the same
# Visualizing using geo-map with latitude and longitude across USA

fig = go.Figure(data = go.Scattergeo(
    lat = train_location_mort_pct['lat'],
    lon = train_location_mort_pct['lng']),
    )
fig.update_layout(
    geo = dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation_lon = -100
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range = [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range = [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title ='Top 2,500 Locations Where Second Mortgage is the Highest and Percent Ownership is Above 10 Percent')
fig.show()


# In[17]:


# Calculating the amount of bad debt with the following equation
# Bad Debt = P (Second Mortgage ∩ Home Equity Loan) Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage

train['bad_debt'] = train['second_mortgage'] + train['home_equity'] - train['home_equity_second_mortgage']


# In[18]:


# Creating a pie chart to showcase overall debt and bad debt 
# Creating bins for the same

#train['bins'] = pd.cut(train['bad_debt'], bins = [0,0.10,1], labels = ["less than 50%","50-100%"])
#train.groupby(['bins']).size().plot(kind = 'pie', subplots = True, startangle = 90, autopct = '%1.1f%%')
#plt.axis('equal')
#plt.show()


# In[19]:


# Alalysing the distribution for the second mortgage, good debt, bad debt, home equity for different cities
# Creating box and whisker plot for the same

cols = []
#train.columns


# In[20]:


# Looking at the data for the following cities; hamilton and manhattan
# Seperating the relevant columns by city

cols = ['second_mortgage','home_equity','debt','bad_debt']
hamilton = train.loc[train['city'] == 'Hamilton']
manhattan = train.loc[train['city'] == 'Manhattan']
box_city = pd.concat([hamilton, manhattan])
box_city.head()


# In[21]:


# Plotting the same for second mortgage
# Outliers are present in hamilton

plt.figure(figsize = (7, 3))
sns.boxplot(data = box_city, x = 'second_mortgage', y = 'city', width = 0.5, palette = "plasma_r")
plt.show()


# In[22]:


# Plotting the same for home quity

plt.figure(figsize = (7, 3))
sns.boxplot(data = box_city, x = 'home_equity', y = 'city', width = 0.5, palette = "plasma_r")
plt.show()


# In[23]:


# Plotting the same for debt

plt.figure(figsize = (7, 3))
sns.boxplot(data = box_city, x = 'debt', y = 'city', width = 0.5, palette = "plasma_r")
plt.show()


# In[24]:


# Plotting the same for bad_debt
# We can clearly notice from all of the plots that manhattan has higher metrics than hamilton

plt.figure(figsize = (7, 3))
sns.boxplot(data = box_city, x = 'bad_debt', y = 'city', width = 0.5, palette = "plasma_r")
plt.show()


# In[25]:


# Creating an income distribution chart for family income, house hold income, and remaining income

plt.figure(figsize = (10, 4))
sns.distplot(train['hi_mean'], color = "g")
plt.title('Household Income Distribution Chart')
plt.show()


# In[26]:


plt.figure(figsize = (10, 4))
sns.distplot(train['family_mean'], color = "y")
plt.title('Family Income Distribution Chart')
plt.show()


# In[27]:


# Remaining income is family mean income subtracted by mean household income
# From all the charts whe can clearly notice normal distibution in income

plt.figure(figsize = (10, 4))
sns.distplot(train['family_mean'] - train['hi_mean'], color = "c")
plt.title('Remaining Income Distribution Chart')
plt.show()


# In[28]:


# Looking into population density and age
# Visualizing the same for population
# Data is skewed to the right side; positive skewness

plt.figure(figsize = (50, 25))
fig,(ax1,ax2,ax3) = plt.subplots(3, 1)
sns.distplot(train['pop'], ax = ax1, color = "g")
sns.distplot(train['male_pop'], ax = ax2, color = "y")
sns.distplot(train['female_pop'], ax = ax3, color = "c")
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
plt.tight_layout()
plt.show()


# In[29]:


# Visualizing the same for age
# Data is normally distributed

plt.figure(figsize = (50, 25))
fig,(ax1, ax2) = plt.subplots(2, 1)
sns.distplot(train['male_age_mean'], ax = ax1, color = "g")
sns.distplot(train['female_age_mean'], ax = ax2, color = "y")
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
plt.tight_layout()
plt.show()


# In[30]:


# Creating a new field - population density from aland and pop
# Plotting the same; very less density is noticed

train['pop_density'] = train['pop']/train['ALand']
test['pop_density'] = test['pop']/test['ALand']

plt.figure(figsize = (10, 4))
sns.distplot(train['pop_density'], color = 'c')
plt.title('Population Density')
plt.show() 


# In[31]:


# Creating a new field - median age from male_age_median, female_age_median, male_pop, and female_pop
# Age of population is mostly between 20 and 60
# Majority are of age around 40 to 50
# Median age distribution follows a normal distribution
# Some right skewness is noticed

train['age_median'] = (train['male_age_median'] + train['female_age_median'])/2
test['age_median'] = (test['male_age_median'] + test['female_age_median'])/2

plt.figure(figsize = (10, 4))
sns.distplot(train['age_median'], color = 'c')
plt.title('Median Age')
plt.show()


# In[32]:


# Lets create a box plot for the two varibales created above

plt.figure(figsize = (10, 3))
sns.boxplot(train['age_median'], width = 0.5, palette = "Spectral" )
plt.title('Population Density')
plt.show() 


# In[33]:


plt.figure(figsize = (10, 3))
sns.boxplot(train['pop_density'], width = 0.5, palette = "Spectral" )
plt.title('Population Density')
plt.show() 


# In[34]:


# Creating bins for population into a new variable by selecting appropriate class interval 
# so that the no of categories(bins) don’t exceed 5 for the ease of analysis

train['pop'].describe()


# In[35]:


# Creating bins for population on the basis of 'very low','low','medium','high','very high'
# Using the cut function and a bin interval of 5 

train['pop_bins']= pd.cut(train['pop'], bins = 5, labels = ['very low','low','medium','high','very high'])
train[['pop','pop_bins']]

train['pop_bins'].value_counts()


# In[36]:


# Analysing the married, separated and divorced population for these population brackets
# Grouping the married, separated and divorced population by pop_bins on the basis of count
# Determining the mean and median for the same
# In very high population groups, there are more married people and less percantage of separated/divorced couples
# In very low population groups, there are more divorced people

train.groupby(by = 'pop_bins')[['married','separated','divorced']].count()
train.groupby(by = 'pop_bins')[['married','separated','divorced']].agg(["mean", "median"])


# In[37]:


# Lets visualize the same for easy analysis
# We can't determine much from the chart below

pop_bin_married = train.groupby(by = 'pop_bins')[['married','separated','divorced']].agg(["mean"])
pop_bin_married.plot(figsize = (15, 6))
plt.legend(loc = 'best')
plt.show()


# In[38]:


# Creating a new variable for rent as a percentage of income at an overall level, and for different states
# Grouping the mean rent on the basis of state

rent_state_mean = train.groupby(by = 'state')['rent_mean'].agg(["mean"])
rent_state_mean.head()


# In[39]:


# Applying the same for income on the basis of state

income_state_mean = train.groupby(by = 'state')['family_mean'].agg(["mean"])
income_state_mean.head()


# In[40]:


# Calculating the rent percent of the overall income

rent_perc_of_income = rent_state_mean['mean']/income_state_mean['mean']
rent_perc_of_income.head()


# In[41]:


# Overall rent as a percentage of income

sum(train['rent_mean'])/sum(train['family_mean'])


# In[42]:


# Performing correlation analysis for all the relevant variables 
# Lets look at the relevant columns

train.columns
r = train[['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']].corr()


# In[43]:


# There is a high positive correaltion between pop, male_pop and female_pop
# There is a high positive correlation between rent_mean,hi_mean, family_mean,hc_mean

plt.figure(figsize = (20, 10))
sns.heatmap(r, annot = True, cmap = 'cubehelix', linewidths = .20)
plt.show()


# In[44]:


# Problem in hand

# The economic multivariate data has a significant number of measured variables. 
# The goal is to find where the measured variables depend on a number of smaller unobserved common factors or latent variables. 
# Each variable is assumed to be dependent upon a linear combination of the common factors
# and the coefficients are known as loadings
# Each measured variable also includes a component due to independent random variability, 
# known as “specific variance” because it is specific to one variable. 
# Obtain the common factors and then plot the loadings
# Use factor analysis to find latent variables in our dataset and gain insight into the linear relationships in the data. 
# Following are the list of latent variables:Highschool graduation rates 
# • Median population age • Second mortgage statistics • Percent own • Bad debt expense


# In[45]:


# Lets utilize factor analysis
# Factor analysis helps in identifying latent/hidden factors that have a significant effect on the variables
# Selecting 5 number of factors
# Excluding the obejct and category from the dataset

# Courtesy of Charles Zaiontz
# One of the main objectives of factor analysis is to reduce the number of parameters. 
# The number of parameters in the original model is equal to the number of unique elements in the covariance matrix.
# The factors which have a high eigenvalue should be retained, 
# while those with a low eigenvalue should be eliminated

# Learn more in https://www.real-statistics.com/multivariate-statistics/factor-analysis/determining-number-of-factors/#:~:text=As%20mentioned%20previously%2C%20one%20of,1)%2F2%20such%20elements.

#pip install factor_analyzer

#from sklearn.decomposition import FactorAnalysis
#from factor_analyzer import FactorAnalyzer

#fa = FactorAnalyzer(n_factors = 5)
#fa.fit_transform(train.select_dtypes(exclude = ('object','category')))
#fa.loadings_


# In[46]:


# Building the data model
# Build a linear Regression model to predict the total monthly expenditure for home mortgages loan. 
# Refer ‘deplotment_RE.xlsx’. Column hc_mortgage_mean is predicted variable. 
# This is the mean monthly mortgage and owner costs of specified geographical location. 
# Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean.


# In[47]:


# building the dataset for modeling
# Looking for unique values in the training dataset
# Creating a dictionary for city, urban, town, CDP, village and borough
# Replacing the same variables 

train['type'].unique()
type_dict = {'type':{'City':1, 
                   'Urban':2, 
                   'Town':3, 
                   'CDP':4, 
                   'Village':5, 
                   'Borough':6}
          }
train.replace(type_dict, inplace = True)


# In[48]:


# Replicating the same on the test dataset

train['type'].unique()
test.replace(type_dict, inplace = True)
test['type'].unique()


# In[49]:


# Selecting the relevant feature columns for modeling

feature_cols = ['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']


# In[50]:


# Predicting on the basis of mortgage mean

x_train = train[feature_cols]
y_train = train['hc_mortgage_mean']

x_test = test[feature_cols]
y_test = test['hc_mortgage_mean']


# In[51]:


# Importing the metrics for predictive modeling
# Scaling the data with standard scaler
# Standardization refers to shifting each feature datapoint towards a mean on 0 and standard deviation of 1

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)


# In[53]:


linereg = LinearRegression()
linereg.fit(x_train_scaled, y_train)


# In[54]:


# Making the prediction on the test set scaled
# RMSE of test > RMSE of train => OVER FITTING of the data
# RMSE of test < RMSE of train => UNDER FITTING of the data
# The closer the value of RMSE is to zero , the better is the Regression Model

y_pred = linereg.predict(x_test_scaled)
print("R2 score of linear regression model", r2_score(y_test,y_pred))
print("RMSE of linear regression model", np.sqrt(mean_squared_error(y_test, y_pred)))


# In[55]:


# Running the model at state level
# There are 50 sates in US
# Picking a few iDs 20, 1 , 45, 6

state = train['STATEID'].unique()
state[0:5]


# In[56]:


# R2 of 60% and above has been achieved

for i in [20,1,45]:
    print("State ID -",i)
    
    x_train_nation = train[train['COUNTYID'] == i][feature_cols]
    y_train_nation = train[train['COUNTYID'] == i]['hc_mortgage_mean']
    
    x_test_nation = test[test['COUNTYID'] == i][feature_cols]
    y_test_nation = test[test['COUNTYID'] == i]['hc_mortgage_mean']
    
    x_train_scaled_nation = sc.fit_transform(x_train_nation)
    x_test_scaled_nation = sc.fit_transform(x_test_nation)
    
    linereg.fit(x_train_scaled_nation,y_train_nation)
    y_pred_nation = linereg.predict(x_test_scaled_nation)
    
    print("Overall R2 score of linear regression model for state,",i,":" ,r2_score(y_test_nation,y_pred_nation))
    print("Overall RMSE of linear regression model for state,",i,":" ,np.sqrt(mean_squared_error(y_test_nation,y_pred_nation)))
    print("\n")


# In[57]:


# Lets check if the predicted variables are normally distributed; they are

r = y_test - y_pred
r
plt.hist(r, color = 'c') 
#sns.distplot(r)


# In[58]:


# Thank You :) 

