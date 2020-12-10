#!/usr/bin/env python
# coding: utf-8

# ### Load the core Libraries
# Program to Investigate the relationship Between New cases of corona Virus and Crude oil prices- A case study of the World and Nigeria.

# In[1]:


import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as dates 

plt.style.use('tableau-colorblind10')


# ### Load the dataset to be used 
# a) Covid 19 datsets 
# 
# source: OWID

# In[2]:


## Load the dataset
world =pd.read_excel("owid-covid-data.xlsx")
world["date"]=pd.to_datetime(world["date"])
world.head(3)


# ### Data Preprocessing 

# In[3]:


# Computing the total daily corona cases from january 1, 2020 till December
daily_cases= world.groupby('date')["new_cases"].sum()
daily_cases.shape


# In[4]:


# Convert to a dataframe
daily_cases=pd.DataFrame(daily_cases)

# Create a Columns of Date- It will be used to Merge this Dataset with that of the Crude Oil
daily_cases["Date"]=daily_cases.index

# Creating Index 
index =index=range(1,339,1)

# Renaming Index 
daily_cases.index=index
daily_cases.tail(5)


# In[5]:


## Extracting the Daily  New Corona cases In Nigeria
nigeria_corona_cases =world[["date","location","new_cases"]][world.location== "Nigeria"]
print(nigeria_corona_cases.shape)

# Create renaming Index 
nigeria_corona_cases['Date']=nigeria_corona_cases.date
nigeria_corona_cases= nigeria_corona_cases.drop('date',axis=1)

# Creating Another index
index1=range(1,317,1)

# Reindexing
nigeria_corona_cases.index = index1

#View it 
nigeria_corona_cases.tail(5)


# b) Crude Oil datasets(WTI)

# In[6]:


crude_oil = pd.read_csv("Crude Oil WTI Futures Historical Data1.csv")

# Convert to date 
crude_oil['Date']=pd.to_datetime(crude_oil['Date'])

crude_oil=crude_oil.iloc[1:,:]
crude_oil.head(5)


# In[7]:


## Info About the dataframe
print(crude_oil.info())
crude_oil.shape


# In[8]:


# Selecting the Date and Price Into the New Dataframe
wti = crude_oil[["Date","Price"]]

#Create a new index
index2 = range(1,246,1)
wti.index =index2

# Sort it
wti = wti.sort_index(ascending=False)
wti.tail(5)


# ### Creating The New Dataframe Merging New Corana Case with Crude Oil Price Per Date
# #### a) World

# In[9]:


World_corona_crude =pd.merge(daily_cases,wti,on="Date",how="left")
print(World_corona_crude.isnull().sum())
World_corona_crude.head(2)


# In[10]:


# Forward Fill the Missing Values 
World_corona_crude.fillna(method="ffill",inplace=True)
print(World_corona_crude.isnull().sum())

# Set the date Variable to Become the Index
World_corona_crude.set_index("Date",drop=True, inplace=True)


# #### b) Nigeria 

# In[11]:


# Creating A new Dataframe of Daily Corona Cases in the Wrold and Crude Oil Price
Nigeria_corona_crude =pd.merge(nigeria_corona_cases,wti,on="Date",how="left")
print(Nigeria_corona_crude.isnull().sum())
Nigeria_corona_crude.head(2)


# In[12]:


## Filling the Missing value using the Previous Value
Nigeria_corona_crude.fillna(method="ffill",inplace=True)
print(Nigeria_corona_crude.isnull().sum())

Nigeria_corona_crude.set_index("Date",drop=True, inplace=True)


# ## a) World
# 
# ### Visualization
# 
# #### World- Time plot of Daily New Corona Cases and  Oil Prices - January to December

# In[13]:


plt.figure(figsize=(10,5))
World_corona_crude["new_cases"].plot(label="New Cases")
World_corona_crude["Price"].plot(label="Price")
plt.title("World")
plt.legend()
plt.show()


# In[14]:


#New Cases 
fig,ax =plt.subplots(figsize=(10,5))
World_corona_crude["new_cases"].plot(label="New Cases")
plt.title("world")
plt.legend()
plt.show()


# In[15]:


# Oil Price
fig,ax =plt.subplots(figsize=(10,5))
World_corona_crude["Price"].plot(color="green",label="Crude Oil Price(WTI)" )
plt.title("World")
plt.legend()
plt.show()


# ### Descriptive Analysis

# In[16]:


World_sum =World_corona_crude.describe()
print(World_sum)


# ### Scatter diagram

# In[17]:


## Scatter Diagram
plt.scatter(World_corona_crude.new_cases,World_corona_crude.Price)
plt.xlabel("New Corona Cases")
plt.ylabel("Crude oil Price(WTI)")
plt.title("World")
plt.show()


# In[18]:


### Outlier 
World_corona_crude.index[World_corona_crude["Price"]< 0]


# April 20th 2020, we  have a negtive Price of WTI crude Oil
# 
# The oil producers are paying buyers to take the commodity off their hands over fears that storage capacity could run out in may(BBC news)

# ### Correlation

# In[19]:


## Checking for corona Virus
world_corr= World_corona_crude[["new_cases","Price"]].corr()
print(world_corr)
sns.heatmap(world_corr,annot=True, fmt="g",cmap="Blues")
plt.title("World")
plt.show()


# #### Time plot of the world daily Corona cases and Crude oil Price- from january to may

# In[20]:


## Janury to may- Dataset
World_jan_may= World_corona_crude[World_corona_crude.index < pd.to_datetime("2020-05-01")]
World_jan_may.tail()


# In[21]:


plt.figure(figsize=(10,5))
World_jan_may["new_cases"].plot(label="New Cases")
World_jan_may["Price"].plot(label="Crude Oil Price(WTI)")
plt.title("World")
plt.legend()
plt.show()


# In[22]:


#New Cases 
fig,ax =plt.subplots(figsize=(10,5))
World_jan_may["new_cases"].plot(label="New Cases")
plt.title("World")
plt.legend()
plt.show()


# In[23]:


# Oil Price
fig,ax =plt.subplots(figsize=(10,5))
World_jan_may["Price"].plot(color="green",label="Crude Oil Price(WTI)" )
plt.title("World")
plt.legend()
plt.show()


# ### Scatter Diagram

# In[24]:


## Scatter Diagram
plt.scatter(World_jan_may.new_cases,World_jan_may.Price)
plt.xlabel("New Corona Cases")
plt.ylabel("Crude oil Price(WTI)")
plt.title("World")
plt.savefig("world1",dpi=150)
plt.show()


# ### Correlation

# In[25]:


## Checking for corona Virus
world_corr1=World_jan_may[["new_cases","Price"]].corr()
print(world_corr1)

sns.heatmap(world_corr1,annot=True, fmt="g",cmap="Reds")
plt.title("World")
plt.savefig("World_corr1",dpi=100)
plt.show()


# ### Regression
# ##### a) January to December

# In[26]:


x= World_corona_crude.new_cases
x = sm.add_constant(x)
y=World_corona_crude.Price

model = sm.OLS(y,x).fit()
model.summary()


# ##### b) January to May

# In[27]:


x=World_jan_may.new_cases
x = sm.add_constant(x)
y=World_jan_may.Price

model = sm.OLS(y,x).fit()
model.summary()


# ### b) Nigeria
# 
# #### Time plot of Daily New Corona Cases and  Oil Prices - January to December 

# In[28]:


plt.figure(figsize=(10,5))
Nigeria_corona_crude["new_cases"].plot(label="New Cases")
Nigeria_corona_crude["Price"].plot(label="Price")
plt.title("Nigeria")
plt.legend()
plt.show()


# In[29]:


#New Cases 
fig,ax =plt.subplots(figsize=(10,5))
Nigeria_corona_crude["new_cases"].plot(label="New Cases")
plt.legend()
plt.show()


# In[30]:


# Oil Price
fig,ax =plt.subplots(figsize=(10,5))
Nigeria_corona_crude["Price"].plot(color="green",label="Crude Oil Price(WTI)" )
plt.legend()
plt.show()


# ####Time Plot of  Nigeria Corona casesand Crdue Oil Price- January to may

# In[31]:


Nigeria_jan_may= Nigeria_corona_crude[Nigeria_corona_crude.index < pd.to_datetime("2020-05-01")]


# In[32]:


plt.figure(figsize=(10,5))
Nigeria_jan_may["new_cases"].plot(label="New Cases")
Nigeria_jan_may["Price"].plot(label="Crude Oil Price(WTI)")
plt.title("Nigeria")
plt.legend()
plt.show()


# In[33]:


#New Cases 
fig,ax =plt.subplots(figsize=(10,5))
Nigeria_jan_may["new_cases"].plot(label="New Cases")
plt.title("Nigeria")
plt.legend()
plt.show()


# ### Scatter Diagram

# In[34]:


## Scatter Diagram
plt.scatter(Nigeria_jan_may.new_cases,Nigeria_jan_may.Price)
plt.xlabel("New Corona Cases")
plt.ylabel("Crude oil Price(WTI)")
plt.title("Nigeria")
plt.show()


# In[35]:


## Checking for corona Virus
nigeria_corr1=Nigeria_jan_may[["new_cases","Price"]].corr()
print(nigeria_corr1)

sns.heatmap(nigeria_corr1,annot=True, fmt="g",cmap="Reds")
plt.title("Nigeria")
plt.show()


# ### Descriptive Analysis

# In[36]:


Nigeria_sum = Nigeria_corona_crude.describe()
print(Nigeria_sum)


# ### Scatter Diagram

# In[37]:


## Scatter Diagram
plt.scatter(Nigeria_corona_crude.new_cases,Nigeria_corona_crude.Price)
plt.xlabel("New Corona Cases")
plt.ylabel("Crude oil Price(WTI)")
plt.title("Nigeria")
plt.show()


# ### Correlation

# In[38]:


## Checking for corona Virus
nigeria_corr=Nigeria_corona_crude[["new_cases","Price"]].corr()
print(nigeria_corr)

sns.heatmap(nigeria_corr,annot=True, fmt="g",cmap="Blues")
plt.title("Nigeria")
plt.show()


# ### Regression Analysis
# 
# #### a) January to December

# In[39]:


x=Nigeria_jan_may.new_cases
x = sm.add_constant(x)
y=Nigeria_jan_may.Price

model = sm.OLS(y,x).fit()
model.summary()


# #### January to May

# In[40]:


x=Nigeria_corona_crude.new_cases
x = sm.add_constant(x)
y=Nigeria_corona_crude.Price

model = sm.OLS(y,x).fit()
model.summary()


# In[ ]:




