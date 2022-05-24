#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries


# In[1]:


import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib as mpl


# In[2]:


# Import the Files


# In[3]:


train = pd.read_csv("C:/Users/Hp/Documents/Wallmart Project/train.csv.zip")
test = pd.read_csv("C:/Users/Hp/Documents/Wallmart Project/test.csv.zip")
features = pd.read_csv("C:/Users/Hp/Documents/Wallmart Project/features.csv.zip")
sampleSubmission= pd.read_csv("C:/Users/Hp/Documents/Wallmart Project/sampleSubmission.csv.zip")
stores = pd.read_csv("C:/Users/Hp/Documents/Wallmart Project/stores.csv")


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


features.head()


# In[7]:


stores.head()


# In[8]:


# Check the data types


# In[9]:


train.dtypes


# In[10]:


test.dtypes


# In[11]:


features.dtypes


# In[12]:


stores.dtypes


# In[13]:


sampleSubmission.dtypes


# In[ ]:


# Change the Data Types of Date to Datetime


# In[14]:


train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
features.Date = pd.to_datetime(features.Date)


# In[ ]:


#Converting a Catagorical to Numerical Values


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Creating a instance of label Encoder


# In[16]:


le = LabelEncoder()


# In[ ]:


# Using .fit_transform function to fit label
# encoder and return encoded label


# In[17]:


label = le.fit_transform(stores['Type'])


# In[18]:


label


# In[ ]:


# removing the column 'Purchased' from df
# as it is of no use now


# In[19]:


stores.drop("Type", axis=1, inplace=True)


# In[ ]:


# Appending the array to our dataFrame
# with column name 'Purchased


# In[20]:


stores["Type"] = label


# In[21]:


stores


# In[ ]:


# Checking the NullValues


# In[25]:


train.isnull().sum()


# In[26]:


test.isnull().sum()


# In[27]:


stores.isnull().sum()


# In[32]:


features.isnull().sum()


# In[29]:


features.shape


# In[ ]:


# Handling the Null Values


# In[31]:


features.fillna(features.mean(), inplace=True)


# In[33]:


features.isnull().sum()


# In[34]:


# Merging the Data


# In[35]:


feature_store = features.merge(stores, how='inner', on = "Store")


# In[36]:


feature_store.head()


# In[37]:


train = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
train.head()


# In[38]:


test = test.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
test.head()


# In[40]:


train.info()


# In[41]:


test.info()


# In[42]:


train.describe()


# In[43]:


test.describe()


# In[ ]:


# Data Visualization


# In[44]:


plt.figure(figsize=(10,6))
plt.hist(train['Weekly_Sales'])
plt.show


# In[45]:


plt.figure(figsize=(10,6))
plt.hist(train['Store'])
plt.show


# In[46]:


plt.figure(figsize=(10,6))
plt.hist(test['Store'])
plt.show


# In[47]:


train.corr().style.background_gradient(cmap="coolwarm")


# In[48]:


corr = train.corr() 

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.show()


# In[49]:


corr = test.corr() 

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.show()


# In[ ]:




