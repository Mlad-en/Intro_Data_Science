#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from pathlib import Path
from os import path
import matplotlib.pyplot as plt


# In[5]:


root = Path(".").resolve()
cleaned_data = root / "cleaned_data"
source_data = root / "exploratory_datasets"


# In[6]:


file = "final_data_without_speech.parquet"
co2file = "co-emissions-per-capita.xlsx"


# In[7]:


x = path.exists(source_data/co2file)
x


# In[11]:


df1 = pd.read_parquet(cleaned_data/file).drop('Annual CO₂ emissions (per capita)', axis=1)
df2 = pd.read_excel(source_data/co2file)
df2 = df2.astype({'Annual COâ‚‚ emissions (per capita)': float})
df2['Anual co2 per capita shifted'] = df2['Annual COâ‚‚ emissions (per capita)'].shift(-1)


# In[12]:


df2 = df2[df2['Year'] > 2012]
df2['Change co2'] = df2['Annual COâ‚‚ emissions (per capita)'] - df2['Anual co2 per capita shifted']
df2 = df2.dropna()
df2


# In[33]:


df_merged = df1.merge(df2, how='left', left_on=['iso_3', 'year'], right_on=['Code', 'Year'])
# df_merged['co2_change'] = df_merged['Annual CO₂ emissions (per capita)_y'] - df_merged['Annual CO₂ emissions (per capita)_x']
df_merged.head(5)


# In[34]:


df_merged = df_merged[['iso_3', 'year','Region Name', 'Sub-region Name', 'sentiment', 'amount_of_time_spent_on_climate', 'Annual COâ‚‚ emissions (per capita)', 'Anual co2 per capita shifted', 'Change co2']]
# df_AFG = df_merged[df_merged['iso_3'] == 'AFG']
df_merged.head(5)
# df.to_csv('C:\\Users\\Sander\\Documents\\Master Data Science\\Fundamentals', index=True, header=True)


# In[35]:


df_merged['Region Name'].unique()


# In[36]:


df_merged = df_merged[df_merged['Anual co2 per capita shifted'] < 7]
df_merged = df_merged[df_merged['Change co2'] < .1]
df_merged = df_merged[df_merged['Change co2'] > -.1]
# df_merged = df_merged[df_merged['Region Name'] == 'Americas']
df_merged = df_merged.dropna()
df_merged['Change co2'].dtype


# In[37]:


df_merged.plot.scatter(x="amount_of_time_spent_on_climate", y="Change co2", c='black')


# In[40]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split


# In[46]:


regr = make_pipeline(PowerTransformer(), PolynomialFeatures(3), LinearRegression())

X = df_merged['amount_of_time_spent_on_climate'].values[:, np.newaxis]
Y = df_merged['Change co2'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

regr.fit(X_train, Y_train)

xfit = np.linspace(min(X_train), max(X_train))
yfit = regr.predict(xfit)

fig, ax = plt.subplots()
ax.scatter(X_train, Y_train, c='black')
ax.set_xlabel("amount of time spent on climate")
ax.set_ylabel("Change Co2")
ax.plot(xfit, yfit, c='red');
plt.show()


# In[ ]:




