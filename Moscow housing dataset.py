#!/usr/bin/env python
# coding: utf-8

# # EDA and Model training on Moscow Housing Price Dataset

# In[1]:


#Loading the necessary libraries needed for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


#Importing the dataset required for this project and then printing the first five rows
df = pd.read_csv("C:/Users/HP/Desktop/Projects/Datasets/data.csv")
df.head()


# # A concise summary of this dataframe

# In[3]:


df.info()


# # Data cleaning 

# ### Converting cols that possess categorical features but not a categorical datatype to categorical datatype

# #### Apartment type column

# In[4]:


df["Apartment type"].unique()


# In[5]:


df["Apartment type"] = df["Apartment type"].astype("category")


# In[6]:


df["Apartment type"].cat.categories


# #### Renovation Column

# In[7]:


df["Renovation"].unique()


# In[8]:


df["Renovation"] = df["Renovation"].astype("category")


# In[9]:


df["Renovation"].cat.categories


# #### Region column

# In[10]:


df["Region"].unique()


# In[11]:


df["Region"] = df["Region"].astype("category")


# In[12]:


df["Region"].cat.categories


# # Checking for missing values in this dataset

# In[13]:


df.isna().sum().sort_values()


# # Handling Outliers

# In[16]:


df["Price"].describe()


# In[43]:


sns.scatterplot(
    x = "Area",
    y = "Price",
    data = df
)
plt.show()


# In[14]:


sns.boxplot(
    y = "Price",
    data = df
)
plt.show()


# In[35]:


df_free_out = df[df["Price"] <= 2.479925e+08]


# In[36]:


df_free_out.info()


# In[37]:


sns.boxplot(
    y = "Price",
    data = df_free_out
)
plt.show()


# In[42]:


sns.scatterplot(
    x = "Area",
    y = "Price",
    data = df_free_out)
plt.show()


# # Answering some questions asked by the publisher of the dataset

# ## What are the most common types of apartments in the region?

# In[45]:


count = df_free_out.groupby("Region")["Apartment type"].value_counts()
count


# In[46]:


count = df_free_out.groupby("Region")["Apartment type"].value_counts(normalize = True)*100
count


# # Data visualization on distribution of the apartment types in all regions

# In[47]:


sns.catplot(
    x = "Region",
    kind = "count",
    col = "Apartment type",
    data = df_free_out,
    col_wrap = 2
)
plt.show()


# Results drawn from analysing the moscow housing dataset shows that; 
# 
# 1. In the Moscow region the "secondary apartment" has the most common apartment type   
# 
# 2. While in the Moscow oblast region the "New building" is the most common apartment type 

# ## Is there a relationship between housing prices and proximity to metro stations?

# In[48]:


corr_price_m2metro = df_free_out["Price"].corr(df_free_out["Minutes to metro"])
print(f"The correlation between Housing prices and Metro stations is: {corr_price_m2metro}")


# In[53]:


sns.scatterplot(
    data = df_free_out,
    x = "Price",
    y = "Minutes to metro"
)
plt.show()


# The correlation statistical analysis and scatter plot data visualization shows a negative correlation   
# which means that there is no relationship between housing prices and proximity to metro stations

# # How does the level of renovation affect the price of an apartment?

# In[54]:


df_free_out.groupby("Renovation")["Price"].describe()


# In[55]:


sns.boxplot(
    x = "Renovation",
    y = "Price",
    data = df_free_out
)
plt.xticks(rotation  = 45)
plt.show()


# In[56]:


sns.barplot(
    x = "Renovation",
    y = "Price",
    data = df_free_out
)
plt.xticks(rotation  = 45)
plt.show()


# Results drawn from analyzing the Moscow housing dataset shows thah;
# 
# 1. Apartments with designer renovation style has the most expensive prices among all other apartment renovation syles
# 
# 2. Apartments without any renovation style comes second to the designer renovation style
# 
# 3. Apartments with the European-style renovations comes third the price rating
# 
# 4. last and the least are apartments with the Cosmetic renovations

# # Is there a price difference in housing between Moscow and the Moscow Oblast region?

# In[57]:


df_free_out.groupby("Region")["Price"].describe()


# In[58]:


sns.barplot(
    x = "Region",
    y = "Price",
    data = df_free_out
)
plt.show()


# Yes there is a huge price difference in housing between Moscow and the Moscow Oblast region
# 

# # Which factors have the greatest impact on housing price

# # Are there any preferences regarding floor levels?

# #### couldn't answer the two  questions aboves but i will really appreiciate if you can guide and put me through on how i can answer this question

# In[60]:


x = df_free_out[["Minutes to metro", "Number of rooms", "Area", "Living area", "Kitchen area", "Floor", "Number of floors"]].values
y = df_free_out["Price"].values


# In[61]:


from sklearn.linear_model import Lasso
names = df_free_out[["Minutes to metro", "Number of rooms", "Area", "Living area", "Kitchen area", "Floor", "Number of floors"]].columns
lasso = Lasso(alpha = 0.1)
lasso_coef = lasso.fit(x, y).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation = 45)
plt.show()


# # Training Model

# In[62]:


from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[63]:


scaler = StandardScaler()


# In[64]:


x_train,x_test,y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[65]:


scale_train = scaler.fit_transform(x_train)


# In[66]:


scale_test = scaler.transform(x_test)


# In[67]:


lasso = Lasso(alpha = 0.1)


# In[68]:


lasso.fit(scale_train, y_train)


# In[69]:


y_predl = lasso.predict(scale_test)


# In[70]:


lasso.score(scale_test, y_test)


# In[71]:


mean_squared_error(y_test, y_predl)


# In[72]:


lmodel = LinearRegression()


# In[73]:


lmodel.fit(scale_train, y_train )


# In[74]:


y_pred = lmodel.predict(scale_test)


# In[75]:


lmodel.score(scale_test, y_test)


# In[76]:


rmse = mean_squared_error(y_test, y_pred, squared = False)
print(f"The Root mean squared error of the linear regression model above is {rmse}")

