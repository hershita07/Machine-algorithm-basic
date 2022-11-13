#!/usr/bin/env python
# coding: utf-8

# ## LINEAR REGRESSION

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
### this is used for display of plots#####


# ### Import dataset and extract the dependent and independant variables

# In[22]:


df = pd.read_csv("Salary_Data.csv")
x=df.iloc[:,:-1].values    ###rows from end to end,col from first to last except last###
y=df.iloc[:,1].values      ### only the colum of index1 


# In[23]:


df


# ### Visualizing the dataset (training)

# In[24]:


sns.distplot(df['YearsExperience'],kde=False,bins=10)


# In[25]:


sns.countplot(y= 'YearsExperience',data=df)


# In[26]:


sns.barplot(x='YearsExperience',y='Salary',data=df)


# In[27]:


sns.heatmap(df.corr())


# ### splitting data for training and testing dataset
# 

# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1/3,random_state=0)


# In[29]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[30]:


y_pred =lr.predict(x_test)
y_pred


# In[31]:


plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, lr.predict(x_train),color='red')
plt.title('Salary = Experience (Train set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show


# In[33]:


# calculating the residuals
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# In[ ]:




