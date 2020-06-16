#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[6]:


data = pd.read_csv("mnist_train.csv")
data.head()


# In[9]:


a= data.iloc[2,1:].values


# In[ ]:





# In[10]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[11]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[14]:


y_train.head()


# In[15]:


rf = RandomForestClassifier(n_estimators=100)


# In[16]:


rf.fit(x_train,y_train)


# In[17]:


pred = rf.predict(x_test)


# In[18]:


pred


# In[19]:


s = y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count = count+1


# In[20]:


count


# In[21]:


len(pred)


# In[22]:


11601/12000


# In[ ]:




