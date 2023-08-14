#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[142]:


cc =  pd.read_csv('../creditcard.csv')
cc.head()


# In[143]:


cc.tail()


# In[ ]:





# In[144]:


#Cleaning Dataset
cc.isnull().sum()


# In[145]:


cc.shape


# In[146]:


cc.describe()


# In[147]:


non_fraud = cc[cc.Class == 0]
fraud = cc[cc.Class == 1]
print("Total Cases" , cc.V1.count())
print("Number of Fruads", len(fraud))
print("Number of Non Fraud", len(non_fraud))


# In[ ]:





# In[148]:


fraud = len(fraud)
non_fraud = len(non_fraud)


# In[149]:


fraud_pct = (fraud/(fraud+non_fraud)) * 100
fraud_pct


# In[150]:


non_fraud_pct = (non_fraud/(non_fraud+fraud)) * 100
non_fraud_pct


# In[151]:


x =  round(fraud_pct, 2)
y =  round(non_fraud_pct,2)

plt.title = "Percentage of Labels and Non Labels"
data = [x, y]
detc= ["Fraud" ,"Non Fraud"]
plt.pie(data, labels = detc, colors=['#90a', '#23daca'])
plt.legend(title=[x, y])
plt.show()


# In[152]:


from sklearn.preprocessing import StandardScaler 


# In[153]:


scaler =  StandardScaler()


# In[ ]:


cc['Normalized_amount'] = scaler.fit_transform(cc["Amount"].values.reshape(-1, 1))
cc.drop(['Amount', 'Time'], inplace=True, axis=1)  
cc.head()


# In[155]:


# get the independent (target data) variable
x =  cc.drop(['Class'], axis=1)
# get dependent (label data) variable
y = cc['Class']


# In[156]:


# DATA TRANING AND TESTING


# In[157]:


from sklearn.model_selection import train_test_split


# In[184]:


x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state =  1)


# In[159]:


x_train.shape


# In[160]:


x_test.shape


# In[161]:


# Creating the Model


# In[162]:


from sklearn.ensemble import RandomForestClassifier


# In[163]:


rf1  = RandomForestClassifier(n_estimators = 100)
rf1.fit(x_train, y_train)


# In[164]:


# Predictions


# In[165]:


predicion_rf1 =  rf1.predict(x_test)
rf1_score = rf1.score(x_test, y_test) * 100 # get the percentage accuracy
rf1_score


# In[166]:


from sklearn.svm import SVC


# In[167]:


svm = SVC()
svm.fit(x_train, y_train)
svm.score(x_test, y_test)


# In[168]:


from datetime import datetime
 
 
timestamp = 172786.0
dt_obj = datetime.fromtimestamp(1140825600)
 
print("date_time:",dt_obj)
print("type of dt:",type(dt_obj))


# In[169]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf


# In[170]:


from sklearn.model_selection import StratifiedKFold


# In[176]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
skf
lst_accu_stratified = []
skf


# In[182]:


model = RandomForestClassifier() 
model


# In[186]:


skf.split(x, y)
#     x_train_fold, x_test_fold = x[train_index], x[test_index] 
#     y_train_fold, y_test_fold = y[train_index], y[test_index] 
#     model.fit(x_train_fold, y_train_fold) 
#     lst_accu_stratified.append(model.score(x_test_fold, y_test_fold))


# In[2]:


# skf.fit(x_train, y_train) 


# In[1]:


# skf.score(x_test, y_test)

