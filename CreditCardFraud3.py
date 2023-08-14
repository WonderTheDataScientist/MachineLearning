#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib import gridspec


# In[4]:


cc =  pd.read_csv('../creditcard.csv')
cc.head()


# In[4]:


cc.tail()


# In[ ]:





# In[5]:


#Cleaning Dataset
cc.isnull().sum()


# In[6]:


cc.shape


# In[7]:


cc.describe()


# In[8]:


non_fraud = cc[cc.Class == 0]
fraud = cc[cc.Class == 1]
print("Total Cases" , cc.V1.count())
print("Number of Fruads", len(fraud))
print("Number of Non Fraud", len(non_fraud))


# In[ ]:





# In[9]:


fraud = len(fraud)
non_fraud = len(non_fraud)


# In[10]:


fraud_pct = (fraud/(fraud+non_fraud)) * 100
fraud_pct


# In[11]:


non_fraud_pct = (non_fraud/(non_fraud+fraud)) * 100
non_fraud_pct


# In[12]:


x =  round(fraud_pct, 2)
y =  round(non_fraud_pct,2)

plt.title = "Percentage of Labels and Non Labels"
data = [x, y]
detc= ["Fraud" ,"Non Fraud"]
plt.pie(data, labels = detc, colors=['#90a', '#23daca'])
plt.legend(title=[x, y])
plt.show()


# In[13]:


# plot the named features 
# amount_value, time_value = plt.subplots(1, 2, figsize=(18,4), sharex = True)


amount_value = cc['Amount'].values # values
time_value = cc['Time'].values # values

sns.histplot(amount_value, kde=True, stat="density").set_title('Distribution of Amount')

# sns.histplot(time_value, kde=True, stat="density").set_title('Distribution of Time')
# sns.histplot(time_value, hist=False, color="m", kde_kws={"shade": True}, ax=axes[1]).set_title('Distribution of Time')

plt.show()


# In[14]:


time_value = cc['Time'].values
sns.kdeplot(cc["Time"]).set_title('Time Analysis Wise Fraud')
# sns.distplot(time_value, kde=True, stat="density").set_title('Distribution of Amount')
plt.show()


# ### EXPLORATORY DATA VISUALIZATION
# It often makes things much easier to understand when visualizations and graphics are used. The same goes for Machine Learning problems. Different aspects of the dataset are visualized to get a better understanding of the data, and this process is called exploratory data visualization.
# 
# Plotting histograms to understand the values of each variable is a good place to start. Using the code given below:

# In[5]:


import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.keys())[10:]
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor=colors[i])
        ax.set_title(feature+" Histogram",color=colors[35])
        ax.set_yscale('log')
    fig.tight_layout() 
    plt.savefig('Histograms.png')
    plt.show()
draw_histograms(cc,cc.columns,8,4)


# In[15]:


# Reorder the columns Amount, Time then the rest
data_plot = cc.copy()
amount = data_plot['Amount']
data_plot.drop(labels=['Amount'], axis=1, inplace = True)
data_plot.insert(0, 'Amount', amount)

# Plot the distributions of the features
columns = data_plot.iloc[:,0:30].columns
plt.figure(figsize=(12,30*4))
grids = gridspec.GridSpec(30, 1)
for grid, index in enumerate(data_plot[columns]):
 ax = plt.subplot(grids[grid])
 sns.histplot(data_plot[index][data_plot.Class == 1],  kde=True, bins=50)
 sns.histplot(data_plot[index][data_plot.Class == 0],  kde=True, bins=50)
 ax.set_xlabel("")
 ax.set_title("Distribution of Column: "  + str(index))
plt.savefig('Distribution_chart.png')
plt.show()


# In[16]:


from sklearn.preprocessing import StandardScaler 


# In[17]:


scaler =  StandardScaler()


# In[18]:


cc['Normalized_amount'] = scaler.fit_transform(cc["Amount"].values.reshape(-1, 1))
cc.drop(["Amount", "Time"], inplace=True, axis=1)  
cc.head()


# In[15]:


# get the independent (target data) variable
x =  cc.drop(['Class'], axis=1)
# get dependent (label data) variable
y = cc['Class']


# In[20]:


# DATA TRANING AND TESTING


# In[16]:


from sklearn.model_selection import train_test_split


# In[116]:


x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.30, random_state =  1)


# In[117]:


x_train.shape


# In[118]:


x_test.shape


# In[25]:


# Creating the Model


# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[27]:


rf1  = RandomForestClassifier(n_estimators = 20)
rf1.fit(x_train, y_train)


# In[28]:


# Predictions


# In[29]:


predicion_rf1 =  rf1.predict(x_test)
rf1_score = rf1.score(x_test, y_test) * 100 # get the percentage accuracy
rf1_score


# #### LOGISTIC REGRESSION

# #### Logistic Regression is most commonly used in problems of binary classification in which the algorithm predicts one of the two possible outcomes based on various features relevant to the problem.

# #### CORRELATION BETWEEN VARIABLES USING HEAT MAP

# In[12]:


plt.figure(figsize=(16, 6))
plt.title = "CORRELATION ANALYSIS OF THE VARIABLES"
sns.heatmap(cc[['Time', 'Amount', 'Class']].corr(), annot=True)
plt.savefig('Correlation.png')
plt.show()


# ### Model Building and Training
# #### There are various packages that make using Machine Learning models as 
# #### simple as function calls or object instantiation, although the underlying code
# is often very complicated and requires good knowledge of the mathematics behind the working of the algorithm.

# #### Building the Logistic Regression Model
# The model can be simply build using the line of code below:

# In[14]:


from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model


# #### Training the model
# ##### The model can be trained by passing train set features 
# and their corresponding target class values. The model will use that to learn to classify unseen examples.

# In[18]:


model.fit(x_train, y_train)


# ### Evaluating the model
# #### It is important to check how well the model performs 
# #### both on unseen examples because it will be only useful
# #### if it can correctly classify examples, not in the training set.

# In[26]:


train_acc = model.score(x_train, y_train)
print('The Accuracy for training set is', round(train_acc * 100, 3))


# ##### Over 99.8% accuracy, which is pretty good, but training accuracy is not that useful, test accuracy is the real metric of success.
# 
# 

# ### Evaluating on Test Set
# #### Checking the performance on the test set.

# In[53]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))


# #### GENERATING CLASSIFICATION REPORT

# 
# Since this data is imbalanced (having very less number of cases when y =1). 
# In cases like this, the Classification report gives more information 
# than simple accuracy measures. It tells about precision and recall as well.

# In[60]:


print(classification_report(y_test, y_pred))


# ### ROOTMEAN, VISUALIZING USING CONFUSION MATRIX

# #### Confusion Matrix also gives similar information to the classification report, but it is easier to understand. 
# #### It shows how many values of each class were correctly or incorrectly classified.

# In[76]:


cm=confusion_matrix(y_test,y_pred)
cm


# #### ROOT MEAN SQUARE
# ##### the root-mean-square deviation of an estimator is a measure of the imperfection of the fit of the estimator to the data.

# In[115]:


import math
RMS = np.square(np.subtract(y_test,y_pred)).mean()
print("The root mean squre of the model is given to be" ,math.sqrt(RMS))


# #### Confusion Matrix

# In[110]:


plt.figure(figsize=(12,6))
plt.title = "Confusion Matrix"
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues').set(ylabel="actual value", xlabel="Predicted Value")
plt.savefig('confusion_matrix.png')


# ### CROSS VALIDATION 

# In[83]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf


# In[84]:


from sklearn.model_selection import StratifiedKFold


# In[85]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
skf
lst_accu_stratified = []
skf


# In[97]:


fold_no = 1
def train_model(train, test, fold_no):
    x_train = train[x]
    y_train = train[y]
    x_test = test[X]
    y_test = test[y]
    model.fit(x_train,y_train)
    return train_model()
print('Fold',str(fold_no),'Accuracy:',accuracy_score(y_test,y_pred))
    


# In[107]:


# for train_index, test_index in skf.split(cc, y):
#     train = x[train_index].values()
#     test = y[test_index].values()
#     train_model(train,test,fold_no)
#     fold_no += 1


# In[108]:


#     x_train_fold, x_test_fold = x[train_index], x[test_index] 
#     y_train_fold, y_test_fold = y[train_index], y[test_index] 
#     model.fit(x_train_fold, y_train_fold) 
#     lst_accu_stratified.append(model.score(x_test_fold, y_test_fold))


# In[ ]:





# In[ ]:


# skf.score(x_test, y_test)

