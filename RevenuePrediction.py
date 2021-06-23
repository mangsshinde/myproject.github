#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_predict


# In[2]:


df = pd.read_csv('G:\MangeshDataScience\Practice\WorkEx\MarketSegmentation/store.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ### Detecting outliers

# In[5]:


df.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.99])


# In[6]:


# There seem to be no outliers in the numerical columns


# ### Splitting into dependent and independent values

# In[7]:


X = df.drop(['revenue'], axis = 1)
y = df['revenue']


# In[8]:


X.head()


# ### Converting into categorical variables

# In[12]:


X.columns.unique()


# In[13]:


X.reps.value_counts()


# In[16]:


from sklearn.preprocessing import LabelEncoder
df_col = list(X.columns)
for i in range(len(df_col)):
    X[df_col[i]] = LabelEncoder().fit_transform(X[df_col[i]])


# In[17]:


X.head()


# In[20]:


for i in X.columns:
    print(X[i].unique(), end = " ")


# ### Splitting into train and test datasets

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[22]:


print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",X_test.shape)
print("Response Rate in Training Data",y_train.mean())
print("Response Rate in Testing Data",y_test.mean())


# In[35]:


def print_score(reg, X_train, y_train, X_test, y_test, train = True):
    if train:
        pred = reg.predict(X_train)
        print('Train Result: ')
        print(f'Training Accuracy for {reg}: {r2_score(y_train, pred)*100:.2f}%')
        print('RMSE for Training Data: ',sqrt(mean_squared_error(y_train, pred)))
    elif train == False:
        pred = reg.predict(X_test)
        print('\nTest Result: ')
        print(f'Testing Accuracy for {reg} : {r2_score(y_test, pred)*100:.2f}%')
        print('RMSE for Testing Data: ',sqrt(mean_squared_error(y_test, pred)))


# In[36]:


reg = LinearRegression()
reg.fit(X_train, y_train)
print_score(reg, X_train, y_train, X_test, y_test, train = True)
print_score(reg, X_train, y_train, X_test, y_test, train = False)


# In[66]:


def print_score(reg, X_train, y_train, X_test, y_test, train = True):
    if train:
        pred = reg.predict(X_train)
        print(f'Train Result for {reg}: ')
        scores = [r2_score(y_train, pred)*100, sqrt(mean_squared_error(y_train, pred))]
        table_data = {'Scores':scores}
        column_names= ['Accuracy Score', 'Mean Squared Error']
        RegressionReport = pd.DataFrame(data= table_data, index = column_names )
        print('Regression Report: ', RegressionReport)
    elif train == False:
        pred = reg.predict(X_test)
        print(f'\nTest Result for {reg}: ')
        scores = [r2_score(y_test, pred)*100, sqrt(mean_squared_error(y_test, pred))]
        table_data = {'Scores':scores}
        column_names= ['Accuracy Score', 'Mean Squared Error']
        RegressionReport = pd.DataFrame(data= table_data, index = column_names)
        print('Regression Report: ', RegressionReport)


# ### 1. Linear Regression

# In[67]:


print_score(reg, X_train, y_train, X_test, y_test, train = True)
print_score(reg, X_train, y_train, X_test, y_test, train = False)


# ### 2. Decision Tree

# In[73]:


dec_tree = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None,)
dec_tree.fit(X_train, y_train)


# In[74]:


print_score(dec_tree, X_train, y_train, X_test, y_test, train = True)
print_score(dec_tree, X_train, y_train, X_test, y_test, train = False)


# ### 3. Random Forest

# In[78]:


rf = RandomForestRegressor(n_estimators=100, max_depth= 5)
rf.fit(X_train, y_train)


# In[79]:


print_score(rf, X_train, y_train, X_test, y_test, train= True)
print_score(rf, X_train, y_train, X_test, y_test, train= False)


# ### 4. Decision Tree with hyperparameters

# In[83]:


from sklearn.model_selection import GridSearchCV
params = { 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


tree_clf = DecisionTreeRegressor(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeRegressor(**best_params)
tree_clf.fit(X_train, y_train)


# In[84]:


print_score(tree_clf, X_train, y_train, X_test, y_test, train= True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train= False)


# ### 5. Random Forest

# In[85]:


n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2,3,4,5,6]
max_depth.append(None)
#min_samples_split = [2, 5, 10]
#min_samples_leaf = [1, 2, 4, 10]


params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth} #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf}


rf_clf = RandomForestRegressor(random_state=42)

rf_cv = GridSearchCV(rf_clf, params_grid, cv=3, verbose=2,n_jobs = -1)


rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

rf_clf = RandomForestRegressor(**best_params)
rf_clf.fit(X_train, y_train)


# In[86]:


print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# ### 6. Support Vector Machine Regressor

# In[91]:


from sklearn.svm import SVR
svm_reg = SVR()
svm_reg.fit(X_train, y_train)


# In[92]:


print_score(svm_reg, X_train, y_train, X_test, y_test, train=True)
print_score(svm_reg, X_train, y_train, X_test, y_test, train=False)


# ### 7. AdaBoost Classifier

# In[93]:


adb = AdaBoostRegressor()
adb.fit(X_train, y_train)


# In[94]:


print_score(adb, X_train, y_train, X_test, y_test, train=True)
print_score(adb, X_train, y_train, X_test, y_test, train=False)


# ### 8. Gradient Boosting

# In[95]:


gbx = GradientBoostingRegressor()
gbx.fit(X_train, y_train)


# In[96]:


print_score(gbx, X_train, y_train, X_test, y_test, train=True)
print_score(gbx, X_train, y_train, X_test, y_test, train=False)

