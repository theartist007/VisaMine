#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA


# Load cleaned up version of dataset from 2015 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[2]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[3]:


# len(df.EMPLOYER_NAME.unique())


# Show first 5 lines of the dataset

# In[4]:


ones_df = df[df['CASE_STATUS']=='DENIED']
zeros_df = df[df['CASE_STATUS']=='CERTIFIED']
df = pd.concat([ones_df, zeros_df.sample(frac=0.02, random_state=99)])


# In[5]:


ones_df.head()
print(ones_df.shape)


# In[6]:


zeros_df.head()
print(zeros_df.shape)


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[7]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['CASE_STATUS'])

X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[8]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# Check value counts to verify stratify split

# In[9]:


n_components = 400
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
pca = pca.fit(X)


# In[10]:


from time import time
t0 = time()
# eigenfaces = pca.components_.reshape((n_components, h, w))
X = pca.transform(X)

print("done in %0.3fs" % (time() - t0))


# Train RandomForestClassifier and predict using X_test

# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Create XGBoost classifier with 11 threads and define parameters to control. Run randomized search with 20 iterations

# In[12]:


import xgboost as xgb


xgb = xgb.XGBClassifier(nthread=11)

params={
    'max_depth': [3,4,5,6,7,8,9], 
    'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5,0.6,0.7,0.8],
    'n_estimators': [1000,2000,3000],
    'reg_alpha': [0.01, 0.02, 0.03, 0.04]
}

from time import time
start = time()

# run grid search
n_iter_search = 20
# grid_search = GridSearchCV(rfc, param_grid=params, cv=5)
random_search = RandomizedSearchCV(xgb, param_distributions=params, scoring='accuracy',
                                   n_iter=n_iter_search, cv=5)


random_search.fit(X, y)
endtime= time()


# Output the feature importance vector and sort by importance

# In[14]:


import numpy as np
def report(results, n_top=30):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((endtime - start), n_iter_search))
report(random_search.cv_results_)


# Second iteration of randomized search, based on the results returned from the above lines

# In[14]:


import xgboost as xgb

# from sklearn.feature_extraction.text import CountVectorizer

xgb = xgb.XGBClassifier(nthread=11)
# vec = CountVectorizer()
# X_train_t = vec.fit_transform(X_train)

# rfc.fit(X, y)

params={
    'max_depth': [3,5], 
    'subsample': [0.7,0.9],
    'colsample_bytree': [0.6,0.8],
    'n_estimators': [1000, 3000],
    'gamma': [0.5, 1, 1.5, 2],
    'reg_lambda': [0.01, 0.2, 0.5 ,1.0],
    'reg_alpha': [0.06, 0.08, 0.1, 0.2],
}

from time import time
start = time()

# run grid search
n_iter_search = 20
# grid_search = GridSearchCV(rfc, param_grid=params, cv=5)
random_search = RandomizedSearchCV(xgb, param_distributions=params, scoring='accuracy',
                                   n_iter=n_iter_search, cv=5)


random_search.fit(X, y)
endtime= time()


# In[15]:


import numpy as np
def report(results, n_top=30):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((endtime - start), n_iter_search))
report(random_search.cv_results_)


# In[16]:


import datetime
datetime.datetime.now()

