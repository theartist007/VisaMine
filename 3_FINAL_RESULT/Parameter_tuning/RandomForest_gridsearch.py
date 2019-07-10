#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2015 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[37]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_second.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[38]:


# len(df.EMPLOYER_NAME.unique())


# Show first 5 lines of the dataset

# In[39]:


ones_df = df[df['CASE_STATUS']=='DENIED']
zeros_df = df[df['CASE_STATUS']=='CERTIFIED']
df = pd.concat([ones_df, zeros_df.sample(frac=0.02, random_state=99)])


# In[40]:


ones_df.head()
print(ones_df.shape)


# In[41]:


zeros_df.head()
print(zeros_df.shape)


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[42]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['CASE_STATUS'])

X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# Train RandomForestClassifier and specify parameters to tune.  Run randomized search for 50 iterations

# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# from sklearn.feature_extraction.text import CountVectorizer

rfc = RandomForestClassifier()
# vec = CountVectorizer()
# X_train_t = vec.fit_transform(X_train)

# rfc.fit(X, y)

params = {'n_estimators': list(range(10,300,40)),
'max_depth': list(range(10,100,10)),
'max_features': list(range(2,20,2)),
# 'min_samples_leaf': list(range(3,10,2)),
# 'min_samples_split': list(range(5,15,3)),
#           'bootstrap': [True, False]
}

from time import time
start = time()

# run grid search
n_iter_search = 50
# grid_search = GridSearchCV(rfc, param_grid=params, cv=5)
random_search = RandomizedSearchCV(rfc, param_distributions=params, scoring='accuracy',
                                   n_iter=n_iter_search, cv=5)


random_search.fit(X, y)
endtime= time()


# In[48]:


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


# In[25]:


import pickle
with open('random_search_result_accuracy.pickle', 'wb') as f:
    pickle.dump(random_search, f, protocol=pickle.HIGHEST_PROTOCOL) 


# In[26]:


print(random_search.cv_results_['mean_test_score'])

