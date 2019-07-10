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


df = pd.read_pickle('/Users/minse_chang/PycharmProjects/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[3]:


df.head()


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


# Apply PCA with 400 features.

# In[8]:


n_components = 400
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
pca = pca.fit(X)


# In[9]:


from time import time
t0 = time()
# eigenfaces = pca.components_.reshape((n_components, h, w))
X = pca.transform(X)

print("done in %0.3fs" % (time() - t0))


# In[10]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Create a SVC with RBF kernel and specify parameters to tune.  Run Randomized search for 7 iterations only since each iteration takes a very long time

# In[12]:


from sklearn.svm import SVC

svc = SVC(kernel='rbf')
params={
    'C':[1,10,100,1000], 
    'gamma':[1,0.1,0.001,0.0001],
}





from time import time
start = time()

# run grid search
n_iter_search = 7
# grid_search = GridSearchCV(svc, param_grid=params, cv=5)
random_search = RandomizedSearchCV(svc, param_distributions=params, scoring='accuracy',
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

