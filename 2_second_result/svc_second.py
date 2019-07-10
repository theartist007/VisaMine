#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Load cleaned up version of dataset from 2012 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[2]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[3]:


ones_df = df[df['CASE_STATUS']=='DENIED']
zeros_df = df[df['CASE_STATUS']=='CERTIFIED']


# In[4]:


# df = pd.concat([ones_df, zeros_df.sample(frac=0.1, random_state=99)])
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


# In[ ]:





# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[9]:


n_components = 400
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized')
pca = pca.fit(X_train)


# In[10]:


from time import time
t0 = time()
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print("done in %0.3fs" % (time() - t0))


# Train SVC and predict using X_test

# In[12]:


from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)


# Print out classification stats

# In[14]:


from sklearn.metrics import classification_report, confusion_matrix
print("** SV Classifier Stats (OVERALL testset)")
print(classification_report(y_test, svc_pred))


# In[ ]:


from sklearn.externals import joblib
joblib.dump(svc, 'svc_final.pkl', compress=9)


# Output the trained model to a pickle file in case we need it in the future

# In[15]:


svc = joblib.load('svc.pkl')


# Evaluate stats for the entire test set

# In[16]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix

print("** SV Classifier Scores (OVERALL testset)")
print("Precision: %s"% precision_score(y_test, svc_pred))
print("Recall: %s"% recall_score(y_test, svc_pred))
print("Accuracy score: %s"% accuracy_score(y_test, svc_pred))
print("F-1 score: %s"% f1_score(y_test, svc_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, svc_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, svc_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, svc_pred))


# Evaluate stats for testset with label 1 (DENIED cases only)

# In[47]:


# pd.DataFrame(data=X_test)


# In[63]:


import numpy as np
ones_index = y_test.nonzero()[0]
x_ones = X_test[ones_index,:]
y_ones = np.take(y_test, ones_index)

# print(x_ones.shape)
# print(y_ones.shape)


# In[64]:


svc_pred_oneonly = svc.predict(x_ones)


# In[65]:


print("** SV Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, svc_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, svc_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, svc_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, svc_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, svc_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, svc_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, svc_pred_oneonly))


# Evaluate stats for testset with label 0 (CERTIFIED cases only)

# In[66]:


zeros_index = np.arange(len(y_test))[(y_test==0)]
x_zeros = X_test[zeros_index,:]
y_zeros = np.take(y_test, zeros_index)
svc_pred_zeroonly = svc.predict(x_zeros)


# In[67]:


print("** SV Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, svc_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, svc_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, svc_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, svc_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, svc_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, svc_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, svc_pred_zeroonly))


# In[ ]:


import datetime
datetime.datetime.now()


# In[ ]:




