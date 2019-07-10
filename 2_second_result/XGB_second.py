#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2012 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[2]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[3]:


ones_df = df[df['CASE_STATUS']=='DENIED']
zeros_df = df[df['CASE_STATUS']=='CERTIFIED']


# In[4]:


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


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# Initialize and train a XGBoost classifier

# In[9]:


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=99, nthread=11)

xgb.fit(X_train, y_train)
# print(xgb.feature_importances_)
print(xgb.score(X_test, y_test))
xgb_pred = xgb.predict(X_test)
print(xgb_pred)


# In[12]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix



print("** XGB Classifier Stats (OVERALL testset)")
print(classification_report(y_test, xgb_pred))


# In[14]:


import operator
nonzero_features = {}
for i in range(len(xgb.feature_importances_)):
    if xgb.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = xgb.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 
for i in range(len(sorted_result)):
    print(sorted_result[-(i+1)])

# with open('/tmp/feature_importances_first_800k_xgb.pickle', 'wb') as f:
#     pickle.dump(sorted_result, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/tmp/feature_importances_first_800k_xgb_dict.pickle', 'wb') as f:
#     pickle.dump(nonzero_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[15]:


print("** XGB Classifier Scores (OVERALL testset)")
print("Precision: %s" % precision_score(y_test, xgb_pred))
print("Recall: %s"% recall_score(y_test, xgb_pred))
print("Accuracy score: %s"% accuracy_score(y_test, xgb_pred))
print("F-1 score: %s"% f1_score(y_test, xgb_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, xgb_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, xgb_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, xgb_pred))
print("\n")


# In[17]:


import numpy as np
ones_index = y_test.nonzero()
x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test, ones_index)[0]

xgb_pred_oneonly = xgb.predict(x_ones)


# In[18]:


print("** XGB Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s" % precision_score(y_ones, xgb_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, xgb_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, xgb_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, xgb_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, xgb_pred_oneonly))
print("\n")


# In[19]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test, zeros_index)

xgb_pred_zeroonly = xgb.predict(x_zeros)


# In[20]:


print("** XGB Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s" % precision_score(y_zeros, xgb_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, xgb_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, xgb_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, xgb_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, xgb_pred_zeroonly))
print("\n")


# In[ ]:




