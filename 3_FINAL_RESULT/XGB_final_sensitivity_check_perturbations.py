#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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


sampled_zeros_df = zeros_df.sample(frac=0.02, random_state=99)
df = pd.concat([ones_df, sampled_zeros_df])


# In[8]:


X = df.drop(columns=['CASE_STATUS'])
y = df['CASE_STATUS'].apply(lambda x: 1 if x=='DENIED' else 0).values

scaler = MinMaxScaler()
X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Slighyly manipulate a few features of X_test

# Multiply prevailing wage by a random number between 0.95 and 1.05.
# 
# At 2% probability, flip the boolean categorical values

# In[12]:


random_constants = np.random.uniform(0.95,1.05,X_test.shape[0])
X_test['PREVAILING_WAGE'] = X_test['PREVAILING_WAGE'] * random_constants


# In[14]:


choices = [True, False]
prob = [0.02, 0.98]
# Toss a coin
# flipped_indices = np.nonzero(np.random.choice(choices, X_test.shape[0], p=prob))
X_test['WAGE_LOWER_THAN_PW'] = X_test['WAGE_LOWER_THAN_PW'].apply(
    lambda x: not x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['FULL_TIME_POSITION_N'] = X_test['FULL_TIME_POSITION_N'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['FULL_TIME_POSITION_Y'] = X_test['FULL_TIME_POSITION_N'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['H1B_DEPENDENT_N'] = X_test['H1B_DEPENDENT_N'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['H1B_DEPENDENT_Y'] = X_test['H1B_DEPENDENT_Y'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['WILLFUL_VIOLATOR_N'] = X_test['WILLFUL_VIOLATOR_N'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)
X_test['WILLFUL_VIOLATOR_Y'] = X_test['WILLFUL_VIOLATOR_Y'].apply(
    lambda x: 1-x if np.random.choice(choices, 1, p=prob)[0] else x)


# Train XGBoost classifier and predict using X_test

# In[15]:


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=99,reg_alpha=0.06, 
colsample_bytree=0.8, 
n_estimators=1000, 
subsample=0.9, 
reg_lambda=0.07, 
max_depth=3, 
gamma=2, nthread=11)

xgb.fit(X_train, y_train)
# print(xgb.feature_importances_)
# print(xgb.score(X_test, y_test))
# xgb_pred = xgb.predict(X_test)
# print(xgb_pred)


# In[16]:


xgb_pred = xgb.predict(X_test)
# X_test_sensitivity.head()


# Evaluate prediction result with the manipulated X_test data

# In[17]:


from sklearn.metrics import classification_report, confusion_matrix
print("** XGBoost Classifier Stats (OVERALL testset)")
print(classification_report(y_test, xgb_pred))


# Output the trained model to a pickle file in case we need it in the future

# Evaluate stats for the entire test set

# In[18]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix

print("** XGBoost Classifier Scores (OVERALL testset)")
print("Precision: %s"% precision_score(y_test, xgb_pred))
print("Recall: %s"% recall_score(y_test, xgb_pred))
print("Accuracy score: %s"% accuracy_score(y_test, xgb_pred))
print("F-1 score: %s"% f1_score(y_test, xgb_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, xgb_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, xgb_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, xgb_pred))


# Evaluate stats for testset with label 1 (DENIED cases only)

# In[19]:


import numpy as np
ones_index = y_test.nonzero()
x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test, ones_index)[0]

xgb_pred_oneonly = xgb.predict(x_ones)


# In[20]:


print("** XGBoost Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, xgb_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, xgb_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, xgb_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, xgb_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, xgb_pred_oneonly))


# Evaluate stats for testset with label 0 (CERTIFIED cases only)

# In[21]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test, zeros_index)
xgb_pred_zeroonly = xgb.predict(x_zeros)


# In[22]:


print("** XGBoost Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, xgb_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, xgb_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, xgb_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, xgb_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, xgb_pred_zeroonly))


# In[23]:


import datetime
datetime.datetime.now()

