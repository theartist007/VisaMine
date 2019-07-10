#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2012 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[3]:


df = pd.read_pickle('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df = df[['CASE_STATUS','WAGE_LOWER_THAN_PW','PREVAILING_WAGE','SOC_CODE','EMPLOYER_NAME']]
# df = df[['CASE_STATUS','WAGE_LOWER_THAN_PW','PREVAILING_WAGE','SOC_CODE']]
df['CASE_STATUS'] = df['CASE_STATUS'].apply(lambda x : 1 if x=='DENIED' else 0)


# In[4]:


df['WAGE_LOWER_THAN_PW'] = df['WAGE_LOWER_THAN_PW'].astype(int)
df.head(10)


# In[5]:


ones_df = df[df['CASE_STATUS']==1]
zeros_df = df[df['CASE_STATUS']==0]


# In[6]:


df = pd.concat([ones_df, zeros_df.sample(frac=0.02, random_state=99)])


# In[7]:


ones_df.head()
print(ones_df.shape)


# In[8]:


zeros_df.head()
print(zeros_df.shape)


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[9]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['CASE_STATUS'])


X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[10]:


y = df['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Check value counts to verify stratify split

# In[17]:


# print(y_train.value_counts())
# print(y_test.value_counts())
X_train.head()


# Train DecisionTreeClassifier and predict using only 4 features from X_test

# In[78]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

clf_pred = clf.predict(X_test)


# In[79]:


from sklearn.metrics import classification_report, confusion_matrix
print("** Random Forest Classifier Stats (OVERALL testset)")
print(classification_report(y_test, clf_pred))


# Print out score statsfor the entire dataset

# In[80]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix

print("** Decision Tree Classifier Scores (OVERALL testset)")
print("Precision: %s"% precision_score(y_test, clf_pred))
print("Recall: %s"% recall_score(y_test, clf_pred))
print("Accuracy score: %s"% accuracy_score(y_test, clf_pred))
print("F-1 score: %s"% f1_score(y_test, clf_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, clf_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, clf_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, clf_pred))


# In[81]:


df.head()


# Evaluate stats for testset with label 1 (DENIED cases only)

# In[82]:


import numpy as np
ones_index = list(y_test.nonzero()[0])


# In[83]:


x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test.values, ones_index)

clf_pred_oneonly = clf.predict(x_ones)


# In[84]:


print("** Decision Tree Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, clf_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, clf_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, clf_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, clf_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, clf_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, clf_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, clf_pred_oneonly))


# Evaluate stats for testset with label 0 (CERTIFIED cases only)

# In[85]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test.values, zeros_index)
clf_pred_zeroonly = clf.predict(x_zeros)


# In[86]:


print("** Decision Tree Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, clf_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, clf_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, clf_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, clf_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, clf_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, clf_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, clf_pred_zeroonly))


# In[87]:


import operator
nonzero_features = {}
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = clf.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 
for i in range(len(sorted_result)):
    print(sorted_result[-(i+1)])


# Evaluate a naive classifier which predicts a case is always denied if wage is lower than PW. It is always certified otherwise.

# In[47]:


status_stats = df.groupby('CASE_STATUS').size().to_frame()
status_stats['percentage'] = status_stats[0] / status_stats[0].sum()
print(status_stats)


# In[12]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix


naive_prediction = X_test['WAGE_LOWER_THAN_PW']
print("** Naive Classifier Scores (Overall)")
print("Precision: %s"% precision_score(y_test, naive_prediction))
print("Recall: %s"% recall_score(y_test, naive_prediction))
print("Accuracy score: %s"% accuracy_score(y_test, naive_prediction))
print("F-1 score: %s"% f1_score(y_test, naive_prediction))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, naive_prediction, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, naive_prediction, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, naive_prediction))


# In[ ]:




