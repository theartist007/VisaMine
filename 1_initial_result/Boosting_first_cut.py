#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2015 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[2]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_FINAL.pickle')
df.drop(columns=['EMPLOYER_CITY'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# Show first 5 lines of the dataset

# In[3]:


# df['NAICS_CODE'] = df['NAICS_CODE'].apply(lambda x : x[:4])
# df['SOC_CODE'] = df['SOC_CODE'].apply(lambda x : x[:4])
df.head()


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[4]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['CASE_STATUS'])

X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# Check value counts to verify stratify split

# In[6]:


X_train.head()


# Train XGBClassifier and predict using X_test

# In[7]:


from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=99)

xgb.fit(X_train, y_train)
# print(xgb.feature_importances_)
print(xgb.score(X_test, y_test))
xgb_pred = xgb.predict(X_test)
print(xgb_pred)
# for i in range(len(y_hat)):
#     if y_test[i]==1:
#         print("{}:: {}".format(y_test[i],y_hat[i]))


# Train ADABOOST classifier and predict using X_test

# In[8]:


from sklearn.ensemble import AdaBoostClassifier
# from sklearn.feature_extraction.text import CountVectorizer

abc = AdaBoostClassifier(random_state=99)
# vec = CountVectorizer()
# X_train_t = vec.fit_transform(X_train)

abc.fit(X_train, y_train)

abc_pred = abc.predict(X_test)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix
print("** XGB Classifier Stats (OVERALL testset)")
print(classification_report(y_test, xgb_pred))

print("** ADABOOST Classifier Stats (OVERALL testset)")
print(classification_report(y_test, abc_pred))


# Output the feature importance vector and sort by importance

# In[20]:


import operator
import pickle


# print(len(abc.feature_importances_))
# print(len(X_train.columns))

nonzero_features = {}
for i in range(len(abc.feature_importances_)):
    if abc.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = abc.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 
for i in range(len(sorted_result)):
    print(sorted_result[-(i+1)])
with open('/tmp/feature_importances_first_800k_abc.pickle', 'wb') as f:
    pickle.dump(sorted_result, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('/tmp/feature_importances_first_800k_abc_dict.pickle', 'wb') as f:
    pickle.dump(nonzero_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[23]:


nonzero_features = {}
for i in range(len(xgb.feature_importances_)):
    if xgb.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = xgb.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 
for i in range(len(sorted_result)):
    print(sorted_result[-(i+1)])

with open('/tmp/feature_importances_first_800k_abc.pickle', 'wb') as f:
    pickle.dump(sorted_result, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('/tmp/feature_importances_first_800k_abc_dict.pickle', 'wb') as f:
    pickle.dump(nonzero_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# Output the trained model to a pickle file in case we need it in the future

# In[19]:


from sklearn.externals import joblib
import cPickle
joblib.dump(abc, 'abc.pkl', compress=9)
joblib.dump(xgb, 'xgb.pkl', compress=9)

# import cPickle
with open('abc_cpickle.pkl', 'wb') as pkl:
    cPickle.dump(abc, pkl)
with open('xgb_cpickle.pkl', 'wb') as pkl:
    cPickle.dump(xgb, pkl)    


# To load the pre-trained model, run the cell below.

# In[124]:


abc = joblib.load('abc.pkl')
xgb = joblib.load('xgb.pkl')

# with open('abc_cpickle.pkl', 'rb') as pkl:
#     abc = cPickle.load(pkl)
# with open('xgb_cpickle.pkl', 'rb') as pkl:
#     xgb = cPickle.load(pkl)


# Evaluate stats for the entire test set

# In[11]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix
# print(abc_pred)

print("** XGB Classifier Scores (OVERALL testset)")
print("Precision: %s" % precision_score(y_test, xgb_pred))
print("Recall: %s"% recall_score(y_test, xgb_pred))
print("Accuracy score: %s"% accuracy_score(y_test, xgb_pred))
print("F-1 score: %s"% f1_score(y_test, xgb_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, xgb_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, xgb_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, xgb_pred))
print("\n")
print("** AB Classifier Scores (OVERALL testset)")
print("Precision: %s"% precision_score(y_test, abc_pred))
print("Recall: %s"% recall_score(y_test, abc_pred))
print("Accuracy score: %s"% accuracy_score(y_test, abc_pred))
print("F-1 score: %s"% f1_score(y_test, abc_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, abc_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, abc_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, abc_pred))


# In[12]:


import numpy as np
ones_index = y_test.nonzero()
x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test, ones_index)[0]

xgb_pred_oneonly = xgb.predict(x_ones)
abc_pred_oneonly = abc.predict(x_ones)


# Evaluate stats for testset with label 1 (DENIED cases only)

# In[13]:


print("** XGB Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s" % precision_score(y_ones, xgb_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, xgb_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, xgb_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, xgb_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, xgb_pred_oneonly))
print("\n")
print("** AB Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, abc_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, abc_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, abc_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, abc_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, abc_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, abc_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, abc_pred_oneonly))


# In[14]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test, zeros_index)
xgb_pred_zeroonly = xgb.predict(x_zeros)
abc_pred_zeroonly = abc.predict(x_zeros)


# 
# Evaluate stats for testset with label 0 (CERTIFIED cases only)
# 

# In[15]:


print("** XGB Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s" % precision_score(y_zeros, xgb_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, xgb_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, xgb_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, xgb_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, abc_pred_zeroonly))
print("\n")
print("** AB Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, abc_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, abc_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, abc_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, abc_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, abc_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, abc_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, abc_pred_zeroonly))


# Show ratio between the two classes in the entire dataset (before split into train and test set)

# In[116]:


status_stats = df.groupby('CASE_STATUS').size().to_frame()
status_stats['percentage'] = status_stats[0] / status_stats[0].sum()
print(status_stats)

