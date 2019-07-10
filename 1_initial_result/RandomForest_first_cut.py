#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2015 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[3]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_FINAL.pickle')
df.drop(columns=['EMPLOYER_CITY','EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# Show first 5 lines of the dataset

# In[4]:


df.head()


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[6]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['CASE_STATUS'])

X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[53]:


ones_index


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# Check value counts to verify stratify split

# In[11]:


X_train.head()


# Train DecisionTreeClassifier and predict using X_test

# In[13]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
# print(clf.feature_importances_)
print(clf.score(X_test, y_test))
tree_pred = clf.predict(X_test)
print(tree_pred)
# for i in range(len(y_hat)):
#     if y_test[i]==1:
#         print("{}:: {}".format(y_test[i],y_hat[i]))


# Train RandomForestClassifier and predict using X_test

# In[15]:


from sklearn.ensemble import RandomForestClassifier


# from sklearn.feature_extraction.text import CountVectorizer

rfc = RandomForestClassifier()
# vec = CountVectorizer()
# X_train_t = vec.fit_transform(X_train)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)


# In[110]:


from sklearn.metrics import classification_report, confusion_matrix
print("** Decision tree Classifier Stats (OVERALL testset)")
print(classification_report(y_test, tree_pred))

print("** Random Forest Classifier Stats (OVERALL testset)")
print(classification_report(y_test, rfc_pred))


# Output the feature importance vector and sort by importance

# In[21]:


import operator
# print(len(rfc.feature_importances_))
# print(len(X_train.columns))

nonzero_features = {}
for i in range(len(rfc.feature_importances_)):
    if rfc.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = rfc.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 
for i in range(len(sorted_result)):
    print(sorted_result[-(i+1)])


# In[ ]:


import pickle
with open('/tmp/feature_importances_first_800k_rfc.pickle', 'wb') as f:
    pickle.dump(sorted_result, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('/tmp/feature_importances_first_800k_rfc_dict.pickle', 'wb') as f:
    pickle.dump(nonzero_features, f, protocol=pickle.HIGHEST_PROTOCOL)


# Output the trained model to a pickle file in case we need it in the future

# In[122]:


from sklearn.externals import joblib
joblib.dump(rfc, 'rfc.pkl', compress=9)
joblib.dump(clf, 'tree.pkl', compress=9)

# import cPickle
# with open('rfc_cpickle.pkl', 'wb') as pkl:
#     cPickle.dump(rfc, pkl)
# with open('tree_cpickle.pkl', 'wb') as pkl:
#     cPickle.dump(clf, pkl)    


# To load the pre-trained model, run the cell below.

# In[124]:


rfc = joblib.load('rfc.pkl')
clf = joblib.load('tree.pkl')

# with open('rfc_cpickle.pkl', 'rb') as pkl:
#     rfc = cPickle.load(pkl)
# with open('tree_cpickle.pkl', 'rb') as pkl:
#     clf = cPickle.load(pkl)


# Evaluate stats for the entire test set

# In[125]:


from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix
# print(rfc_pred)

print("** Decision tree Classifier Scores (OVERALL testset)")
print("Precision: %s" % precision_score(y_test, tree_pred))
print("Recall: %s"% recall_score(y_test, tree_pred))
print("Accuracy score: %s"% accuracy_score(y_test, tree_pred))
print("F-1 score: %s"% f1_score(y_test, tree_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, tree_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, tree_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, tree_pred))
print("\n")
print("** Random Forest Classifier Scores (OVERALL testset)")
print("Precision: %s"% precision_score(y_test, rfc_pred))
print("Recall: %s"% recall_score(y_test, rfc_pred))
print("Accuracy score: %s"% accuracy_score(y_test, rfc_pred))
print("F-1 score: %s"% f1_score(y_test, rfc_pred))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_test, rfc_pred, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_test, rfc_pred, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_test, rfc_pred))


# In[ ]:


import numpy as np
ones_index = y_test.nonzero()
x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test, ones_index)[0]

tree_pred_oneonly = clf.predict(x_ones)
rfc_pred_oneonly = rfc.predict(x_ones)


# Evaluate stats for testset with label 1 (DENIED cases only)

# In[120]:


print("** Decision tree Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s" % precision_score(y_ones, tree_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, tree_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, tree_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, tree_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, tree_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, tree_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, tree_pred_oneonly))
print("\n")
print("** Random Forest Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, rfc_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, rfc_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, rfc_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, rfc_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, rfc_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, rfc_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, rfc_pred_oneonly))


# In[114]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test, zeros_index)
tree_pred_zeroonly = clf.predict(x_zeros)
rfc_pred_zeroonly = rfc.predict(x_zeros)


# 
# Evaluate stats for testset with label 0 (CERTIFIED cases only)
# 

# In[119]:


print("** Decision tree Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s" % precision_score(y_zeros, tree_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, tree_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, tree_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, tree_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, tree_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, tree_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, rfc_pred_zeroonly))
print("\n")
print("** Random Forest Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, rfc_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, rfc_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, rfc_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, rfc_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, rfc_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, rfc_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, rfc_pred_zeroonly))


# Show ratio between the two classes in the entire dataset (before split into train and test set)

# In[116]:


status_stats = df.groupby('CASE_STATUS').size().to_frame()
status_stats['percentage'] = status_stats[0] / status_stats[0].sum()
print(status_stats)

