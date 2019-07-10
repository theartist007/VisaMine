#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Load cleaned up version of dataset from 2012 to 2018.  Drop geographic information about employer and worksite to reduce dimensionality

# In[59]:


df = pd.read_pickle('/Users/ml/Desktop/Udacity_ML_Capstone/data/H1B_15-18_new.pickle')
df.drop(columns=['EMPLOYER_CITY','JOB_TITLE','EMPLOYER_NAME'], inplace=True)
df.drop(columns=['EMPLOYER_STATE','WORKSITE_STATE'], inplace=True)


# In[60]:


ones_df = df[df['CASE_STATUS']=='DENIED']
zeros_df = df[df['CASE_STATUS']=='CERTIFIED']


# In[61]:


# df = pd.concat([ones_df, zeros_df.sample(frac=0.1, random_state=99)])
df = pd.concat([ones_df, zeros_df.sample(frac=0.02, random_state=99)])


# In[62]:


ones_df.head()
print(ones_df.shape)


# In[63]:


zeros_df.head()
print(zeros_df.shape)


# We will use column 'CASE_STATUS' as our label.  DENIED will be 1 and CERTIFIED will be 0 after it gets incoded by label_encoder.
# 
# Normalize PREVAILING_WAGE using minmax scaler since it varies quite a bit.

# In[64]:


scaler = MinMaxScaler()
X = df.drop(columns='CASE_STATUS')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['CASE_STATUS'])

X[['PREVAILING_WAGE']] = scaler.fit_transform(X[['PREVAILING_WAGE']])
X = pd.get_dummies(X)


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# Check value counts to verify stratify split

# In[66]:


# print(y_train.value_counts())
# print(y_test.value_counts())
X_train.head()


# Train XGBoost and predict using X_test

# In[67]:


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
print(xgb.score(X_test, y_test))
xgb_pred = xgb.predict(X_test)
print(xgb_pred)


# In[68]:


from sklearn.metrics import classification_report, confusion_matrix
print("** XGBoost Classifier Stats (OVERALL testset)")
print(classification_report(y_test, xgb_pred))


# Evaluate stats for the entire test set

# In[70]:


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

# In[71]:


import numpy as np
ones_index = y_test.nonzero()
x_ones = X_test.iloc[ones_index]
y_ones = np.take(y_test, ones_index)[0]

xgb_pred_oneonly = xgb.predict(x_ones)


# In[72]:


print("** XGBoost Classifier Scores (Only rows with label DENIED->1)")
print("Precision: %s"% precision_score(y_ones, xgb_pred_oneonly))
print("Recall: %s"% recall_score(y_ones, xgb_pred_oneonly))
print("Accuracy score: %s"% accuracy_score(y_ones, xgb_pred_oneonly))
print("F-1 score: %s"% f1_score(y_ones, xgb_pred_oneonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_ones, xgb_pred_oneonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_ones, xgb_pred_oneonly))


# Evaluate stats for testset with label 0 (CERTIFIED cases only)

# In[73]:


zeros_index = np.arange(len(y_test))[(y_test==0)]

x_zeros = X_test.iloc[zeros_index]
y_zeros = np.take(y_test, zeros_index)
xgb_pred_zeroonly = xgb.predict(x_zeros)


# In[74]:


print("** XGBoost Classifier Scores (Only rows with label CERTIFIED->0)")
print("Precision: %s"% precision_score(y_zeros, xgb_pred_zeroonly))
print("Recall: %s"% recall_score(y_zeros, xgb_pred_zeroonly))
print("Accuracy score: %s"% accuracy_score(y_zeros, xgb_pred_zeroonly))
print("F-1 score: %s"% f1_score(y_zeros, xgb_pred_zeroonly))
print("F-beta score with beta=0.5: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.5))
print("F-beta score with beta=0.2: %s"% fbeta_score(y_zeros, xgb_pred_zeroonly, beta=0.2))
print("Confusion Matrix: \n%s"% confusion_matrix(y_zeros, xgb_pred_zeroonly))


# In[75]:


import datetime
datetime.datetime.now()


# Plot ROC curve for the trained XGBoost classifier

# In[81]:


from sklearn.metrics import roc_curve, auc,recall_score,precision_score
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

FlasePositive, TruePositive, _ = roc_curve(y_test, xgb_pred)
roc_auc = auc(FlasePositive, TruePositive)
plt.figure(figsize=(6,6))
lw = 2
plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False-Positive Rate')
plt.ylabel('True-Positive Rate')
plt.title('ROC curve')
plt.legend(loc="best")
plt.show()


# Plot feature importance (top 25) of encoded features in a horizontal bar chart

# In[77]:


importances = []
# = xgb.feature_importances_




import operator
nonzero_features = {}
for i in range(len(xgb.feature_importances_)):
    if xgb.feature_importances_[i] > 0.0:
        nonzero_features[list(X_train.columns)[i]] = xgb.feature_importances_[i]
sorted_result = sorted(nonzero_features.items(), key=operator.itemgetter(1)) 


# In[88]:


for i in range(len(sorted_result)):
    importances.append(sorted_result[-(i+1)])
    
fi = pd.DataFrame(importances, columns=['Feature Name', 'Feature Importance'])
fi = fi.set_index("Feature Name")
fi.iloc[:25].plot(kind='barh', figsize=(6,6)).invert_yaxis()
plt.legend(loc="lower right")
plt.title('Feature Name vs Feature Importance')
plt.xlabel('Feature Importance')


# In[79]:


fi.head(60)

