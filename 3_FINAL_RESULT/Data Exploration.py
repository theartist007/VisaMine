#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_raw = pd.read_csv('/Users/anuprava/PycharmProjects/Udacity_ML_Capstone/data/H1B_RAW_15-18.csv', encoding='utf-8')


# In[3]:


totalCount  = df_raw.shape[0]
nullCount = len(df_raw[df_raw['WAGE_RATE_OF_PAY_TO'].isnull()])
zeroCount = len(df_raw[df_raw['WAGE_RATE_OF_PAY_TO']==0.0])

print("{}% of samples are missing feature WAGE_RATE_OF_PAY_TO.".format(round(nullCount/float(totalCount)*100)))
print("{}% of samples have value 0 for feature WAGE_RATE_OF_PAY_TO.".format(round(zeroCount/float(totalCount)*100)))


# In[11]:


print("There are {} unique values for SOC Code.".format(len(df_raw.SOC_CODE.unique())))
print("There are {} unique values for NAICS Code.".format(len(df_raw.NAICS_CODE.unique())))




# In[ ]:




print("There are {} unique values for SOC Code.".format(len(df_raw.SOC_CODE.unique())))
print("There are {} unique values for NAICS Code.".format(len(df_raw.NAICS_CODE.unique())))


# In[37]:


grouped = df_raw.groupby('SOC_CODE').count().reset_index()
g = grouped.sort_values('Unnamed: 0', ascending=False)['Unnamed: 0'][30:]
count_codes_other_than_top_30 = sum(g)
print(count_codes_other_than_top_30)
print("Only {}% of the total samples have SOC code other than top 30 SOC codes".format(count_codes_other_than_top_30/float(int(df_raw.shape[0]))*100))


# In[39]:


df_raw.groupby('CASE_STATUS').size()


# In[42]:


print("Only {}% of the total samples have DENIED label.".format(len(df_raw[df_raw['CASE_STATUS']=='DENIED'])/float(df_raw.shape[0])*100))


# In[4]:


df_raw.PREVAILING_WAGE.mean()


# In[5]:


df_raw.PREVAILING_WAGE.median()


# In[13]:


df_raw.PREVAILING_WAGE.describe()


# In[14]:


df_raw.WAGE_RATE_OF_PAY_FROM.describe()


# Remove Outliers and show stats again

# In[19]:


df_raw[(df_raw['WAGE_RATE_OF_PAY_FROM']>df_raw['WAGE_RATE_OF_PAY_FROM'].quantile(0.05)) & 
       (df_raw['WAGE_RATE_OF_PAY_FROM']<df_raw['WAGE_RATE_OF_PAY_FROM'].quantile(0.95))]\
        .WAGE_RATE_OF_PAY_FROM.describe()


# In[20]:


df_raw[(df_raw['PREVAILING_WAGE']>df_raw['PREVAILING_WAGE'].quantile(0.05)) & 
       (df_raw['PREVAILING_WAGE']<df_raw['PREVAILING_WAGE'].quantile(0.95))]\
        .PREVAILING_WAGE.describe()


# In[27]:


len(df_raw['PW_SOURCE_OTHER'].unique())
# df_raw.columns
df_raw.groupby('PW_SOURCE').size()


# In[47]:


status_stats = df_raw.groupby('CASE_STATUS').size().to_frame()
status_stats['% Distribution'] = status_stats[0] / status_stats[0].sum() * 100.0
status_stats.index.rename("Case Status", inplace=True)


# In[52]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

status_stats['% Distribution'].plot(kind='barh', colormap='tab10', figsize=(12,3)).invert_yaxis()
plt.legend(loc="lower right")
plt.title('% Distribution of Samples with Each Class')
plt.xticks(np.arange(0,110,step=10))
plt.xlabel('% Distribution')


# In[41]:


status_stats['% Distribution']


# In[ ]:




