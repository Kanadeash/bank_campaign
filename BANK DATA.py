#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing librariese
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#read data into dataframe
import pandas as pd
import numpy as np
df = pd.read_csv('D:/EDA/bank_marketing_updated_v1.csv')


# In[5]:


df.head()


# In[ ]:





# In[7]:


df.info()


# In[4]:


df = pd.read_csv('D:/EDA/bank_marketing_updated_v1.csv',skiprows=2)


# In[8]:


df.head()


# In[ ]:





# In[5]:


#drop column
df.drop(['customerid'],axis=1,inplace=True)


# In[12]:


df.head()


# In[6]:


#split column
df['job']=df.jobedu.apply(lambda x: x.split(",")[0])


# In[ ]:





# In[18]:


df.head()


# In[7]:


df['education']=df.jobedu.apply(lambda x: x.split(",")[1])


# In[20]:


df.head()


# In[8]:


df.drop(['jobedu'],axis=1,inplace=True)


# In[22]:


df.head()


# In[9]:


df1['month_name']=df1.month.apply(lambda x: x.split(",")[0])


# In[27]:


df1.head()


# In[28]:


df1['year']=df1.month.apply(lambda x: x.split(",")[1])


# In[29]:


df1.head()


# In[10]:


df[df.month.apply(lambda x: isinstance(x,float)==True)]


# In[9]:


df.month.isnull().sum()


# In[11]:


#find missing valuese in each column
df.isnull().sum()


# # Handling Missing Valuese

# In[28]:


df.age.isnull().sum()


# In[30]:


df.shape


# In[31]:


#percentage of null valuese in age column
100*20/45211


# In[9]:


#dropping null records
df1=df[~df.age.isnull()].copy()


# In[14]:


df1.shape


# In[15]:


#compute null valuse
df1.month.value_counts(normalize=True)


# In[10]:


#checking mode of month
month_mode=df1.month.mode()


# In[13]:


#store mode in new variable
month_mode=df1.month.mode()[0]


# In[14]:


#filling missing valuese
df1.month.fillna(month_mode,inplace=True)


# In[15]:


df1.month.isnull().sum()


# In[30]:


#count missing values of response column
df1.response.isnull().sum()


# In[16]:


df1=df[~df.response.isnull()]


# In[17]:


df1.response.isnull().sum()


# In[25]:


df1.pdays.describe()


# In[18]:


df1.loc[df1.pdays<0,"pdays"]=np.NAN


# In[27]:


df1.pdays.describe()


# # Handling outliers

# In[28]:


#Age variable
df1.age.describe()


# In[36]:


#create boxplot for age variale
sns.boxplot(df1.age)
plt.show()


# In[37]:


#checking balance variable
df1.balance.describe()


# In[38]:


#create boxplot for balance variable
sns.boxplot(df1.balance)
plt.show()


# In[39]:


#different quantile
df1.balance.quantile([0.5,0.7,0.9,0.95,0.99])


# In[40]:


df1[df1.balance>15000].describe()


# In[32]:


#Duratuion variable
df1.duration.describe()


# In[33]:


df1.head(10)


# In[48]:


df1.duration.head()


# In[19]:


df1.duration=df1.duration.apply(lambda x: float(x.split()[0])/60 if x.find("sec")>0 else float(x.split()[0]))


# In[46]:


df1.duration.describe()


# In[41]:


df1.head()


# # Univariate Analysis

# # categorical unorderd variable

# # Marital Status

# In[12]:


df1.type()


# In[16]:


df1.marital.value_counts()


# In[ ]:





# In[15]:


print(df1.marital.dtypes)


# In[12]:


df1.marital.value_counts(normalize=True)


# In[13]:


df1.marital.value_counts(normalize=True).plot.bar()
plt.show()


# # Job

# In[14]:


df1.job.value_counts(normalize=True)


# In[15]:


df1.job.value_counts(normalize=True).plot.bar()
plt.show()


# # Categorical ordered variable

# # Education

# In[16]:


df1.education.value_counts(normalize=True)


# In[17]:


df1.education.value_counts(normalize=True).plot.pie()
plt.show()


# # Poutcome

# In[18]:


df1.poutcome.value_counts(normalize=True)


# In[19]:


df1.poutcome.value_counts(normalize=True).plot.bar()
plt.show()


# # response Variable

# In[20]:


df1.response.value_counts(normalize=True)


# In[21]:


df1.response.value_counts(normalize=True).plot.pie()
plt.show()


# # #Bivariate Aanalysis

# # numeric-numric analysis

# In[26]:


#salary-balance
df1.plot.scatter(x='salary',y='balance')
plt.show()


# In[23]:


#age-balance
df1.plot.scatter(x='age',y='balance')
plt.show()


# In[27]:


#create pair plot 
sns.pairplot(data=df1,vars=['salary','balance','age'])
plt.show()


# In[30]:


#quantify using correlation values
df1[['age','salary','balance']].corr()


# # Correlation heatmap

# In[32]:


sns.heatmap(df1[['age','salary','balance']].corr(),annot=True,cmap='Reds')


# # Numerical-categorical variale

# # salary vs response

# In[33]:


#for each value of response whats the mean value of salaey is?
df1.groupby('response')['salary'].mean()


# In[34]:


df1.groupby('response')['salary'].median()


# In[36]:


sns.boxplot(data=df1,x='response',y='salary')
plt.show()


# # balance vs response

# In[37]:


sns.boxplot(data=df1,x='response',y='balance')
plt.show()


# In[38]:


df1.groupby('response')['balance'].mean()


# In[39]:


df1.groupby('response')['salary'].median()


# In[50]:


#lets create function to calculate 75%
def per75(x):
    np.quantile(x,0.75)
#lets see mean ,median,75% of balance against response
df1.groupby('response')['balance'].aggregate(["mean", "median", per75]).plot.bar()


# # Education vs Salary

# In[51]:


sns.boxplot(data=df1,x='education',y='salary')
plt.show()


# In[52]:


#lets see mean ,median,75% of salary against education
df1.groupby('education')['salary'].aggregate(["mean", "median"]).plot.bar()


# # Job vs Salary

# In[58]:


sns.bar(data=df1,x='job',y='salary')
plt.show()


# In[62]:


plt.plot(data=df1,x='job',y='salary',bar())
plt.show()


# # Categorical-categorical variable

# In[ ]:





# In[65]:


#calculate response rate(no of ones divided by total ones)
df1['response_flag']=np.where(df1.response=='yes',1,0)


# In[ ]:





# In[ ]:





# In[66]:


df1.response_flag.value_counts()


# # Education vs Response rate

# In[67]:


df1.groupby(['education'])['response_flag'].mean()


# # Marital vs Response rate

# In[68]:


df1.groupby(['marital'])['response_flag'].mean()


# In[69]:


df1.groupby(['marital'])['response_flag'].mean().plot.bar()
plt.show()


# # Loans vs Response rate

# In[71]:


df1.groupby(['loan'])['response_flag'].mean()


# In[72]:


df1.groupby(['loan'])['response_flag'].mean().plot.bar()


# # Housing loans vs Response rate

# In[74]:


df1.groupby(['housing'])['response_flag'].mean().plot.bar()


# # Age vs Response

# In[75]:


sns.boxplot(data=df1,x='age',y='response')


# In[77]:


#making bucket from age column
df1['age_group']=pd.cut(df1.age,[0,30,40,50,60,999],labels=['<30','30-40','40-50','50-60','60+'])


# In[79]:


df1.age_group.value_counts(normalize=True)


# In[80]:


df1.age_group.value_counts(normalize=True).plot.bar()


# In[87]:


df1.groupby(['age_group'])['response_flag'].mean()


# In[88]:


plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
df1.age_group.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
df1.groupby(['age_group'])['response_flag'].mean().plot.bar()
plt.show()


# # Multivariate Analysis

# # Education vs marital vs response

# In[92]:


res=pd.pivot_table(data=df1,index='education',columns='marital',values='response_flag')
res


# In[94]:


sns.heatmap(res,annot=True,cmap='RdYlGn')


# # Job vs marital vs response

# In[95]:


res=pd.pivot_table(data=df1,index='job',columns='marital',values='response_flag')
res


# In[96]:


sns.heatmap(res,annot=True,cmap='RdYlGn')


# # Education vs Poutcome vs response

# In[97]:


res=pd.pivot_table(data=df1,index='job',columns='poutcome',values='response_flag')
res


# In[98]:


sns.heatmap(res,annot=True,cmap='RdYlGn')


# In[20]:


df1.to_csv('siri1.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




