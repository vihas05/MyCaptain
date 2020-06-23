#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
print('Python: {}' .format(sys.version))
import scipy
print('Sci {}' .format(scipy.__version__))
import numpy
print('num: {}' .format(numpy.__version__))
import matplotlib
print('mat: {}' .format(matplotlib.__version__))
import pandas
print('Pandas: {}' .format(pandas.__version__))
import sklearn
print('Sklearn: {}' .format(sklearn.__version__))


# In[5]:


import pandas 
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[7]:


a = pandas.read_csv("iris.csv")


# In[8]:


a


# In[10]:


print(a.shape)


# In[11]:


print(a.head(20))


# In[12]:


print(a.describe())


# In[15]:


print(a.groupby('species').size())


# In[16]:


#univarient
a.plot(kind='box',subplots=True, layout=(2,2), sharex=False ,sharey=False)


# In[17]:


a.hist()
pyplot.show()


# In[18]:


#multivarient
scatter_matrix(a)
pyplot.show()


# In[19]:


array = a.values
X = array[:,0:4]
Y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y, test_size=0.2,random_state=1)


# In[25]:


#LogisticsRegression
#Linear Discriminant analysis
#K-nearist neighbor
#classification and Regression tress
#Gassian naive bayes
#Support Vector MAchines

models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[29]:


results =[]
names = []
for name,model in models :
    kfold = StratifiedKFold(n_splits=10,random_state=1)
    cv_result= cross_val_score(model , X_train , Y_train , cv=kfold , scoring  = 'accuracy')
    results.append(cv_result)
    names.append(name)
    print('%s : %f(%f)'%(name,cv_result.mean(),cv_result.std()))


# In[31]:


pyplot.boxplot(results,labels=names)
pyplot.title("Algorithm Comparison")
pyplot.show()


# In[33]:


model = SVC (gamma='auto')
model.fit(X_train,Y_train)
prediciton = model.predict(X_validation)


# In[35]:


print(accuracy_score(Y_validation , prediciton))
print(confusion_matrix(Y_validation , prediciton))
print(classification_report(Y_validation , prediciton))


# In[ ]:




