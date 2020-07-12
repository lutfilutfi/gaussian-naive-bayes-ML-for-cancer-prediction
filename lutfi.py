import pandas as pd
import numpy as np
#index=[n+1 for n in range(8)]
#print(index)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#df = pd.DataFrame(np.zeros(shape=(8,3)),index=index,columns=['A', 'B', 'C'])

data=load_breast_cancer()
#print(data['target_names'])
labeln=data['target_names']  #not necessary
#print(data['target'])
label=data['target']
#print(data['feature_names'])   
featuren=data['feature_names']  #not necessary
#print(data['data'])
feature=data['data']
df = pd.DataFrame(data['data'],columns=data['feature_names'])    
print(df.head())#not neccesary,simply to visualize a dataset
train,test,train_labels,test_labels=train_test_split(feature,label,test_size=.11,random_state=42)
#change the value of test to get higher  or lower accuracy
#train labels are the value whether tumour is malignant or benign and train is the values of features
print(pd.DataFrame(train).head())
print(pd.DataFrame(test).head())
gnb=GaussianNB()
model=gnb.fit(train,train_labels) #this trains the model using GNB
preds=gnb.predict(test)
print(pd.DataFrame(preds))
print(accuracy_score(preds,test_labels))

