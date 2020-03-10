# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:32:24 2020

@author: Vasanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the csv file
df=pd.read_csv(r"C:\Users\Vasanth\Desktop\New folder\train.csv")
df1=df.copy()
#visulizing the data
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
#filling the null coloumns
df1=df1.drop(['Alley'],axis=1)
df1=df1.drop(['MiscFeature'],axis=1)
df1=df1.drop(['PoolQC'],axis=1)
df1=df1.drop(['Fence'],axis=1)
df1['FireplaceQu']=df1['FireplaceQu'].fillna(df1['FireplaceQu'].mode()[0])
df1['BsmtQual']=df1['BsmtQual'].fillna(df1['BsmtQual'].mode()[0])
df1['BsmtCond']=df1['BsmtCond'].fillna(df1['BsmtCond'].mode()[0])
df1['BsmtExposure']=df1['BsmtExposure'].fillna(df1['BsmtExposure'].mode()[0])
df1['BsmtFinType1']=df1['BsmtFinType1'].fillna(df1['BsmtFinType1'].mode()[0])
df1['BsmtFinType2']=df1['BsmtFinType2'].fillna(df1['BsmtFinType2'].mode()[0])
df1['LotFrontage']=df1['LotFrontage'].fillna(df1['LotFrontage'].mean())
df1['GarageYrBlt']=df1['GarageYrBlt'].fillna(df1['GarageYrBlt'].mean())
df1['GarageType']=df1['GarageType'].fillna(df1['GarageType'].mode()[0])
df1['GarageFinish']=df1['GarageFinish'].fillna(df1['GarageFinish'].mode()[0])
df1['GarageQual']=df1['GarageQual'].fillna(df1['GarageQual'].mode()[0])
df1['GarageCond']=df1['GarageCond'].fillna(df1['GarageCond'].mode()[0])
df1=df1.dropna(axis=0)
df1=df1.drop(['Id'],axis=1)

#categorical and feature engineerin
df3=df1.copy()
for col in df1.columns:
    if(df1[col].dtypes=='object'):
        a=pd.get_dummies(df1[col],drop_first=True)
        df3=pd.concat([df3,a],axis=1)
        df3=df3.drop(col,axis=1)
        
#splitting train test set
X=df3.iloc[:,0:-1]
Y=df3['SalePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#predicting the values

'''from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)'''

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)