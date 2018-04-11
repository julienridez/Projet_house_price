# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:53:47 2018

@author: nisha
"""

# Analyse des datas
import numpy as np
import pandas as pd

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn import manifold, datasets

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(42)

train = pd.read_csv('train.csv')
kaggle = pd.read_csv('test.csv')



correlation = train.corr()
correlation = correlation['SalePrice'].abs().sort_values(ascending=False)
correlation

train = (train.drop(['PoolArea', 'MSSubClass','MoSold','3SsnPorch','LowQualFinSF', 
                     "MiscVal", 'BsmtHalfBath','BsmtFinSF2', 'PoolQC', 'Fence',
                     'Alley','GarageFinish','LandContour','LotConfig','RoofMatl',
                     'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                     'MasVnrArea','LowQualFinSF',
                     'Functional','GarageType','GarageFinish',
                     'PavedDrive','EnclosedPorch','3SsnPorch','ScreenPorch',
                     'MiscFeature','MiscVal','MoSold','YrSold','SaleType',
                     'GarageArea', 'GarageYrBlt','HalfBath',
                     'Foundation' ,'YearRemodAdd','Utilities',
                     'LotFrontage','CentralAir', 'OpenPorchSF', 
                     'WoodDeckSF',
                     'SaleCondition','LandSlope','FullBath','Alley','Electrical' ], axis = 1))

kaggle = (kaggle.drop(['PoolArea', 'MSSubClass','MoSold','3SsnPorch','LowQualFinSF', 
                     "MiscVal", 'BsmtHalfBath','BsmtFinSF2', 'PoolQC', 'Fence',
                     'Alley','GarageFinish','LandContour','LotConfig','RoofMatl',
                     'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                     'MasVnrArea','LowQualFinSF',
                     'Functional','GarageType','GarageFinish',
                     'PavedDrive','EnclosedPorch','3SsnPorch','ScreenPorch',
                     'MiscFeature','MiscVal','MoSold','YrSold','SaleType',
                     'GarageArea', 'GarageYrBlt','HalfBath',
                     'Foundation' ,'YearRemodAdd','Utilities',
                     'LotFrontage','CentralAir', 'OpenPorchSF', 
                     'WoodDeckSF',
                     'SaleCondition','LandSlope','FullBath', 'Alley', 
                     'Electrical','MSZoning', 'BsmtFinSF1', 'BsmtUnfSF', 
                     'TotalBsmtSF', 'BsmtFullBath', 'GarageCars'], axis = 1))

#print(train.columns)

#one_hot_encoded_training_predictors = pd.get_dummies(train)
#one_hot_encoded_test_predictors = pd.get_dummies(kaggle)
#
#one_hot_encoded_training_predictors.columns
#
#set1 = set(one_hot_encoded_training_predictors.columns.tolist())
#set2 = set(one_hot_encoded_test_predictors.columns.tolist())
##print(set2.difference(set1))
##print(set1.difference(set2))
#dataCol_commun = list(set1 and set2)
#
##one_hot_encoded_training_predictors= (one_hot_encoded_training_predictors.drop(['Condition2_RRNn', 'Heating_OthW', 'HouseStyle_2.5Fin', 'Condition2_RRAn','Condition2_RRAe'], axis = 1)) 
##    
#one_hot_encoded_training_predictors = one_hot_encoded_training_predictors[dataCol_commun]
#one_hot_encoded_test_predictors = one_hot_encoded_test_predictors[dataCol_commun] 

dataY_train = train['SalePrice']
print(dataY_train)


def convert_data(dataset):
    le = preprocessing.LabelEncoder()
    categorie=['ExterCond','ExterQual','OverallQual','OverallCond','BsmtQual', 'BsmtCond', 'BsmtExposure',
                         'BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual', 'GarageCond',
                        ]
    
    for col in dataset.columns:
        if dataset[col].dtype == object:
            dataset[col].fillna(value='None', inplace = True)
            if col in categorie:
                dataset = pd.concat([dataset, pd.get_dummies(dataset[col], prefix=col)], axis=1)
            else:
                dataset[col]=le.fit_transform(dataset[col])
            
        elif dataset[col].dtype == int or dataset[col].dtype ==float :
            dataset[col].fillna(value=dataset[col].median(), inplace = True)
        
    return dataset


def drop_cat(dataset):
    datatrain2 = convert_data(dataset) 
    categorie=['ExterCond','ExterQual','OverallQual','OverallCond','BsmtQual', 'BsmtCond', 'BsmtExposure',
                         'BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual', 'GarageCond',
                        ]
    for col in categorie:
        datatrain2 = datatrain2.drop(col, axis=1)
        #print(datatrain2.head(5))
    return datatrain2
  
train = drop_cat(train)    
print(train.head(2))
test=drop_cat(kaggle)
print(test.head(2))

#%%
#train.info()
#kaggle.info()
#train.columns[train.isnull().any()]
#kaggle.columns[kaggle.isnull().any()] #affiche les colonne où y a des NA
#train= train.dropna() #supprime les ligne contenant des NA, mais supprime aussi des lignes donc c'est nul.
#kaggle=kaggle.dropna()

#%%
#train.info()
##kaggle.info()
#
#numerique = ['LotArea','OverallQual','YearBuilt', 
#             '1stFlrSF','2ndFlrSF', 'GrLivArea', 
#             'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotRmsAbvGrd', 
#             'Fireplaces']
#for name in numerique:
#        train[name] = np.nan_to_num(train[name]) #remplace les valeurs nan en 0 et infin par val max/min

#numerique1 = ['LotArea ', 'OverallQual', 'YearBuilt ', 
#             'BsmtFinSF1 ', 'BsmtUnfSF', 'TotalBsmtSF', 
#             '1stFlrSF ', '2ndFlrSF','GrLivArea', 
#             'BsmtFullBath', 'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd', 
#             'Fireplaces','GarageCars']
#for name in numerique:
#        df[name] = np.nan_to_num(df[name]) #remplace les valeurs nan en 0 et infin par val max/min
#%%
#renommer les colonnes
#train.rename(columns={"Cabin":"num_cabin", })
#test.rename(columns={"Cabin":"num_cabin", })



                                                            
#
dataX_train = train
dataY_train = train['SalePrice']
dataX_test = kaggle


#%%

#ma_liste = []
#
#for col in train.keys() :   
#    le = le.fit_transform([train].astype(str))
#    le.fit(train[col])
#    print(le.transform(train[col]))


#enc = OneHotEncoder()
#enc.fit(train)  
#
#print(enc.n_values_)
#
#print(enc.feature_indices_)
#
#enc.transform([[0, 1, 1]]).toarray()




#Lineaireregression
data_linear = linear_model.LinearRegression()
##on entraine:
data_linear.fit(dataX_train, dataY_train)
###on prédit:
data_test_lr = data_linear.predict(dataX_test)
data_test_lr = pd.DataFrame(data_test_lr)
print('LR:', data_linear.score(dataX_train, dataY_train))
##0.84
#0.173 KAGGLE

#isolationforest

#clf = IsolationForest(max_samples=100, random_state=rng)
#clf.fit(dataX_train,dataY_train)
#y_pred_train = clf.predict(dataX_train)
#y_pred_test = clf.predict(dataX_test)
#print(y_pred_train)
#print(y_pred_test)


NaiveGaussian
gnb = GaussianNB()
data_NBayes = gnb.fit(dataX_train, dataY_train).predict(dataX_test)
print('NB:', gnb.score(dataX_train, dataY_train))
0.57

Arbrededecision
data_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 30)

data_DTC = data_tree.fit(dataX_train, dataY_train).predict(dataX_test)
print('Tree:', data_tree.score(dataX_train, dataY_train))
0.06
0.29 KAGGLE


#random forest 
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#rf.fit(dataX_train, dataY_train)
## Predict on new data
#predictions = rf.predict(dataX_test)


#def RandomForestClassifierFunction(dataX_train,dataX_test,dataY_train,parameters):
#
#    acc_scorer = make_scorer(accuracy_score)
#    clf = RandomForestClassifier()
#    grid_obj = GridSearchCV(clf, parameters,scoring=acc_scorer)
#    grid_obj = grid_obj.fit(X_train, dataY_train)
#    clf = grid_obj.best_estimator_
#    clf.fit(dataX_train,dataY_train)
#    prediction = clf.predict(dataX_test)
#    print (accuracy_score(y_test,prediction))
#    return (accuracy_score(y_test,prediction)
#    parameters={"criterion": ['entropy','gini']}
#RandomForestClassifierFunction(dataX_train,dataX_test,dataY_train,parameters)

#Essai ramdom forest regressor
#dataX_train, dataY_train = make_regression(n_features=4, n_informative=2,
#                      random_state=0, shuffle=False)
#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#           oob_score=False, random_state=0, verbose=0, warm_start=False)
#regr = RandomForestRegressor(max_depth=2, random_state=0)
#regr.fit(dataX_train, dataY_train).predict(dataX_test)

#TSNE
#
#def tsne_function(data, n_components):
#    tsne = manifold.TSNE(n_components= 2, perplexity=30, init='random',random_state=0)
#    Y = tsne.fit_transform(train)
#    plt.figure()
#    plt.scatter(Y[:,0], Y[:,1])
#    plt.show()
#    r = tsne.predict()
#
#tsne_function(train, 2)



#FICHIER CVS POUR ARBRE DE DECISION
#submission = pd.DataFrame({'Id': dataX_test.Id, 'SalePrice': data_DTC})
#submission.to_csv('KagglePrix', index=False)

#FICHIER CSV POUR RL
#submissionLR = pd.DataFrame({'Id': dataX_test.Id, 'SalePrice': data_test_lr })
#print(type(data_test_lr), data_test_lr, submissionLR)
#submissionLR.to_csv('KagglePriix', index=False)

