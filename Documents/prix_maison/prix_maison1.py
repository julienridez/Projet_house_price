import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
prix = pd.read_csv('sample_submission.csv') 
 

print(df.columns)
print(prix.columns)

correlation = train.corr()
correlation = correlation['SalePrice'].abs().sort_values(ascending= False)
print(correlation)

x = 'SalePrice'
y = 'OverallQual'
a = 'GrLivArea'


plt.figure()
sns.barplot(a,y,data=df)
sns.barplot(a,y,data=df)
