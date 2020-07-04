import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

dataset = pd.read_csv('heart.csv')
dataset.columns
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values
dataset.describe([0.01,0.09])
print(dataset.describe([0.01,0.09]))
plt.figure(figsize=(10,8))#相关性图
sns.heatmap(dataset.corr())
#plt.show()
dataset.corr()[np.abs(dataset.corr())>0.5]#获取相关性绝对值大于0.5的
print(dataset.corr()[np.abs(dataset.corr())>0.5])
#多重共线性判断
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(dataset.values,i) for i in range(dataset.shape[1]-1)]
print(vif)#值越大，说明存在明显的验证多重共线特征
for feature in dataset.columns: #通过图表查看标签与特征间线性关系
    sns.regplot(x=feature,y='target',data=dataset)
    plt.xlabel(feature)
   # plt.show()
dataset.corr()#查看房价与特征相关性
print(dataset.corr())
from sklearn.model_selection import train_test_split  #划分训练集和测试机
Xtrain,Xtest,Ytrain,Ytest= train_test_split(dataset.iloc[:,:-1],dataset.iloc[:,-1],test_size=0.2)
print("========XTrain")
print(Xtrain)
print("=======YTrain")
print(Ytrain)
print("========XTest")
print(Xtest)
print("=======YTest")
print(Ytest)
from sklearn.tree import DecisionTreeRegressor        #决策树回归算法
from sklearn.ensemble import RandomForestRegressor    #随机森林回归算法
rfr_0=RandomForestRegressor(n_estimators=10)
rfr_0.fit(Xtrain,Ytrain)
rfr_0.score(Xtrain,Ytrain)
rfr_0.score(Xtest,Ytest)
rfr_0.feature_importances_
print("随机森林训练集的分",rfr_0.score(Xtrain,Ytrain))#随机森林训练集的分 0.8921666666666667
print("随机森林测试集得分",rfr_0.score(Xtest,Ytest))#随机森林测试集得分 0.24740259740259762
print("随机森林提取出的关键特征",rfr_0.feature_importances_)#关键特征RM=0.39521684,LSTAT=0.4185393
from sklearn.model_selection import cross_val_score
#score = []
#for n in range(1,200):
#    rfr = RandomForestRegressor(n_estimators=n,n_jobs=1)
#    rfrscore = cross_val_score(rfr,features,labels,cv=5).mean()
#    score.append(rfrscore)
#print(max(score),score.index(max(score)))#最大得分=0.07869665358726514 最大得分下标=44
#plt.figure(figsize=(20,5))
#plt.plot(range(1,200),score)
#plt.show()
rfr_1 = RandomForestRegressor(n_estimators=45)
rfr_1.fit(Xtrain,Ytrain)
rfr_1.score(Xtrain,Ytrain)
rfr_1.score(Xtest,Ytest)
rfr_1.feature_importances_
print("最优随机森林训练集的分",rfr_1.score(Xtrain,Ytrain))#最优随机森林训练集的分 0.9803355097743902
print("最优随机森林测试集得分",rfr_1.score(Xtest,Ytest))#最优随机森林测试集得分 0.28076854632410186
print("最优随机森林提取出的关键特征",rfr_1.feature_importances_)#关键特征RM=0.39521684,LSTAT=0.4185393