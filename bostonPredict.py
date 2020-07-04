import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor        #决策树回归算法
from sklearn.ensemble import RandomForestRegressor    #随机森林回归算法
from sklearn.model_selection import train_test_split  #划分训练集和测试机
from sklearn.linear_model import Ridge                #岭回归
from sklearn.linear_model import Lasso                #LASSO回归


boston = load_boston()
boston
boston.keys()
#查看每一个key值#
print('data值为：',boston.data)
print('target值为：',boston.target)
print('feature_names值为：',boston.feature_names)
dfboston = pd.DataFrame(boston['data'],columns=boston['feature_names']) #重新设置数据格式
dfboston.head()
dfboston.info()
print(dfboston.info())
dfboston.describe([0.01,0.09])#设定数值型特征的统计量，默认是[.25, .5, .75],也就是返回25%，50%，75%数据量时的数字
print(dfboston.describe([0.01,0.09]))
#相关性检验
plt.figure(figsize=(10,8))#相关性图
sns.heatmap(dfboston.corr())
#plt.show()
dfboston.corr()[np.abs(dfboston.corr())>0.5]#获取相关性绝对值大于0.5的
print(dfboston.corr()[np.abs(dfboston.corr())>0.5])
#多重共线性判断
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(dfboston.values,i) for  i in range(dfboston.shape[1])]
print(vif)#值越大，说明存在明显的验证多重共线特征
#加入房价标签，探索特征与标签关系
dfboston['Price'] = pd.Series(boston['target'])
dfboston.describe()
print(dfboston.describe())
for feature in boston['feature_names']: #通过图表查看标签与特征间线性关系
    sns.regplot(x=feature,y='Price',data=dfboston)
    plt.xlabel(feature)
    #plt.show()
dfboston.corr()#查看房价与特征相关性
print(dfboston.corr())#LSTAT 最大负相关-0.737663，RM最大正相关0.695360
#==================建模与评估
#====构建决策树
Xtrain,Xtest,Ytrain,Ytest= train_test_split(dfboston.iloc[:,:-1],dfboston.iloc[:,-1],test_size=0.2)
dtr_0 = DecisionTreeRegressor()
dtr_0.fit(Xtrain,Ytrain)
dtr_0.score(Xtrain,Ytrain)
dtr_0.score(Xtest,Ytest)
dtr_0.feature_importances_
print("训练集得分",dtr_0.score(Xtrain,Ytrain))#训练集得分 1.0
print("测试集得分",dtr_0.score(Xtest,Ytest))#测试集得分 0.5140914168707086
print("特征重要性",dtr_0.feature_importances_)
#决策树剪枝，网格搜索，寻找最优参数组合
from sklearn.model_selection import GridSearchCV
#param_grid = {'criterion':['mse','friedman_mse','mse'],'max_depth':range(2,13),'min_samples_leaf':range(1,10),'min_samples_split':range(2,20)}
#GR= GridSearchCV(DecisionTreeRegressor(),param_grid,cv=5)
#GR.fit(Xtrain,Ytrain)
#GR.best_params_
#GR.best_score_
#print("参数",GR.best_params_)#参数 {'criterion': 'friedman_mse', 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2}
#print("准确率",GR.best_score_)#准确率 0.7894199731969926
#最优决策树组合代入模型
#dtr_1 = DecisionTreeRegressor(criterion='friedman_mse',max_depth=4,min_samples_leaf=1,min_samples_split=2)
#dtr_1.fit(Xtrain,Ytrain)
#dtr_1.score(Xtrain,Ytrain)
#dtr_1.score(Xtest,Ytest)
#dtr_1.feature_importances_
#print("最优决策树训练集得分",dtr_1.score(Xtrain,Ytrain))#最优决策树训练集得分 0.896365622899498
#print("最优决策树测试集得分",dtr_1.score(Xtest,Ytest))#最优决策树测试集得分 0.7586706930268079
#print("最优决策树提取出的关键特征",dtr_1.feature_importances_)#关键特征RM=0.61619267,LSTAT=0.19705723
#=======构建随机森林回归
rfr_0=RandomForestRegressor(n_estimators=10)
rfr_0.fit(Xtrain,Ytrain)
rfr_0.score(Xtrain,Ytrain)
rfr_0.score(Xtest,Ytest)
print("随机森林训练集的分",rfr_0.score(Xtrain,Ytrain))#随机森林训练集的分 0.9734746620125392
print("随机森林测试集得分",rfr_0.score(Xtest,Ytest))#随机森林测试集得分 0.8822827362981865
#交叉验证
from sklearn.model_selection import cross_val_score
score = []
for n in range(1,200):
    rfr = RandomForestRegressor(n_estimators=n,n_jobs=1)
    rfrscore = cross_val_score(rfr,boston.data,boston.target,cv=5).mean()
    score.append(rfrscore)
print(max(score),score.index(max(score)))#最大得分=0.6546851250183744 最大得分下标=11
plt.figure(figsize=(20,5))
plt.plot(range(1,200),score)
plt.show()
rfr_1 = RandomForestRegressor(n_estimators=174)
rfr_1.fit(Xtrain,Ytrain)
rfr_1.score(Xtrain,Ytrain)
rfr_1.score(Xtest,Ytest)
rfr_1.feature_importances_
print("最优随机森林训练集的分",rfr_1.score(Xtrain,Ytrain))#最优随机森林训练集的分 0.9803355097743902
print("最优随机森林测试集得分",rfr_1.score(Xtest,Ytest))#最优随机森林测试集得分 0.907664441661649
print("最优随机森林提取出的关键特征",rfr_1.feature_importances_)#关键特征RM=0.39521684,LSTAT=0.4185393
