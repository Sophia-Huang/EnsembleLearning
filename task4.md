 ### 2.1.5 超参调优

**定义**

**模型参数**：模型参数是模型内部的配置变量，其值可以根据数据进行估计。

- 模型在进行预测时需要它们。
- 它们的值定义了可使用的模型。
- 他们是从数据估计或获悉的。
- 它们通常不由编程者手动设置。
- 他们通常被保存为学习模型的一部分。

**参数是机器学习算法的关键**。它们通常由过去的训练数据中总结得出。最优化算法是估计模型参数的有效工具。

**模型超参数**：模型超参数是模型外部的配置，其值无法从数据中估计。

- 它们通常用于帮助估计模型参数。
- 它们通常由人工指定。
- 他们通常可以使用启发式设置。
- 他们经常被调整为给定的预测建模问题。

**如果必须手动指定模型参数，那么它可能是一个模型超参数。**

**分类**

超参优化的方式有**网格搜索**和**随机搜索**。

**网格搜索**：网格搜索的思想非常简单，比如有2个超参数需要去选择，就把所有的超参数选择列出来分别做排列组合。然后针对每组超参数分别建立一个模型，然后选择测试误差最小的那组超参数。

**随即搜索**：网格搜索需要把参数空间中可能的情况都尝试一遍，因此复杂度较高，一种更为高效的方法是随机搜索，每个参数是从可能的参数值的分布中进行采样的。

**超参优化示例**

```python
import numpy as np
from sklearn.svm import SVR     # 引入SVR类
from sklearn.pipeline import make_pipeline   # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score # 引入K折交叉验证
from sklearn import datasets


boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
pipe_SVR = make_pipeline(StandardScaler(), SVR())
score1 = cross_val_score(estimator=pipe_SVR,
                         X = X,
                         y = y,
                         scoring = 'r2',
                         cv = 10)       # 10折交叉验证
print("CV accuracy: %.3f +/- %.3f" % ((np.mean(score1)),np.std(score1)))
```

> CV accuracy: 0.187 +/- 0.649

```python
from sklearn.pipeline import Pipeline
pipe_svr = Pipeline([("StandardScaler",StandardScaler()), ("svr",SVR())])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]}, 
              {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
gs = GridSearchCV(estimator=pipe_svr,
                  param_grid = param_grid,
                  scoring = 'r2',
                  cv = 10)       # 10折交叉验证
gs = gs.fit(X,y)
print("网格搜索最优得分：",gs.best_score_)
print("网格搜索最优参数组合：\n",gs.best_params_)
```

> 网格搜索最优得分： 0.6081303070817127
> 网格搜索最优参数组合：
> {'svr_C': 1000.0, 'svr_gamma': 0.001, 'svr_kernel': 'rbf'}

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform  # 引入均匀分布设置参数
pipe_svr = Pipeline([("StandardScaler",StandardScaler()), ("svr",SVR())])
distributions = dict(svr__C=uniform(loc=1.0, scale=4),  # 构建连续参数的分布
                     svr__kernel=["linear","rbf"],  # 离散参数的集合
                     svr__gamma=uniform(loc=0, scale=4))

rs = RandomizedSearchCV(estimator=pipe_svr,
                        param_distributions = distributions,
                        scoring = 'r2',
                        cv = 10)       # 10折交叉验证
rs = rs.fit(X,y)
print("随机搜索最优得分：",rs.best_score_)
print("随机搜索最优参数组合：\n",rs.best_params_)
```

> 随机搜索最优得分： 0.3035885508009957
> 随机搜索最优参数组合：
>  {'svr_C': 1.0726462730530213, 'svr_gamma': 1.7738373995390608, 'svr_kernel': 'linear'}

