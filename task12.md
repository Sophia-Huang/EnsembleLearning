# Blending

## Blending算法原理

Blending集成学习步骤为：                             
   - (1) 将数据划分为训练集和测试集(test_set)，其中训练集需要再次划分为训练集(train_set)和验证集(val_set)；
   - (2) 创建第一层的多个模型，这些模型可以使同质的也可以是异质的；
   - (3) 使用train_set训练步骤2中的多个模型，然后用训练好的模型预测val_set和test_set得到val_predict, test_predict1；
   - (4) 创建第二层的模型,使用val_predict作为训练集训练第二层的模型；
   - (5) 使用第二层训练好的模型对第二层测试集test_predict1进行预测，该结果为整个测试集的结果。   

## Blending的实践

基于iris数据集进行实践。

**导入相关包**

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use("ggplot")
```

**读入数据**

```python
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature = iris.feature_names
data = pd.DataFrame(X, columns=feature)
data['target'] = y

X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.3, random_state=1)
```

**构建模型**

```python
# 第一层
rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='gini')
dt = DecisionTreeClassifier()
svc = SVC(probability=True)
clfs = [dt, rf, svc]

# 第二层
lr = LogisticRegression()

val_features = np.zeros((X_val.shape[0], len(clfs)))  # 初始化验证集结果
test_features = np.zeros((X_test.shape[0], len(clfs)))  # 初始化测试集结果

for i,clf in enumerate(clfs):
    clf.fit(X_train,y_train)
    val_feature = clf.predict_proba(X_val)[:, 1]
    test_feature = clf.predict_proba(X_test)[:,1]
    val_features[:,i] = val_feature
    test_features[:,i] = test_feature

lr.fit(val_features,y_val)
```

**预测结果**

```python
cross_val_score(lr, test_features, y_test, cv=5)
```

> array([0.83333333, 0.83333333, 0.83333333, 0.66666667, 0.66666667])