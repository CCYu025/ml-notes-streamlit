import streamlit as st

st.markdown("""
# 第7章：結合不同模型來做整體學習

# 從整體中學習

這句話簡潔地概括了整體學習方法的理念。它強調的是，單個模型的局限性可以透過多個模型協同工作來克服，就像「三個臭皮匠勝過一個諸葛亮」。

### **整體方法 (Ensemble Methods)**

整體方法是機器學習中提升模型**穩定性、準確性**和**泛化能力**的強大技術。其基本思想是訓練多個基礎模型（通常稱為**基分類器**或**弱學習器**），然後以某種方式（如多數決、加權平均等）結合它們的預測結果，以產生一個更強大、更魯棒的最終預測。

### **多數決 (Majority Voting)**

多數決是整體方法中最直觀且常用的結合策略之一。它適用於分類問題：當多個基分類器對同一個樣本進行預測時，最終的預測結果由**獲得最多基分類器支持的類別**決定。這就像民主投票，最終結果由大多數人的意見決定。其背後的假設是，如果每個基分類器犯錯的機率是獨立的，那麼多個獨立基分類器犯同樣錯誤的機率會隨著數量的增加而迅速降低。

```python

# 實作 "機率密度函數"
# 單一錯誤率設為 0.25  整體方法錯誤率0.034
from math import comb
import math

# 定義計算整體錯誤率的函數
# n_classifier: 整體中基分類器的數量
# error: 單一基分類器的錯誤率（假設每個基分類器的錯誤率相同且獨立）
def ensemble_error(n_classifier, error):
    # k_start: 計算整體犯錯所需的最小基分類器錯誤數量
    # 例如，如果 n_classifier=11，那麼超過一半（即 6 個或更多）的基分類器犯錯，整體就犯錯
    k_start = int(math.ceil(n_classifier / 2.))
    
    # 計算每個 k 值（即 k 個基分類器犯錯）的機率，並將這些機率加總
    # 這裡使用了二項分佈的機率公式：C(n, k) * p^k * (1-p)^(n-k)
    # 其中 C(n, k) 是組合數 (n_classifier 選 k)，p 是單一錯誤率 (error)
    probs = [comb(n_classifier, k) *
             error**k *
             (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

# 範例計算：當有 11 個基分類器，且每個基分類器的錯誤率為 0.25 時，整體的錯誤率
ensemble_error(n_classifier=11, error=0.25)  
# 輸出結果 0.034，表示整體方法能將錯誤率從 0.25 顯著降低到 0.034
```
""")

st.image("pages/07/01.png", use_container_width=True)

st.markdown("""
```python

# 繪製 "整體方法的錯誤率" 與 "基本錯誤率" 之間的關係
import numpy as np
import matplotlib.pyplot as plt

# 定義單一基分類器錯誤率的範圍，從 0.0 到 1.0，步長為 0.01
error_range = np.arange(0.0, 1.01, 0.01)
# 計算在每個錯誤率點上，11 個基分類器的整體錯誤率
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]
plt.plot(error_range, ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range, error_range,
         linestyle='--', label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()

```
""")

st.image("pages/07/02.png", use_container_width=True)

st.markdown("""
# 以多數決結合分類器

## 實作一個簡單的多數決分類器

```python

# 使用 NumPy的argmax和bincount函數 呈現 加權多數決 的觀念
import numpy as np

# 範例：三個分類器對同一個樣本的預測結果，假設為 [0, 0, 1]
# 同時給予這些預測結果的權重，[0.2, 0.2, 0.6]
# 這表示第一個分類器預測類別 0，權重 0.2
# 第二個分類器預測類別 0，權重 0.2
# 第三個分類器預測類別 1，權重 0.6
# 
# np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]) 會計算每個類別的加權總和：
# 類別 0 的加權總和 = 0.2 + 0.2 = 0.4
# 類別 1 的加權總和 = 0.6
# 
# np.argmax 會返回加權總和最大的類別索引
np.argmax(np.bincount([0, 0, 1],
            weights = [0.2, 0.2, 0.6])) # 輸出為 1 (因為類別 1 的加權總和 0.6 > 類別 0 的 0.4)

```
""")

st.image("pages/07/03.png", use_container_width=True)

st.markdown("""
```python

# 類別機率：三個分類器對兩個類別的預測機率
# 第一個分類器：類別 0 機率 0.9，類別 1 機率 0.1
# 第二個分類器：類別 0 機率 0.8，類別 1 機率 0.2
# 第三個分類器：類別 0 機率 0.4，類別 1 機率 0.6
ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])   # 類別機率

# 使用 np.average 計算加權平均機率
# axis=0 表示對每一列（即每個類別）進行加權平均
# weights=[0.2, 0.2, 0.6] 則是對每個分類器的預測給予的權重
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])

p

```
""")

st.image("pages/07/04.png", use_container_width=True)

st.markdown("""
```python
# 找出加權平均機率最高的類別
np.argmax(p)

```
""")

st.image("pages/07/05.png", use_container_width=True)

st.markdown("""
```python

# 實作 MajorityVoteClassifier 類別
from sklearn.base import BaseEstimator # 用於提供 estimator 的基本功能 (如 get_params, set_params)
from sklearn.base import ClassifierMixin # 用於標記這是一個分類器，提供預設的 score 方法
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone  # 用於複製基礎分類器，確保訓練獨立性
from sklearn.pipeline import _name_estimators # 用於內部命名分類器，方便在 get_params 中操作
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    """''' A majority vote ensemble classifier
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        Different classifiers for the ensemble
        
    Vote : str, {'classlabel', 'probability'}
        Default : 'classlabel'
        If 'classlabel' the prediction is based on
        the argmax of class labels. Else if
        'probability', the argmax of the sum of
        probabilities is used to predict the class label
        (recommended for calibrated classifiers).
        
    Weights : array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of 'int' or 'float' values are
        provided, the classifiers are weighted by
        importance; Uses uniform weights if 'weights=None'.'''
    """
    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for 
                                  key, value in 
                                  _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self, X, y):
        """ '''Fit classifiers.
        
        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_examples, n_features]
            Matrix of training examples.
            
        y : array-like, shape = [n_examples]
            Vector of target class labels.
        
        Returns
        ----------
        self : object'''
        
        """
        # 驗證 'vote' 參數的合法性
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability'"
                             "or 'classlabel'; got (vote=%r)"
                             % self.vote)
        
        # 驗證 'weights' 參數的合法性，如果提供則其長度需與分類器數量一致
        if self.weights \
            and len(self.weight) != len(self.classifiers):
            raise ValueError("Number of classifiers and weights"
                             "must be equal; got %d weights,"
                             "%d classifiers"
                             % (len(self.weights),
                                len(self.classifiers)))
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,
                                        self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    

    def predict(self, X):
        """ '''Predict class labels for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix},
            Shape = [n_examples, n_features]
            Matrix of training examples.
            
        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.'''
        
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),
                                 axis=1)
        
        else:  # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in
                                      self.classifiers_]).T
            
            maj_vote = np.apply_along_axis(
                            lambda x:
                                np.argmax(np.bincount(x,
                                            weights=self.weights)),
                                axis=1,
                                arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        """''' Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_examples, n_features]
            Training vectors, where n_examples is
            the number of examples and
            n_features is the number of features.
        
        Returns
        ----------
        avg_proba : array-like,
            shape = [n_examples, n_classes]
            Weighted average probability for
            each class per example.'''
        
        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas,
                               axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        """ '''Get classifier parameter names for GridSearch'''"""
        if not deep:
		        # 如果 deep=False，則只返回 MajorityVoteClassifier 本身的參數
            return super(MajorityVoteClassifier,
                         self).get_params(deep=False)
        else:
		        # 否則，返回所有基礎分類器的參數
            out = self.named_classifiers.copy()
            # 遍歷每個命名分類器，將其參數加入到 'out' 字典中
            for name, step in \
                    self.named_classifiers.items():
                for key, value in step.get_params(
                        deep=True).items():
                    # 參數命名格式為 '分類器名稱__參數名'，例如 'logisticregression__C'
                    out['%s__%s' % (name, key)] = value
            return out
    
```

## 以多數決原理來做預測

```python

# 載入鳶尾花數據集
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
# 選取鳶尾花數據集中第 50 行及之後的數據，特徵為第 1 和第 2 列（花萼寬度和花瓣長度）
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

```

```python

# 分割 50% 數據集
X_train, X_test, y_train, y_test =\
    train_test_split(X, y,
                     test_size=0.5,
                     random_state=1,
                     stratify=y)
    
```

```python

# 使用訓練數據集訓練3個不同分類器
# 邏輯斯迴歸、決策樹、k最近鄰
# 使用 10折交叉驗證法
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# 定義邏輯斯迴歸分類器
# penalty='l2' 使用 L2 正則化
# C=0.001 是正則化強度的倒數，值越小正則化越強
# solver='lbfgs' 是優化演算法
clf1 = LogisticRegression(penalty='l2',
                          C=0.001, solver='lbfgs',
                          random_state=1)
# 定義決策樹分類器
# max_depth=1 設置最大深度為 1（一個決策樁，Decision Stump），這是一個弱分類器
# criterion='entropy' 使用資訊增益作為分裂標準
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
# 定義 k最近鄰分類器
# n_neighbors=1 設置 k=1 (即只考慮最近的一個鄰居)
# p=2 表示使用歐幾里得距離 (L2 範數)
# metric='minkowski' 表示使用 Minkowski 距離，當 p=2 時即為歐幾里得距離
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

# 決策樹 (clf2) 通常不需要標準化，所以直接使用
# 由於邏輯斯迴歸和 KNN 對特徵縮放敏感，將它們放入 Pipeline 中，先標準化再分類
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    # 打印平均 ROC AUC 分數和標準差
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))
  
```
""")

st.image("pages/07/06.png", use_container_width=True)

st.markdown("""
```python

# 使用 MajorityVoteClassifier 類別 結合個別分類器完成多數決整體方法
mv_clf = MajorityVoteClassifier(
                classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

```
""")

st.image("pages/07/07.png", use_container_width=True)

st.markdown("""
## 評估並微調整體分類器

```python

# 對 MajorityVoteClassifier 使用測試數據集繪製 ROC曲線檢視效能是否一樣良好
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
        in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,
                     y_train).predict_proba(X_test)[:, 1]
    # 計算 FPR (False Positive Rate), TPR (True Positive Rate) 和閾值
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    # 計算 ROC 曲線下的面積 (AUC)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()

```
""")

st.image("pages/07/08.png", use_container_width=True)

st.markdown("""
```python

# 對 整體分類器 繪製決策區域圖 及比較其他分類器
# 為視覺目的相同尺規標準化 訓練數據集
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

from itertools import product

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                  X_train_std[y_train==0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                  X_train_std[y_train==1, 1],
                                  c='green',
                                  marker='o',
                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -5.,
         s='Sepal Width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5,
         s='Petal length [standardized]',
         ha='center', va='center',
         fontsize=12, rotation=90)
plt.show()

```
""")

st.image("pages/07/09.png", use_container_width=True)

st.markdown("""
```python

# get_params方法回傳參數值
mv_clf.get_params()

```

```python

# 網格搜尋 調整邏輯斯迴歸分類器的 反正規化參數C 及 決策樹的深度並列印分數
from sklearn.model_selection import GridSearchCV

params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_['mean_test_score'][r],
             grid.cv_results_['std_test_score'][r] / 2.0,
             grid.cv_results_['params'][r]))

print('Best parameters: %s' % grid.best_params_)

print('Accuracy: %.2f' % grid.best_score_)

```
""")

st.image("pages/07/10.png", use_container_width=True)

st.markdown("""
# 裝袋法 - 以自助樣本建立整體分類器

也被稱為 自助聚集 (bootstrap aggregating)

**裝袋法** (Bagging)，是 "Bootstrap Aggregating" 的縮寫，是一種強大的**整體學習 (Ensemble Learning)** 技術。它的核心思想是透過**重複有放回地從原始訓練集中抽取樣本 (自助取樣)**，建立多個不同的訓練子集。然後，在每個訓練子集上**獨立訓練一個基分類器 (base estimator)**。最終，這些獨立訓練好的基分類器會透過**多數決 (對於分類問題) 或平均 (對於迴歸問題)** 的方式，結合它們的預測結果，得出最終的整體預測。

## 簡單說明袋裝法

透過裝袋法獲得的隨機樣本，完成個別分類器的適合，就會進行投票，以”多數決”來做預測

每個子樣本集用於訓練一個獨立的基分類器。由於這些子樣本集是隨機且有放回地抽取的，每個子集都可能略有不同，這導致訓練出的基分類器也會有所差異。這些略有差異但表現不俗的基分類器，最終會透過「多數決」（對於分類任務）來匯聚預測結果。

## 運用裝袋法來對葡萄酒數據做分類

```python

# 載入葡萄酒數據集 
# 並選擇兩個特徵 酒精(Alcohol) 
# 和 稀釋葡萄酒的OD280/OD315 (OD280/OD315 of diluted wines)
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash',
                   'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
# 從原始數據集中移除 'Class label' 為 1 的類別
# 這樣做將數據集轉換為一個二元分類問題 (Class label 2 和 3)
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol',
             'OD280/OD315 of diluted wines']].values

```

```python

# 類別標籤 編碼為二元格式
# 分割80%訓練數據集 和 20%測試數據集

from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.2,
                     random_state=1,
                     stratify=y)

```

```python

# 實作 裝袋分類演算法
# 載入 ensemble模組 使用 BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# 使用未修剪的決策樹作為基本分類器
# criterion='entropy' 使用資訊增益作為分裂標準
# random_state=1 確保結果可重現
# max_depth=None 表示決策樹會完全生長，直到葉節點純淨或所有特徵用完，這容易導致過度擬合
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=None)
# 以500棵決策樹結合 整體方法
bag = BaggingClassifier(estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,      # 從 X_train 中抽取的樣本比例 (1.0 = 100%)
                        max_features=1.0,     # 從特徵中抽取的特徵比例 (1.0 = 100%)
                        bootstrap=True,       # 啟用自助取樣（有放回抽樣）
                        bootstrap_features=False,  # 不對特徵進行自助取樣
                        n_jobs=1,             # 使用 1 個 CPU 核心進行並行處理 (若為 -1 則使用所有核心)
                        random_state=1)

```

```python

# 未修剪決策樹做適合及預測
# 預測正確率過低，此模型高異變 (過度適合)
from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_true=y_train, y_pred=y_train_pred)
tree_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

```
""")

st.image("pages/07/11.png", use_container_width=True)

st.markdown("""
```python

# 裝袋分類器做適合及預測
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))

```
""")

st.image("pages/07/12.png", use_container_width=True)

st.markdown("""
```python

# 比較 決策樹 和 裝袋分類器的 決策區域圖
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',\
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, bag],
                        ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(0, -0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

```
""")

st.image("pages/07/13.png", use_container_width=True)

st.markdown("""
# 利用適應強化來提升弱學習效能

AdaBoost法 (Adaptive Boosting)

## 強化法的工作方式

AdaBoost 的工作方式可以概括為以下步驟：

1. **初始化權重：** 每個訓練樣本一開始都被賦予相同的權重。
2. **迭代訓練弱學習器：**
    - 在每一次迭代中，一個新的**弱學習器**（例如：單層決策樹）會被訓練。
    - 在訓練過程中，**模型會更關注那些被前一個弱學習器錯誤分類的樣本**。這是通過**增加錯誤分類樣本的權重**，並**降低正確分類樣本的權重**來實現的。
    - 訓練完成後，根據該弱學習器**預測的準確度**，會給予它一個**自身的權重**（準確度越高的學習器，權重越大）。
3. **加權投票組合：** 最終的預測結果是由所有訓練好的弱學習器進行**加權投票**（分類問題）或**加權平均**（迴歸問題）而得出的，其中每個弱學習器的貢獻由其自身的權重決定。

## 套用 scikit-learn 中的 AdaBoost

```python

# 設定500棵 單層決策樹 訓練AdaBoostClassifier
# 單層決策樹做適合及預測
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy',  # criterion='entropy' 使用資訊增益作為分裂標準
                              random_state=1,
                              max_depth=1)
ada = AdaBoostClassifier(estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,  # learning_rate=0.1 控制每個弱學習器的貢獻大小，值越小疊代次數可能需要越多
                         random_state=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

```
""")

st.image("pages/07/14.png", use_container_width=True)

st.markdown("""
```python

# AdaBoost分類器做適合及預測
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred =ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))

```
""")

st.image("pages/07/15.png", use_container_width=True)

st.markdown("""
```python

# 比較 單層決策樹 和 AdaBoost分類器的 決策區域圖
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue',
                       marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='red',
                       marker='o')
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(0, -0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

```
""")

st.image("pages/07/16.png", use_container_width=True)