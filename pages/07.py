# 余振中 (Yu Chen Chung)
import streamlit as st

st.markdown("""
# 第6章：學習模型評估和超參數調校的最佳實作

- 得到 模型效能 的 無偏估計
- 診斷 機器學習演算法 的常見問題
- 微調機器學習模型
- 使用不同的 效能指標 來評估預測模型

# 以管線來簡化工作流程

scikit-learn Pipeline類別 ( 管線，pipeline)

- **清晰結構**：將標準化、降維、分類器等步驟串起來，一次呼叫即可完成訓練與測試。
- **避免資訊外洩**：在交叉驗證或網格搜尋時，確保前處理只在訓練摺中擬合，不會用測試集資訊來調整參數。
- **程式簡潔**：一行 `fit`、`predict` 取代多行 `fit_transform + fit + transform + predict`。

## 載入威斯康辛乳癌數據集

```python

# 載入數據集
# 將後面30個特徵指派給陣列X
# 使用 LabelEncoder 將類別標籤轉換編碼
import pandas as pd

df = pd.read_csv('http://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header=None)

from sklearn.preprocessing import LabelEncoder

# X 為第3至32欄特徵；y 為診斷標籤
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

```
""")

st.image("pages/06/01.png", use_container_width=True)

st.markdown("""

```python

# 將 'M' / 'B' 轉成 1 / 0，方便後續模型處理
le.transform(['M', 'B'])

```
""")

st.image("pages/06/02.png", use_container_width=True)

st.markdown("""
```python

# 數據集分拆"訓練數據集"和"測試數據集"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =\
    train_test_split(X, y,
                     test_size=0.20,
                     stratify=y,
                     random_state=1)
 
```

## 結合轉換器與估計器到管線中

```python

# 函數make_pipeline
# 接受任意數目的scikit-learn轉換器(有支援fit和transform方法作為輸入的物件)當作輸入參數
# 後面接著的是一個有實作fit和predict方法的scikit-learn估計器
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),                  # 1. 將每個特徵標準化為均值0、方差1
                        PCA(n_components=2),               # 2. PCA 提取前兩個主成分
                        LogisticRegression(random_state=1,
                                           solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_red = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

```
""")

st.image("pages/06/03.png", use_container_width=True)

st.markdown("""
### 小結

1. **Pipeline** 能自動套用前處理與模型，是實務中維持乾淨且可重現工作流程的最佳方式。
2. `make_pipeline` 自動命名步驟，若要在後續做 `GridSearchCV`，只需以 `stepname__param` 方式指定參數。
3. 在整合交叉驗證時，使用 Pipeline 可以避免測試樣本在標準化或降維時流入訓練，確保評估結果不帶偏。

# 使用 k折交叉驗證法來評估模型效能

## 保留交叉驗證法 (holdout cross-validation)

- 將資料分為訓練集、驗證集與測試集。
- 可以重複使用驗證集以測試不同超參數(hyperparameter)設定。
- 對資料拆分的方式敏感，不同拆分可能導致效能估計差異較大。

## k 折交叉驗證法 **(k-fold Cross-Validation)**

- 將整組資料隨機分為 k 份 (folds)，每次以 k-1 份做訓練，剩下一份做驗證。
- 重複 k 次，共訓練 k 個模型並分別評估。
- 最終報告平均準確率和標準差，以衡量模型穩定性。

```python

# 分層k折交叉驗證法 (stratified k-fold cross-validation)
import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
            np.bincount(y_train[train]), score))

# scores串列中計算模型的 平均正確率 與 估計的標準差
print('\nCV accuracy: %.3f +/- %.3f' %
      (np.mean(scores), np.std(scores)))

```
""")

st.image("pages/06/04.png", use_container_width=True)

st.markdown("""
> 附註：
> 
> - `np.bincount(y_train[train_idx])` 用於計算訓練子集中各類別樣本數，有助於確認資料分佈是否平衡。
> - 標準差越小，表示模型在不同分割上表現越穩定。

```python

# 實作scikit-learn的計分器(scorer)
# n_jobs=1 一個CPU被用於效能評估
# n_jobs=-1 所有可用的CPU
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,      # 要評估的管道或模型
                         X=X_train,
                         y=y_train,
                         cv=10,                  # k 值
                         n_jobs=1)               # CPU 數量，-1 表示使用所有可用核心
print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

```
""")

st.image("pages/06/05.png", use_container_width=True)

st.markdown("""
> 補充說明：
> 
> - `cross_val_score` 內部自動完成資料拆分、模型訓練與評估，適合快速取得交叉驗證結果。
> - `n_jobs=-1` 可充分利用多核心加速計算，但在多模型或大型資料時要留意記憶體消耗。

# 使用學習曲線和驗證曲線來對演算法除錯

學習曲線 (learning curve)

驗證曲線 (validation curve)

## 運用學習曲線診斷偏誤與變異數

- **用途**：檢測模型在不同訓練集大小下的學習行為。
- **偏誤 (Bias)**：若訓練與驗證準確率皆低且接近，表示欠擬合 (High Bias)。
- **變異 (Variance)**：若訓練準確率高，驗證準確率明顯低，表示過擬合 (High Variance)。

```python

# learning_curve函數會使用 分層k折交叉驗證法來計算準確性
# train_sizes=np.linspace(0.1, 1.0, 10) 使用10個均勻區分隔訓練集
# 利用cv參數 來完成 分層10折交叉驗證法
# 使用 fill_between函數圖形中加入平均正確率的標準差
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           solver='lbfgs',
                                           max_iter=10000))

train_sizes, train_scores, test_scores =\
    learning_curve(estimator=pipe_lr,
                   X=X_train,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),  # 使用 10 個不同訓練大小
                   cv=10,                                  # 分層10折交叉驗證
                   n_jobs=1)                               # 使用單核心
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.show()

```
""")

st.image("pages/06/06.png", use_container_width=True)

st.markdown("""
> 解讀：
> 
> - 若兩條曲線同時低且接近，需增加模型複雜度或特徵；
> - 若存在明顯差距，考慮正則化或獲得更多訓練資料。

## 運用驗證曲線討論低度適合與過度適合

- **用途**：探索單一超參數 (如正則化參數 C) 對模型效能的影響。
- **低度適合 (Underfitting)**：低 C 值 (強正則化) 導致偏誤高。
- **過度適合 (Overfitting)**：高 C 值 (弱正則化) 導致變異高。

```python

# validation_curve函數 預設使用 分層k折交叉驗證法 來評估效能
from sklearn.model_selection import validation_curve

# 測試不同 C 值對模型準確率的影響
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                    estimator=pipe_lr,
                    X=X_train,
                    y=y_train,
                    param_name='logisticregression__C',  # 管道中模型的參數名稱
                    param_range=param_range,
                    cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()

```
""")

st.image("pages/06/07.png", use_container_width=True)

st.markdown("""
> 解讀：
> 
> - 找到使驗證準確率最高且差距最小的 C 值；
> - 可視情況調整正則化強度以平衡偏誤與變異。

# 以網格搜尋微調機器學習模型

- **用途**：自動搜尋最佳超參數組合，透過多組設定比較模型效能。

## 以網格搜尋微調超參數

```python

# sklearn.model_selection模組
# 初始化GridSearchCV來訓練、微調管線中的支援向量機 (SVM)
# 設定param_grid參數為一個"字典"型別串列，包含要微調的參數
# 最佳模型表現分數best_score_ ，最佳模型屬性best_params_
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 建立 SVM 管道，含標準化與 SVC
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]

# 初始化 GridSearchCV，cv=10，使用所有核心加速
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
# 最佳交叉驗證準確率
print(gs.best_score_)
# 最佳參數組合
print(gs.best_params_)

```
""")

st.image("pages/06/08.png", use_container_width=True)

st.markdown("""
> 提示：
> 
> - `best_score_` 顯示交叉驗證中最高平均準確率；
> - `best_params_` 為對應的超參數設定。

```python

# 最佳模型 可用 GridSearchCV物件中的 best_estimator_屬性取得
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

```
""")

st.image("pages/06/09.png", use_container_width=True)

st.markdown("""
## 以巢狀交叉驗證選擇演算法

### 巢狀交叉驗證 (Nested Cross-Validation)

- **目的**：在超參數搜尋的同時，外層保留測試集以評估最終效能，避免過度樂觀。

```python

# k折交叉驗證法外層迴圈 外折(outer fold)與內層迴圈 內折(inner fold)
# 五個外折和兩個內折 "5 * 2交叉驗證法"
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

```
""")

st.image("pages/06/10.png", use_container_width=True)

st.markdown("""
```python

# 巢狀交叉驗證作一個簡單的 決策樹分類器，只微調決策樹的參數深度
from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(
                            random_state=0),
                    param_grid=[{'max_depth': [1, 2, 3,
                                        4, 5, 6, 7, None]}],
                    scoring='accuracy',
                    cv=2)
scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

```
""")

st.image("pages/06/11.png", use_container_width=True)

st.markdown("""
> 註解：
> 
> - 巢狀交叉驗證可提供更客觀的效能估計；
> - 外層維持測試集未曝光，內層進行超參數搜尋。

# 其他不同的效能指標

精準度 (precision)

召回率 (recall)

F1 分數 (F1-score)

## 混淆矩陣的讀取

confusion matrix

- **組成**：TF、TN、FP、FN
- **用途**：全面觀察分類器對各類別的預測情況。

```python

# scikit-learn 的 confusion_matrix函數
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

```
""")

st.image("pages/06/12.png", use_container_width=True)

st.markdown("""
```python

# 使用 matplotlib的 matshow函數顯示 混淆矩陣
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,
                s=confmat[i, j],
                va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

```
""")

st.image("pages/06/13.png", use_container_width=True)

st.markdown("""
## 最佳化分類模型的精準度與召回率

- **精準度 (Precision)**：TP / (TP + FP)
- **召回率 (Recall)**：TP / (TP + FN)
- **F1-score**：Precision 與 Recall 的調和平均。

```python

# 實作 scikit-learn.metrics 計分指標
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('Precision: %.3f' % precision_score(
                    y_true=y_test, y_pred=y_pred))

print('Recall: %.3f' % recall_score(
                    y_true=y_test, y_pred=y_pred))

print('F1: %.3f' % f1_score(
                    y_true=y_test, y_pred=y_pred))

```
""")

st.image("pages/06/14.png", use_container_width=True)

st.markdown("""
```python

# 使用 make_scorer函數建立計分器 (此用f1_score)
# 給 GridSearchCV中的scoring參數
from sklearn.metrics import make_scorer

c_gamma_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]
# 自訂 F1-score 計分器，用於 GridSearchCV 的 scoring
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)

print(gs.best_params_)

```
""")

st.image("pages/06/15.png", use_container_width=True)

st.markdown("""
## 製作接收操作特徵圖

接收操作特徵 (Receiver operating characteristic，ROC)

ROC 曲面下面積 (ROC area under the curve，ROC AUC)

```python

# roc_curve函數計算 分類器的效能
# auc函數計算 曲線下面積 (AUC)
from sklearn.metrics import roc_curve, auc
# from scipy import interp # 用NumPy取代

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           solver='lbfgs', C=100.0))
# 只選取兩個特徵作示範
X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3,
                          random_state=1,
                          shuffle=True).split(X_train,
                                                y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                 % (i+1, roc_auc))
# 隨機猜測基線
plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='random guessing')
# 計算平均 AUC
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.show()

```
""")

st.image("pages/06/16.png", use_container_width=True)

st.markdown("""
## 多類別分類的計分指標

- `average='micro'`、`'macro'`、`'weighted'` 等參數決定平均方式。

```python

# 藉由設定 average參數指定用甚麼 平均方法
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')

```

# 處理類別不平衡的狀況

當目標類別分佈嚴重不平衡時，單純準確率可能失真，可透過抽樣或加權方式處理。

```python

# 不平衡數據集
# 取前40個惡性腫瘤建立類別失衡數據集
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

```

```python

# 以預測所有樣本為良性，計算基準準確率
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

```

**解讀：** 若基準準確率很高，但模型無法偵測少數類別，表示需要處理不平衡。
""")

st.image("pages/06/17.png", use_container_width=True)

st.markdown("""
**解讀：** 若基準準確率很高，但模型無法偵測少數類別，表示需要處理不平衡。

```python

# 實作scikit-learn中的resample函數 (放回式抽樣)
from sklearn.utils import resample

# 原始惡性樣本數
print('Number of class 1 examples before:',
      X_imb[y_imb == 1].shape[0])

# 對少數類別做放回式抽樣至與多數類別數量相同
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],         # 少數類別特徵
                                    y_imb[y_imb == 1],         # 少數類別標籤
                                    replace=True,              # 放回抽樣
                                    n_samples=X_imb[y_imb == 0].shape[0],  # 目標樣本數
                                    random_state=123)
print('Number of class 1 examples after:',
      X_upsampled.shape[0])

```
""")

st.image("pages/06/18.png", use_container_width=True)

st.markdown("""
```python

# 合併上採樣後的少數類別與原多數類別，得到平衡資料集
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

```

```python

# 以同樣方法評估基準準確率
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
```
""")

st.image("pages/06/19.png", use_container_width=True)