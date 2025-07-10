# 余振中 (Yu Chen Chung)
import streamlit as st
import numpy as np

st.markdown("""
# 第三章：使用 scikit-learn 巡覽機器學習分類器

## 選擇一個分類演算法

- **天下沒有白吃的午餐定理 (No Free Lunch Theorem)**  
  沒有放諸四海皆準的最佳分類器，必須根據任務特性與資料分布來選擇。

- **考量維度：效能、計算能力、預測能力**  
  - **效能 (Accuracy、F1、AUC…)**：取決於錯誤成本、正負類比例等。  
  - **計算能力 (訓練／推論時間、記憶體需求)**：大資料或高維度特徵時需考慮演算法複雜度。  
  - **預測能力 (泛化性、穩定性)**：在不同測試集或未見數據上的表現。

---

### 選擇機器學習演算法的五個主要步驟

1. **選擇「特徵」並收集訓練數據**  
   - 確定哪些欄位（features）最能區分各類別。  
   - 資料需包含代表性樣本，並注意類別不平衡 (Imbalanced) 問題。

2. **選擇「效能指標」 (performance metric)**  
   - **分類準確度 (Accuracy)**：適用類別平衡時。  
   - **精確率/召回率 (Precision/Recall) 或 F1 分數**：適用正、負類錯誤成本不對等時。  
   - **ROC AUC、PR AUC**：適用二元分類且需要不同閾值評估時。

3. **選擇分類器和最佳化演算法**  
   - **常見分類器**：Logistic Regression、Support Vector Machine (SVM)、Decision Tree、Random Forest、Gradient Boosting (XGBoost、LightGBM) 等。  
   - **最佳化 (Optimizer)**：對於線性模型可選 SGD、LBFGS；對於神經網路則選 Adam、RMSprop 等。  
   - **考量點**：模型可解釋性、訓練時間、參數調整難易度。

4. **評估模型效能**  
   - **交叉驗證 (Cross-Validation)**：K-Fold、Stratified K-Fold 確保分層抽樣。  
   - **學習曲線 (Learning Curve)、驗證曲線 (Validation Curve)**：判斷是否過度擬合 (Overfitting) 或欠擬合 (Underfitting)。  
   - **混淆矩陣 (Confusion Matrix)、ROC 曲線**：更直觀地檢視模型在各類別的表現。

5. **調教演算法 (Hyperparameter Tuning)**  
   - **搜尋策略**：Grid Search、Random Search、Bayesian Optimization。  
   - **重要超參數**：例如樹的深度 (max_depth)、正則化係數 (C) 或學習率 (learning_rate) 等。  
   - **監控過程**：同時觀察驗證集指標，避免只優化訓練集分數。

---

> **小結**：  
> 「資料與特徵」是基礎，「指標與評估」確保方向，「模型與調參」決定最終表現。完整流程需反覆迭代，才能在特定任務中找到最合適的分類方案。  

## 首次使用 scikit-learn 訓練感知器

### 1. 載入鳶尾花資料與分割資料集

```python
# 鳶尾花數據集 簡單又代表性的數據集
# 150 筆樣本的 petal length 與 petal width 設給特徵矩陣 X; 
# 相對應類別標籤定義在向量 y

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]] # 取出第 2、3 欄（petal length、petal width）作為特徵
y = iris.target
print('Class label:', np.unique(y))
# 鳶尾花的類別名轉成為整數
```

### 2. 分割訓練與測試資料

```python

# 隨機分割陣列 X 與 y 的 30% 為"測試數據集"，70% 為"訓練數據集"
# random_state=1 設定一個固定亂數種子
# 此方式製作一個"偽"隨機亂數產生器，是為了每次分割結果都一樣，便於重現
# stratitify=y 回傳與"輸入數據集"具有相同比例類別標籤"訓練子集"和"測試子集"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

```

### 3. 驗證分層抽樣結果

```python

# 使用 Numpy 中的 bincount 計算陣列中每個值得出現次數，以驗證 stratitify=y
# 理想情況下，training 與 test 各類別數量比例應與原始 y 一致
print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))

```

### 4. 特徵標準化

```python

# 最佳化演算法需要做 "特徵縮放" 以便得到最佳效能
# 在此使用 preprocessing 模組裡的 StandardScaler 做特徵標準化
# 只有在用梯度式演算法（例如 Perceptron）時特別需要標準化，如果是樹模型則不一定要求
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)  # 對 "訓練數據集" 每個特徵維度估計參數，即 "樣本均數" 和 "標準差"
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# 藉由 transform 使用估計參數來"標準化" "訓練數據集" 與 "測試數據集"
# "標準化" 後的值才能互相比較

```

### 5. 訓練感知器模型

```python

# 訓練 "感知器模型"
# eta0=0.1 指定學習率 eta0；若不指定會使用預設學習率
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

```

### 6. 預測與計算錯誤分類數

```python

# 預測 並 對照錯誤分類
# (y_test != y_pred).sum() 算出預測錯誤的筆數
y_pred = ppn.predict(X_test_std)
print('Misclassified example: %d' % (y_test != y_pred).sum())

```

### 7. 計算準確度

```python

# metrics 模組有許多種不同 "效能指標"
# 計算 感知器 對 測試數據集 的 "分類正確率"
# accuracy_score(y_true, y_pred) 回傳一個浮點數，介於 0~1 之間
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

```

```python

# 或 每個分類器都有一個 score 方法，
# 會呼叫 predict 與 accuracy_score來計算分類器的預測準確度
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

```

### 8. 繪製決策區域

```python

# 線性決策區域圖
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # resolution=0.02 的意義：網格步長為 0.02，值越小越細緻，但繪圖會比較慢                      
    # step marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'  # 'x' 會忽略，但對 'o'、's' 有用
        )
    
    # highlight test examples
    if test_idx is not None:
        # plot all examples
        # 這裡把測試集點標示為空心圓，以和訓練集做視覺區分
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            facecolors='none',    # 只畫邊框，不填內部
            edgecolors='black',   # 邊框為黑色
            alpha=1.0,
            linewidth=1,
            marker='o',           # 使用實心圓標示
            s=100,
            label='test set'
        )

# 把訓練與測試資料堆疊起來
# 這樣繪決策區域時才能一次把訓練+測試點都畫在同一張圖上
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# 畫圖
plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=ppn,
    test_idx=range(len(y_train), len(y_combined))
)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/01.png", use_container_width=True)

st.markdown("""
## 以邏輯斯迴歸對類別機率塑模

雖然感知器規則簡單易懂，但如果資料並非線性可分，感知器可能無法收斂。相較之下，邏輯斯迴歸（Logistic Regression）是用來建模**二元或多元分類**的模型（與迴歸任務名稱相似，卻非迴歸問題），能透過機率輸出及決策邊界解決這類問題。

---

### 邏輯斯迴歸與條件機率

- **勝算比 (Odds Ratio)**：描述正類與負類的相對機率。
- **Sigmoid 函數**：把任何實數 \(z\) 映射到 \([0,1]\) 之間，公式為
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
可以當作「機率估計函數」。

```python
# 繪製 sigmoid 函數：輸入 z 範圍 -7 到 7，形狀為 S 型
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')      # 在 z=0 畫一條垂直線
plt.ylim(-0.1, 1.1)               # y 軸範圍稍微延伸
plt.xlabel('z')
plt.ylabel(r'$\\varphi(z)$')      # 用 LaTeX 語法顯示函數名稱
# y 軸標籤為 0, 0.5, 1.0，並顯示網格
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
```
""")

st.image("pages/03/02.png", use_container_width=True)

st.markdown("""
### 學習邏輯斯成本函數的加權

- 對於二元分類，若真實類別 y=1y=1y=1，成本函數為 −ln⁡(σ(z))\ln(\sigma(z))−ln(σ(z))；
    
    若 y=0y=0y=0，則成本為 −ln⁡(1−σ(z))-\ln(1 - \sigma(z))−ln(1−σ(z))。
    
- 當 σ(z)\sigma(z)σ(z) 越接近「正確標籤」時，成本越接近 0；如果錯誤預測，成本會趨近無窮大。

```python

# 繪製輸出介於 0~1 之間的 sigmoid 函數，並且畫出對應的成本函數
import matplotlib.pyplot as plt
import numpy as np

def cost_1(z):
    return -np.log(sigmoid(z))       # 如果 y=1 的成本
def cost_0(z):
    return -np.log(1 - sigmoid(z))   # 如果 y=0 的成本

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
# 畫成本函數 J(w) 在 y=1 時的曲線
c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')
# 畫成本函數 J(w) 在 y=0 時的虛線
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
plt.ylim(0.0, 5.1)          # y 軸範圍
plt.xlim([0, 1])           # x 軸範圍
plt.xlabel(r'$\varphi(z)$')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/03.png", use_container_width=True)

st.markdown("""
### 將 Adaline 實作轉換為邏輯斯迴歸演算法

以下程式碼展示如何把 AdalineGD（批次梯度下降）改為 Logistic Regression，更新部分改用 **logistic 代價函數**、並且使用 sigmoid 作為 activation。程式中已維持與 Adaline 相似的介面與流程，只改變代價與輸出部分。

```python

import numpy as np

class LogisticRegressionGD(object):
    \"""Logistic Regression Classifier using gradient descent.

    Parameters
    --------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset (epochs)
    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting (含偏置位)
    cost_ : list
        每輪 epoch 的 logistic cost 值
    \"""
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # 初始化權重：平均為 0，標準差 0.01
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.cost_ = []

        # 批次梯度下降
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)   # sigmoid 輸出
            errors = (y - output)
            # 更新非偏置權重
            self.w_[1:] += self.eta * X.T.dot(errors)
            # 更新偏置
            self.w_[0] += self.eta * errors.sum()

            # 計算 logistic 代價函數
            cost = (-y.dot(np.log(output)) -
                    ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        \"""Calculate net input: w^T x + b\"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        \"""Compute logistic sigmoid activation，使用 clipping 避免 overflow\"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        \"""Return class label after unit step: sigmoid >= 0.5 => 1 else 0\"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # 等價：np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

```

---

```python

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    \"""繪製決策區域 (限二元分類)\"""
    markers = ('s', 'x', 'o', '^', 'v')
    raw_colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    n_classes = len(np.unique(y))
    # 把顏色名稱轉為 RGBA tuples
    rgba_colors = [mcolors.to_rgba(c) for c in raw_colors[:n_classes]]
    cmap = ListedColormap(rgba_colors)

    # 1. 建立網格來判斷各點預測類別
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    # predict 方法傳入一個 (n_samples, 2) 的陣列
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # 2. 用不同顏色填滿各決策區域
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 3. 畫各類別的散佈點
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            color=rgba_colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )

    # 4. 如果有 test_idx，就額外標示測試資料為空心圓
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            facecolors='none',    # 空心圓
            edgecolors='black',   # 邊框為黑
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='test set'
        )

```

```python

# 只適用於二元分類（標籤 0 vs 1）
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(
    eta=0.05,     # 學習率
    n_iter=1000,  # 迭代次數
    random_state=1
)

lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(
    X=X_train_01_subset,
    y=y_train_01_subset,
    classifier=lrgd
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/04.png", use_container_width=True)

st.markdown("""
### 使用 scikit-learn 訓練一個邏輯斯迴歸模型

若要同時支援三個類別，可以直接呼叫 `sklearn.linear_model.LogisticRegression`，它預設使用 OvR（One-vs-Rest）策略，並且有整合 L2 正規化。

```python

from sklearn.linear_model import LogisticRegression

# C：正則化強度的反參數，C 越大正則化效果越弱。
lr = LogisticRegression(
    C=100.0,
    random_state=1,
    solver='lbfgs',         # 使用 BFGS 類優化演算法
    multi_class='ovr'       # One-vs-Rest，多元分類
)
lr.fit(X_train_std, y_train)

plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=lr,
    test_idx=range(105, 150)  # 將測試集高亮
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/05.png", use_container_width=True)

st.markdown("""
```python

# 類別成員機率 (class-membership probabilities)
proba = lr.predict_proba(X_test_std[:3, :])
print("每筆樣本的各類別機率：\n", proba)

# 檢查每列總和是否為 1
print("每列機率總和：", proba.sum(axis=1))

# 取得最大機率對應索引（預測標籤）
print("每筆樣本最高機率對應的類別：", proba.argmax(axis=1))

# 等價於直接呼叫 predict
print("直接呼叫 predict：", lr.predict(X_test_std[:3, :]))

# 單筆樣本預測：reshape 為 (1, -1)
print("單筆樣本預測：", lr.predict(X_test_std[0, :].reshape(1, -1)))

```
""")

st.image("pages/03/06.png", use_container_width=True)

st.markdown("""
### 以正規化處理過度適合現象

- **過度擬合 (Overfitting) / 高變異 (High Variance)**：模型過度複雜，對訓練集擬合太好，泛化能力下降。
- **欠擬合 (Underfitting) / 高偏誤 (High Bias)**：模型過於簡單，無法捕捉資料中的模式。
- 要在「偏誤–變異」之間取得平衡，其中一個方法就是**正規化 (Regularization)**，可抑制模型權重過大，減輕共線性影響並防止雜訊帶來的過度擬合。

```python

# 探索不同 C 值對權重係數的影響 (C 為正則化強度的反參數)
weights, params = [], []
for c in np.arange(-5, 5):  # c 從 -5 到 4
    lr = LogisticRegression(
        C=10.0**c,
        random_state=1,
        solver='lbfgs',
        multi_class='ovr'
    )
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])  # 只觀察對 label=1 的權重
    params.append(10.0**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')  # 將 x 軸設定為對數尺度
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/07.png", use_container_width=True)

st.markdown("""
- 隨著 C 變大，正則化效果減弱，權重會變得較大；
- 若 C 很小，正則化強度高，權重被迫趨近 0，模型較簡單，偏誤較大但變異較小。

---

### 小結

1. 以 Sigmoid 函數塑造機率輸出，並用對數損失函數（Logistic cost）進行梯度下降優化。
2. 自行實作 `LogisticRegressionGD` 來理解梯度更新與成本計算流程。
3. `sklearn.linear_model.LogisticRegression` 已內建多分類與正則化機制，推薦直接使用。
4. 正規化（L2）能有效抑制過度擬合，透過調整 C 值即可在偏誤–變異之間取得平衡。

完整流程：

- **前處理**：標準化特徵；
- **模型訓練**：調整正則化強度；
- **視覺化**：決策邊界、機率輸出；
- **評估**：準確度、學習曲線、權重影響。
""")

st.markdown("""
## 以支援向量機處理最大化分類邊界

支援向量機 (Support Vector Machine, SVM) 透過在特徵空間中尋找一條「最寬的分隔超平面（separating hyperplane）」，來讓不同類別的樣本保持最大距離（margin），即決策邊界到最接近的訓練樣本的距離最大。

- **Margin (邊界)**：決策平面到最近訓練樣本的距離。
- **最大化 Margin 的好處**：在資料線性可分的情況下，擁有較大 Margin 的模型通常能取得較低的泛化誤差 (generalization error)，而較小 Margin 的平面容易過度擬合 (overfitting)。

---

### 使用鬆弛變數處理非線性可分的情況

在現實資料中，許多情況無法嚴格線性可分，這時可允許某些樣本「越界」至錯誤一側，並針對這些「違反約束」之點引入鬆弛變數 (slack variables)，稱為**軟邊界分類 (soft-margin classification)**。調整參數 C 來平衡 margin 與違約懲罰，C越大，代表違約成本越高，模型偏向「減少錯誤」；C 越小，margin 可以更寬，容忍更多違約。

```python
# 訓練 SVM 模型（線性 kernel）並視覺化決策區域
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=svm,
    test_idx=range(105, 150)  # 將測試集高亮
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```

- `kernel='linear'`：使用線性核 (直接在原空間尋找分隔超平面)。
- `C=1.0`：控制軟邊界違約懲罰強度。
""")

st.image("pages/03/08.png", use_container_width=True)

st.markdown("""
### 其他 SVM 的實作方式

也可以透過 `SGDClassifier`，指定不同 loss 參數來模擬其他線性分類器：

```python

from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')  # 感知器
lr  = SGDClassifier(loss='log')         # 邏輯斯迴歸
svm = SGDClassifier(loss='hinge')       # 線性 SVM

```

- `loss='hinge'`：使用 Hinge loss，等同於線性 SVM 的目標，且內部同樣可加上正規化。

---

## 使用核支援向量機解決非線性問題

當資料在原始空間無法線性分離時，可使用「核技巧 (kernel trick)」將資料投影到高維空間，在高維特徵空間中找到分隔超平面。常見的核函數包含：

- **RBF 核 (Radial Basis Function kernel, 高斯核)**：exp⁡(−γ∥x−x′∥^2)
- γ 控制高斯球的「寬度」：γ越大，高斯球越窄，決策邊界會變得更複雜；γ越小，邊界更平滑。

---

### 不可線性分離數據的示範：XOR 資料集

```python

# 先生成一個帶有隨機雜訊的 XOR 二維數據集
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(
    X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
    c='b', marker='x', label='1'
)
plt.scatter(
    X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
    c='r', marker='s', label='-1'
)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

```

XOR 資料在原始特徵空間顯然無法用一條線性分隔，但在高維空間（經過適合的映射）可以線性可分。
""")

st.image("pages/03/09.png", use_container_width=True)

st.markdown("""
### 使用 RBF 核支援向量機分離 XOR 資料集

```python

# RBF 核使得非線性可分的 XOR 數據可被分離
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

plot_decision_regions(
    X=X_xor,
    y=y_xor,
    classifier=svm
)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```

- `kernel='rbf'`：使用徑向基底函數核 (Gaussian kernel)。
- `gamma=0.10`：高斯球寬度，決定「局部」影響範圍。
- `C=10.0`：軟邊界違約懲罰。
""")

st.image("pages/03/10.png", use_container_width=True)

st.markdown("""
### RBF 核在鳶尾花資料集上的效果

```python

# 先將 gamma 設為中等值，觀察決策邊界
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=svm,
    test_idx=range(105, 150)
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```

- 透過 RBF 核，可在二維特徵空間上畫出「非線性」的分隔邊界。
""")

st.image("pages/03/11.png", use_container_width=True)

st.markdown("""
```python

# 將 gamma 設大，觀察過度擬合現象
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=svm,
    test_idx=range(105, 150)
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```

- 當 `gamma` 值非常大時，RBF 核會把每個訓練點都當作「非常窄的高斯球」，使得決策邊界極度貼合訓練資料，容易**過度擬合**，在未見過的資料上泛化誤差反而變高。
""")

st.image("pages/03/12.png", use_container_width=True)

st.markdown("""
### 小結

1. **線性 SVM (kernel='linear')**：在原始特徵空間找出最大 margin 的分隔超平面。
2. **軟邊界 (Soft-Margin)**：使用鬆弛變數允許錯誤樣本落在 margin 區，通過參數 C 控制違約懲罰強度。
3. **核技巧 (Kernel Trick)**：對於非線性可分數據集（如 XOR），可透過 RBF 等核函數將資料映射到高維空間，使資料線性可分。
4. **參數調整**：
    - **C**：越大，違約懲罰越嚴苛，margin 變窄；越小，margin 越寬，但容忍更多錯誤。
    - **γ**（RBF）：值越大，高斯球越窄，更容易過度擬合；值越小，球越寬，邊界更平滑，但可能欠擬合。

完整流程：

- **資料標準化** → **選擇核函數與參數** → **訓練 SVM** → **視覺化決策邊界** → **交叉驗證或網格搜尋調整 C,γ** → **評估模型泛化能力**。
""")

st.markdown("""
## 決策樹學習

決策樹分類器（Decision Tree Classifier）是一種結構化且易於理解的模型，其決策過程可以視覺化並追溯到每個節點所依據的特徵條件，具有很好的可解釋性（interpretability）。

- **分割 (Split)：** 根據某個特徵值在節點處分裂資料
- **資訊增益 (Information Gain)：** 衡量分裂前後「不純度」的減少量，目標是每次分裂都能最大化資訊增益
- **不純度指標 (Impurity Index)：**
    - **Gini 不純度 (Gini impurity)**
    - **熵 (Entropy)**
    - **分類錯誤率 (Misclassification Error)**

---

### 最大化資訊增益 — 取得最大收益

決策樹的目標函數（objective function）是選擇能讓資訊增益最大化的特徵與分裂點。不同的不純度度量方式對樹的結構和效果會有細微影響。

```python
# 視覺化比較三種不純度指標：Gini, Entropy, Misclassification Error
import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    return p * (1 - p) + (1 - p) * p

def entropy(p):
    # 注意 p=0 或 1 時不計算 log2(0)
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p not in (0, 1) else 0 for p in x]
sc_ent = [e * 0.5 if e else 0 for e in ent]  # 將 entropy 縮放一半，比較顆粒度
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip(
    [ent, sc_ent, gini(x), err],
    ['Entropy', 'Entropy (scaled)', 'Gini impurity', 'Misclassification error'],
    ['-', '-', '--', '-.'],
    ['black', 'lightgray', 'red', 'green']
):
    ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')
plt.tight_layout()
plt.show()

```

- **Entropy vs. Gini：**
    - Entropy 較為平滑，對樹的分裂選擇較敏感
    - Gini 計算略微簡便且常用
    - Misclassification Error 用於剪枝時作為簡化指標
""")

st.image("pages/03/13.png", use_container_width=True)

st.markdown("""
## 建構決策樹

1. **不需要對特徵做標準化**
    
    決策樹本身對於原始值的尺度不敏感，可以直接使用原始特徵，例如花萼長度、花瓣寬度（以 cm 為單位）。
    
2. **設定樹的超參數**
    - `criterion='gini'`：使用 Gini 不純度作為分裂依據
    - `max_depth=4`：限制樹的最大深度，避免過度擬合
    - `random_state=1`：固定隨機種子，讓結果可重現

```python

# 訓練決策樹模型並視覺化決策區域
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=1
)

# 直接使用原始 X_train, y_train（切分過的未標準化特徵）
tree_model.fit(X_train, y_train)

# 合併訓練與測試集，用於繪製決策邊界
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(
    X_combined,
    y_combined,
    classifier=tree_model,
    test_idx=range(105, 150)  # 測試集在堆疊後的索引
)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```

- 可以看到不同深度下決策樹如何在二維特徵空間中劃分區域
- 如果不設定 `max_depth`，樹會繼續往下分裂直到所有葉子節點是純淨或達到其他停止條件
""")

st.image("pages/03/14.png", use_container_width=True)

st.markdown("""
```python

from sklearn import tree

# 繪製整棵決策樹結構
tree.plot_tree(tree_model)
plt.show()

```

- `plot_tree` 可以快速把整個樹的節點與分裂條件顯示出來，但較簡單
- 若想要更精美或自訂節點格式，建議用 Graphviz
""")

st.image("pages/03/15.png", use_container_width=True)

st.markdown("""
```python

# conda install -c conda-forge python-graphviz pydotplus
# 若要 export 成 DOT 格式並用 Graphviz 生成 PNG
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(
    tree_model,
    filled=True,
    rounded=True,
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    feature_names=['petal length', 'petal width'],
    out_file=None
)

graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')  # 預設儲存在使用者目錄下（依工作目錄而定）

```

- 這段程式會把決策樹以節點填色、圓角、輸出類別名稱與特徵名稱的方式匯出為 `tree.png`
- 如果需要調整儲存路徑，可直接在 `write_png()` 指定絕對或相對路徑
""")

st.image("pages/03/16.png", use_container_width=True)

st.markdown("""
### 小結

1. **不純度指標 (Impurity Index)**：Entropy、Gini、Error 各有優劣，可根據資料規模與速度需求選擇
2. **決策樹優點**：
    - 易於解釋，可視覺化
    - 不須做特徵標準化
    - 能處理離散與連續特徵
3. **要注意**：
    - 樹過深容易過度擬合，需要透過 `max_depth`、`min_samples_leaf`、`min_samples_split` 等參數剪枝
    - 可搭配交叉驗證或網格搜尋 (GridSearchCV) 找到最佳樹深與分裂條件
4. **Graphviz 匯出**：
    - 先安裝 `pydotplus` 及 Graphviz，再正確配置 PATH，就能匯出更精美的決策樹圖檔
""")

st.markdown("""
## 使用隨機森林來結合多個決策樹

隨機森林（Random Forest）是基於「Bagging （Bootstrap Aggregation）」的整體（ensemble）方法，將多棵決策樹的結果平均或多數投票，以降低單棵樹容易過度擬合的問題，同時保留決策樹的可解釋性與非線性擬合能力。

- `n_estimators=25`：建立 25 棵樹
- `criterion='gini'`：每棵樹內部仍使用 Gini 不純度分裂
- `n_jobs=2`：使用兩個 CPU 核心並行訓練

```python
from sklearn.ensemble import RandomForestClassifier

# 建立並訓練隨機森林
forest = RandomForestClassifier(
    criterion='gini',
    n_estimators=25,
    random_state=1,
    n_jobs=2
)
forest.fit(X_train, y_train)

# 繪製決策區域（合併訓練+測試資料）
plot_decision_regions(
    X_combined, y_combined,
    classifier=forest,
    test_idx=range(105, 150)
)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/17.png", use_container_width=True)

st.markdown("""
## k 最近鄰 — 惰式學習演算法

k 最近鄰（k-Nearest Neighbors, KNN）是**惰性學習器 (lazy learner)**，訓練階段只存資料，不構建顯式模型；預測時才計算樣本與訓練點的距離，依最近的 k 個鄰居多數票決定分類。

- `n_neighbors=5`：考慮離目標點最近的 5 個樣本
- `p=2, metric='minkowski'`：使用歐氏距離；若 `p=1`，則為曼哈頓距離

```python

from sklearn.neighbors import KNeighborsClassifier

# 建立 KNN 並訓練（實際上只記憶訓練資料）
knn = KNeighborsClassifier(
    n_neighbors=5,
    p=2,
    metric='minkowski'
)
knn.fit(X_train_std, y_train)

# 繪製決策區域（標準化後特徵，因為距離計算需要特徵尺度一致）
plot_decision_regions(
    X_combined_std, y_combined,
    classifier=knn,
    test_idx=range(105, 150)
)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/03/18.png", use_container_width=True)

st.markdown("""
## 第三章小結

- **隨機森林**：
    - 結合多棵決策樹，透過隨機抽樣資料與隨機選擇特徵（bagging + feature bagging），有效降低過度擬合。
    - 不須對單棵樹做剪枝，實務上調參較簡單且效果穩定。
- **k 最近鄰 (KNN)**：
    - 無需顯式「訓練」就能立即預測，但每次預測都要計算距離，計算成本高。
    - 是非參數模型（non-parametric），可應對複雜數據分布。

**比選擇演算法更重要的是**：

- **訓練資料質量與特徵選擇** — 若資料中沒有可辨識的訊號或關鍵特徵，再好的模型也無法做出有效預測。
- **正確的資料前處理** — 如標準化、類別平衡、缺失值處理等，都會顯著影響最終模型表現。
""")