# 余振中 (Yu Chen Chung)

import streamlit as st
import numpy as np

# st.title("2.訓練簡單的機器學習分類演算法")

st.markdown("""
# 第二章 訓練簡單的機器學習分類演算法

## 類神經元 - 早期機器學習的驚鴻一撇

- **人工神經元數學模型**：
    - 多個輸入信號加權求和，通過激活函數（如階躍函數）輸出。
- **感知器學習規則**：
    
    Δw=η (y−y^) x\Delta w = \eta \,(y - \hat y)\, x
    
    - 根據預測誤差調整權重。

### 限制與擴展

- 只能處理線性可分資料，非線性問題需使用多層結構（Multi-Layer Perceptron）。
- 可結合 **學習率衰減**、**數據標準化** 等技巧，改善收斂與性能。

---

## 以 Python 實作感知器學習演算法

### 1. 定義 Perceptron 類別

```python
import numpy as np

class Perceptron:
    \"""
    感知器分類器（使用階躍函數）

    參數:
    - eta: float, 學習率 (0.0 ~ 1.0)
    - n_iter: int, 資料集迭代次數 (epoch)
    - random_state: int, 權重初始化種子

    屬性:
    - w_: numpy.array, 權重向量 (含偏置)
    - errors_: list, 每輪錯誤更新次數
    \"""
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # 擬合訓練資料: 初始化權重並進行 n_iter 次迭代
        # 初始化隨機數生成器
        rgen = np.random.RandomState(self.random_state)
        # 權重向量大小 = 特徵數 + 1 (偏置)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # 開始每輪迭代
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 計算更新值 = 學習率 * (真實 - 預測)
                update = self.eta * (target - self.predict(xi))
                # 更新非偏置權重
                self.w_[1:] += update * xi
                # 更新偏置
                self.w_[0] += update
                # 如果有更新就算一次錯誤
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # 計算淨輸入 = 特徵加權總和 + 偏置
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # 階躍函數輸出類別 (+1 或 -1)
        return np.where(self.net_input(X) >= 0.0, 1, -1)

```

### 2. 載入與前處理 Iris 數據集

```python
import os
import pandas as pd
import numpy as np

# 讀取 UCI Iris 資料集
data_url = os.path.join(
    'https://archive.ics.uci.edu/',
    'ml',
    'machine-learning-databases',
    'iris',
    'iris.data'
)

df = pd.read_csv(data_url, header=None)
# 選擇前 100 筆 (setosa, versicolor)
y = df.iloc[0:100, 4].values
# 標籤轉換: setosa -> -1, versicolor -> +1
y = np.where(y == 'Iris-setosa', -1, 1)
# 特徵: 花萼長(第0欄)、花瓣長(第2欄)
X = df.iloc[0:100, [0, 2]].values

```

""")

st.image("pages/02/00.png", use_container_width=True)

st.markdown("""
### 3. 散點圖視覺化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
# setosa
plt.scatter(
    X[y == -1, 0], X[y == -1, 1],
    color='red', marker='o', label='setosa'
)
# versicolor
plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color='blue', marker='x', label='versicolor'
)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('Iris 前兩類別散點圖')
plt.grid(True)
plt.show()

```
""")

st.image("pages/02/01.png", use_container_width=True)

st.markdown("""
### 4. 訓練感知器並觀察學習曲線

```python
# 初始化感知器並擬合資料
ppn = Perceptron(eta=0.1, n_iter=10, random_state=42)
ppn.fit(X, y)

# 畫出每次迭代的錯誤數
plt.figure(figsize=(6, 4))
plt.plot(
    range(1, len(ppn.errors_) + 1),
    ppn.errors_,
    marker='o'
)
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('感知器學習曲線')
plt.grid(True)
plt.show()

```
""")

st.image("pages/02/02.png", use_container_width=True)

st.markdown("""
### 5. 決策區域視覺化

```python
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 定義標記與顏色
    markers = ('s', 'x', 'o', '^', 'v')
    color_names = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # 顏色名稱轉 RGBA
    rgba_colors = [mcolors.to_rgba(c) for c in color_names]
    # 建立 colormap
    n_classes = len(np.unique(y))
    cmap = ListedColormap(rgba_colors[:n_classes])

    # 網格範圍
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 預測與 reshape
    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    # 決策面
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 繪製樣本點
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cl, 0], X[y == cl, 1],
            alpha=0.8,
            color=cmap(idx),
            marker=markers[idx],
            label=str(cl),
            edgecolor='black'
        )

# 呼叫並顯示
plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('感知器決策邊界')
plt.grid(True)
plt.show()

```
""")

st.image("pages/02/03.png", use_container_width=True)

st.markdown("""
---
## 補充與延伸

1. **數據標準化**：

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)
ppn.fit(X_std, y)

```

1. **性能評估**：

```python
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = ppn.predict(X_std)
print('Accuracy:', accuracy_score(y, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y, y_pred))

```

1. **延伸實驗**：
- 比較不同學習率 ηη 對收斂速度影響。
- 與 Logistic Regression、SVM 等其他線性分類器比較。
""")

st.markdown("""
## 適應線性神經元（Adaline）與學習的收斂

**Adaline（ADAptive LInear NEuron）** 是早期的神經網路模型之一，與感知器不同的是：

- 感知器使用**類別輸出**（離散值，經過 threshold）
- Adaline 使用**連續值**作為輸出（未經 threshold 的線性輸出），並以此進行**損失函數的最小化**。

---

### 以梯度下降法最小化成本函數

Adaline 的學習目標是最小化以下成本函數（平方和誤差）：

J(w)=12∑(y(i)−ϕ(z(i)))2J(w) = \frac{1}{2} \sum (y^{(i)} - \phi(z^{(i)}))^2

J(w)=21∑(y(i)−ϕ(z(i)))2

使用**批次梯度下降（Batch Gradient Descent）**更新權重，優點是學習具有明確收斂趨勢，缺點是對學習率敏感。

---

### Adaline 實作（Gradient Descent）

```python
python
複製編輯
import numpy as np
import matplotlib.pyplot as plt

class AdalineGD:
    \"""ADAptive LInear NEuron classifier using batch gradient descent.\"""

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                  # 學習率
        self.n_iter = n_iter            # 訓練輪數
        self.random_state = random_state

    def fit(self, X, y):
        \"""訓練模型：透過梯度下降法最小化成本函數。\"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # 權重初始化（含偏置）
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)           # 計算加權總輸入
            output = self.activation(net_input)     # 線性啟用函數
            errors = (y - output)                   # 計算預測誤差
            self.w_[1:] += self.eta * X.T.dot(errors)  # 更新權重
            self.w_[0] += self.eta * errors.sum()      # 更新偏置
            cost = (errors**2).sum() / 2.0          # 成本函數（平方和誤差）
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        \"""計算總輸入（線性組合）：Xw + b\"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        \"""線性啟用函數（Adaline 使用的是 identity）\"""
        return X

    def predict(self, X):
        \"""根據階躍函數，回傳類別標籤（1 or -1）\"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

```

---

### 不同學習速率下的收斂比較

```python
python
複製編輯
# 假設 X, y 是已處理好的資料，如邏輯回歸的輸入格式（標準化後）

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# 學習率 0.01（較大）
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# 學習率 0.0001（較小）
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
           ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()
plt.show()

```
""")

st.image("pages/02/04.png", use_container_width=True)

st.markdown("""
### 特徵縮放 (Feature Scaling)

將特徵標準化（平均值 0、標準差 1），可避免不同量綱影響，顯著加快梯度下降收斂。

```python
# 標準化 X
X_std = X.copy()
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

```

```python
# 使用較大學習率 (eta=0.01) 並訓練
ada = AdalineGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

# 視覺化決策邊界
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.tight_layout()
plt.show()

# 繪製學習曲線 (成本)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()

```

> 說明：經過標準化後，使用較高學習率也能穩定收斂；成本曲線單調下降。
>
""")

st.image("pages/02/05.png", use_container_width=True)
st.image("pages/02/06.png", use_container_width=True)

st.markdown("""
## 隨機梯度下降 (Stochastic Gradient Descent)

AdalineSGD 支援逐樣本更新與打亂 (shuffle)，適合大規模資料與在線學習。

```python
class AdalineSGD:
    \"""Adaptive Linear Neuron (SGD)

    參數:
    - eta: float, 學習率
    - n_iter: int, 迭代次數
    - shuffle: bool, 是否打亂訓練數據
    - random_state: int, 隨機種子

    屬性:
    - w_: 權重向量
    - cost_: 每輪平均平方誤差
    \"""
    def __init__(self, eta=0.01, n_iter=15,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        \"""擬合訓練資料\"""
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            costs = []
            for xi, target in zip(X, y):
                costs.append(self._update_weights(xi, target))
            # 記錄平均成本
            self.cost_.append(sum(costs) / len(y))
        return self

    def partial_fit(self, X, y):
        \"""在不重新初始化權重下擬合資料\"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        \"""打亂訓練資料\"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        \"""初始化權重\"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        \"""根據 Adaline 規則更新權重\"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        return 0.5 * error**2

    def net_input(self, X):
        \"""計算淨輸入\"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        \"""線性激活函數\"""
        return X

    def predict(self, X):
        \"""階躍函數分類\"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
```

```python
# 訓練 AdalineSGD 並繪製決策邊界
ada_sgd = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 繪製平均成本曲線
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()

```

> 說明：SGD 每次只更新單一樣本，成本曲線較 AdalineGD 更具波動性，但能快速處理大資料。
>
""")

st.image("pages/02/07.png", use_container_width=True)
st.image("pages/02/08.png", use_container_width=True)

st.markdown("""
## 小結

- 特徵標準化可顯著提升梯度下降收斂速度與穩定性
- 批次梯度下降 (AdalineGD) 適合小批量或離線訓練
- 隨機梯度下降 (AdalineSGD) 適合大規模、在線學習，需處理成本波動
""")