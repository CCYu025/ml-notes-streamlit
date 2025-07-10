# 余振中 (Yu Chen Chung)
import streamlit as st

st.markdown("""
# 第5章：透過降維來壓縮數據

本章聚焦三種常見的降維技術：

1. **主成分分析** (PCA)：針對非監督式數據，線性地投影到低維子空間，最大化資料變異。
2. **線性判別分析** (LDA)：針對監督式資料，投影同時最大化類別間離散度、最小化類別內變異。
3. **核主成分分析** (KPCA)：用不同核函數實現非線性降維。

---

# 以主成分分析對非監督式數據壓縮

類似於 “特徵選擇”，使用不同的 “特徵提取”技術 的技術來降低 數據集 中特徵的個數。

## 主成分分析的主要步驟

主成分分析 (principal component analysis，PCA) 是一種 “非監督式線性轉換技術”

常用於 特徵提取 與 降維

PCA目的是針對高維度的數據，最大化 “變異數” 並投影到與原數據集相同或是較低維數的新 “特徵子空間中”。

PCA 之前如果特徵以不同尺度規範測量，一定要做 特徵標準化

### 核心流程

1. **標準化**：各特徵零均值、單位標準差
2. **共變異數矩陣**：衡量特徵間線性相關
3. **Eigendecomposition**：求解特徵值／向量
4. **排序**：依特徵值大小選 top-k 向量
5. **投影**：原始資料乘上前 k 個向量組成的矩陣

> 注意：一定要先對訓練集 fit() 再用同樣的 scaler transform() 測試集，避免資訊洩漏。
> 

---

```python

# 1) 載入葡萄酒資料
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

```

```python

# 2) 切分並標準化
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=0)
# standardize the features
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)  # 正確使用 transform 而非 fit_transform

```
```python

# 3) 計算共變異數矩陣並分解
# 使用 numpy.cov函數可對標準化後的訓練數據集計算 "共變異數矩陣"
# 使用 linalg.eig函數可以完成特徵分解 (eigendecomposition)
import numpy as np

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

```
""")

st.image("pages/05/01.png", use_container_width=True)

st.markdown("""
## 總變異數與解釋變異數

```python

# 4) 計算解釋變異數比例與累積比例
# 使用 NumPy的 comsum函數計算 解釋異變數 (explained variance) 的加總
# 用matplotlib的step函數製圖
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')  # 個別比例
plt.step(range(1,14), cum_var_exp, where='mid',
         label='cumulative explained variance')  # 累積比例
plt.ylabel('Explained variance ratio')  # 變異數解釋率
plt.xlabel('Pricipal component index')  # 主成分索引
plt.legend(loc='best')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/02.png", use_container_width=True)

st.markdown("""
## 特徵轉換

```python

# 5) 特徵對排序與選取前兩主成分
# 依"特徵值"得遞減對"特徵對"進行排序
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalues, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

```
```python

# 為了繪製數據散點圖目的，只選擇兩個特徵向量
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)  # 投影矩陣

```
""")

st.image("pages/05/03.png", use_container_width=True)

st.markdown("""
```python

# 6) 投影到二維空間
# 將訓練數據集藉由計算矩陣內積轉換兩個 "主成分"
X_train_std[0].dot(w)

```
""")

st.image("pages/05/04.png", use_container_width=True)

st.markdown("""
```python

X_train_pca = X_train_std.dot(w)

```

```python

# 以散點圖視覺化三類別在新空間的分布
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/05.png", use_container_width=True)

st.markdown("""
## Scikit-learn 中的主成分分析

```python

# 決策區域圖
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 1. markers 與顏色名稱
    markers = ('s', 'x', 'o', '^', 'v')
    color_names = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    # 2. 先把名稱轉 RGBA tuple
    rgba_colors = [mcolors.to_rgba(c) for c in color_names]
    
    # 3. 建立 colormap（只留所需類別數量）
    n_classes = len(np.unique(y))
    cmap = ListedColormap(rgba_colors[:n_classes])
    
    # 4. 建立網格點
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    
    # 5. 預測並 reshape
    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    # 6. 畫決策面
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # 7. 畫各類別的樣本點
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            alpha=0.8,
            color=cmap(idx),     # 透過 colormap(idx) 取得 RGBA
            marker=markers[idx],
            label=str(cl),
            edgecolor='black'
        )

```

```python

# 訓練數據的決策區域圖
# 減少到只有兩個"主成分"軸
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# 1) 建立 PCA transformer 與分類器
# initailizing the PCA transformer and
# logistic regression estimator:
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1,
                        solver='lbfgs')

# 2) 使用 fit_transform/transform 做降維
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# 3) 在降維後資料上訓練與繪製決策區域
# fitting the logistic regression model on the reduced dataset:
lr = lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/06.png", use_container_width=True)

st.markdown("""
```python

# 測試集上的表現
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/07.png", use_container_width=True)

st.markdown("""
```python

# 4) 查看所有主成分的解釋變異數比例
# n_components=None 以排序好的方式回傳
# 解釋異變數比率可以透過 explained_variance_ratio_屬性讀取
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
```
""")

st.image("pages/05/08.png", use_container_width=True)

st.markdown("""
---

## 小結

- **PCA** 非監督式降維，適合在無標籤時提取最能解釋資料變異的特徵方向。
- 降維後可視化、做為後續分類/回歸模型的前置步驟。
- **關鍵**：先標準化、再 `fit` 於訓練集，測試集僅 `transform`，並透過 `explained_variance_ratio_` 判斷保留成分數。
# 利用線性判別分析做監督式數據壓縮

線性判別分析 (linear discriminant analysis，LDA)

嘗試找出可以最佳化類別分離的 “特徵子空間”

## 比較主成分分析與線性判別分析

LDA 與 PCA都是 線性轉換技術，都可用來減少數據集中的維度

PCA 是一種非監督式演算法；而 LDA 則是監督式演算法

## 線性判別分析的內部工作原理

線性判別分析（LDA）是一種**監督式**降維技術，與 PCA 相似都透過特徵值分解來尋找新的投影空間，

但 LDA 會利用類別標籤資訊，最大化**類別間**散佈與最小化**類別內**散佈，提升分離能力。

## 計算散佈矩陣

```python

# 標準化 的葡萄酒數據的特徵
# 先對特徵標準化（已於前章完成），再計算每類別的平均向量
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],
                             axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))

```
`mean_vecs[i]`：第 `i+1` 類的特徵均值向量，用於後續散佈矩陣計算。
""")

st.image("pages/05/09.png", use_container_width=True)

st.markdown("""
```python

#　使用平均值向量可計算出"類別內"的 "散佈矩陣"
# 逐類別累加每筆樣本與該類均值的外積，得到 SW
d = X_train_std.shape[1] # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

```

**SW** 衡量同一類別樣本的分散程度。
""")

st.image("pages/05/10.png", use_container_width=True)

st.markdown("""
```python

# "訓練數據集" 的 "類別標籤"是不均勻分布的
print('Class label distribution: %s' % np.bincount(y_train)[1:])

```
""")

st.image("pages/05/11.png", use_container_width=True)

st.markdown("""
```python

# 紀算 縮放的類別內"散佈矩陣" (或 共變異數矩陣)
d = X_train_std.shape[1] # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s'
      % (S_W.shape[0], S_W.shape[1]))

```
""")

st.image("pages/05/12.png", use_container_width=True)

st.markdown("""
```python

# 計算 類別間 的 散佈矩陣
# 計算各類別均值與整體均值的差乘以該類樣本數，累加得到 SB
mean_overall = np.mean(X_train_std, axis=0)
d = 13 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # make column vector
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot(
                (mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (
                S_B.shape[0], S_B.shape[1]))

```
**SB** 反映不同類別中心之間的離散程度。
""")

st.image("pages/05/13.png", use_container_width=True)

st.markdown("""
## 為新特徵子空間選擇線性判別式

```python

# 計算 "特徵對" 之後對"特徵值"以遞減方式排序
# 求解 inv(SW)·SB 的特徵值／向量，並依特徵值由大到小排序
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs,
                     key=lambda k: k[0], reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])  # 降序特徵值
 
```

排序後的特徵值代表每個判別方向的重要性。
""")

st.image("pages/05/14.png", use_container_width=True)

st.markdown("""
```python

# 計算並繪製每個線性判別式的「判別力」及其累積比例
# 類別判別資訊 為 判別力 (discriminability)
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/15.png", use_container_width=True)

st.markdown("""
```python

# 取最具判別力的前兩個特徵向量，水平堆疊成 W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)  # 投影矩陣

```
""")

st.image("pages/05/16.png", use_container_width=True)

st.markdown("""
## 投影樣本到新特徵子空間

```python

# 投影訓練集到新子空間
# 繪製投影後的散點圖（類別標籤相同的點共用顏色/符號）
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s',  'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1] * (-1),  # 乘 -1 只是為了視覺上讓第一維向右延伸
                c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/17.png", use_container_width=True)

st.markdown("""
## 以 scikit-learn 完成 LDA

```python

# the following import statement is one line
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

```

```python

# 在降維後的資料上訓練分類器並繪製決策邊界
lr = LogisticRegression(multi_class='ovr', random_state=1,
                        solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/18.png", use_container_width=True)

st.markdown("""
```python

# 測試集投影與可視化
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
```

- 觀察：
    - 訓練集僅誤判一筆第二類
    - 測試集完全正確分類，只用了原始 13 維中的 2 個線性判別成分
""")

st.image("pages/05/19.png", use_container_width=True)

st.markdown("""
---

## 小結

1. **LDA** 在有標籤時更能強化類別可分性，適用監督式降維。
2. 手動計算流程可加深理解，但實務上直接用 `sklearn.discriminant_analysis.LDA` 最為高效。
3. 投影後的低維資料既可視化，也能作為下游分類／回歸的高效特徵。

# 利用核主成分分析處理非線性對應

Kernel PCA

何用核技巧將非線性結構的資料，在降維後依然保持可分離性。

## 核函數與核技巧

徑向基函數核 (Radial Basis Function kernel，RBF)

- **核函數**：隱式地將原始空間映射到更高維，以線性方式分離原本非線性可分的資料。
- **RBF 核 (Gaussian kernel)**：
\[
K(x, x') = \exp\bigl(-\gamma \|x - x'\|^2\bigr)
\]
    - γ 控制高斯“寬度”，γ 越大，核函數越局部化。

---

## 以 Python 實作核主成分分析

```python

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    '''RBF kernel PCA implementation.
    
    Parameters
    -----------
    X: {NumPy ndarray}, shape = [n_examples, n_features]
    
    gamma: float
        Tuning parameter of the RBF kernel
        
    n_components: int
        Number of principal components to return
    
    Returns
    ----------
        X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
            Projected dataset'''
            
    """
    # 1) 距離計算：pdist/squareform 得到 pairwise sqeuclidean 距離矩陣
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')  
    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    
    # 2) 核矩陣：用 NumPy 的 exp 產生 RBF 核
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    
    # 3) 居中：K ← K − 1_N·K − K·1_N + 1_N·K·1_N
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # 4) 特徵分解：eigh 回傳升冪，反轉以取最大特徵值/向量
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    # 5) 取 top-k 特徵向量 (alphas)，組成投影
    # Collect the top k eigenvectors (projected examples)
    X_pc = np.column_stack([eigvecs[:, i]
                            for i in range(n_components)])
    return X_pc

```

## 範例1 - 半月形的分離

- **原始資料**：兩個半月形無法線性分離。
- **標準 PCA**：即使降到 2 維，第一主成分只能捕捉最大變異，但仍無法分開紅／藍兩群；1 維投影更不行。
- **核 PCA**：以 γ=15，降維後在 2 維空間即可線性分離，也可在第一主成分上透過簡單閾值分群。

```python

# 半月形數據集
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/20.png", use_container_width=True)

st.markdown("""
```python

# 使用 標準PCA 轉換後的數據
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/21.png", use_container_width=True)

st.markdown("""
```python

# 使用核PCA函數rbf_kernel_pca 完成線性分離
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/22.png", use_container_width=True)

st.markdown("""
## 範例2 - 同心圓的分離

- **原始資料**：內外兩圈完全重疊，標準 PCA 無法分隔。
- **核 PCA**：同樣以 RBF 核「攤平」同心結構，在降到 2 維後，內圈與外圈可輕易用線性分類器區分。

```python

# 同心圓數據集
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000,
                    random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/23.png", use_container_width=True)

st.markdown("""
```python

# 使用 標準PCA 轉換數據集
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/24.png", use_container_width=True)

st.markdown("""
```python

# 使用 RBF核的PCA轉換數據集
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

```
""")

st.image("pages/05/25.png", use_container_width=True)

st.markdown("""
## 投影新數據點

```python

# 在 rbf_kernel_pca 中算出訓練點的 alphas（投影座標）與對應 lambdas（特徵值）。
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    '''RBF kernel PCA implementation.
    
    Parameters
    ----------
    X: {NumPy ndarray}, shape = [n_examples, n_features]
    gamma: float
        Tuning parameter of the RBF kernel
        
    n_components: int
        Number of principal components to return
    
    Returns
    ----------
        X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
            Projected dataset
        
        lambdas: list
            Eigenvalues'''
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')
    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)
    
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    # Collect the top k eigenvectors (projected examples)
    alphas = np.column_stack([eigvecs[:, i]
                             for i in range(n_components)])
    
    # Coleect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]
    
    return alphas, lambdas

```

```python

# 建立新的半月形數據集，並使用新版 RBF核PCA
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

```

```python

#　確認新程式碼，檢設半月數據第26筆據點
x_new = X[25]
x_new

```
""")

st.image("pages/05/26.png", use_container_width=True)

st.markdown("""
```python

x_proj = alphas[25] # original projection
x_proj

```
""")

st.image("pages/05/27.png", use_container_width=True)

st.markdown("""
```python

def project_x(x_new, X, gamma, alphas, lambdas):
		# 1) 計算新點與所有舊點的 RBF 核向量 k
    pair_dist = np.array([np.sum(
                (x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    # 2) 用 k·(alphas / lambdas) 回歸到主成分座標
    return k.dot(alphas / lambdas)

```

```python

# 使用project_x函數"投影"任何新數據
x_reproj = project_x(x_new, X,
                     gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj

```
""")

st.image("pages/05/28.png", use_container_width=True)

st.markdown("""
```python

# 視覺化正確地將樣本對應到主成分中
plt.scatter(alphas[y==0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]',
            marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]',
            marker='x', s=500)
plt.yticks([], [])
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()

```

驗證 X[25] 的原投影與重投影幾乎重合。
""")

st.image("pages/05/29.png", use_container_width=True)

st.markdown("""
## Scikit-learn 中的核主成分分析

```python

# sklearn.decomposition子模組實作一個 核PCA
from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2,
                       kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

```

```python

# 驗證KernelPCA與實作的結果
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

```

結果與手動實作一致，並支援多種核函數（polynomial、sigmoid…）。
""")

st.image("pages/05/30.png", use_container_width=True)

st.markdown("""
## 小結

1. **Kernel PCA** 能在非線性可分資料上，透過核技巧「攤平」結構後再降維。
2. 參數 **γ**（核寬度）與 **n_components** 影響分群效果，建議用交叉驗證微調。
3. 生態與 sklearn 內建方法可互補：快速原型可用 `KernelPCA`，深入理解可參考手動實作步驟。
""")