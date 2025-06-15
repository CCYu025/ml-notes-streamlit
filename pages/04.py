# 余振中 (Yu Chen Chung)
import streamlit as st

st.markdown("""
# 第4章：建置良好的訓練數據集 - 數據預處理

## 為何要做好數據預處理

- **數據品質** 直接影響模型表現
- **資訊量** 決定模型能否學到有效資訊

---

# 處理數據遺漏

遺漏值 (missing values)

數據表中的 “空格”、”佔位符號”NaN (Not A Number) 或是 NULL (未知數據)

## 識別數據表格中的遺漏值

### 識別遺漏值

- 可能形式：空格、佔位符 (NaN)、NULL
- 用 `df.isnull().sum()` 快速檢查每個欄位的遺漏數量

```python

# 將CSV數據檔藉由 pandas的 read_csv函式讀入DataFrame
import pandas as pd
from io import StringIO

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
    # If you are using Python 2.7, you need
    # to convert the string to unicode:
    # csv_data = (csv_data)
df = pd.read_csv(StringIO(csv_data))
df

```
""")

st.image("pages/04/01.png", use_container_width=True)

st.markdown("""
```python

# isnull檢查DataFame是否有 "遺漏值"
# 有遺漏值 回傳布林值(真)，反之回傳布林值(偽)
# 套用sum回傳每行幾個 遺漏值
df.isnull().sum()

```
""")

st.image("pages/04/02.png", use_container_width=True)

st.markdown("""
## 刪除具有遺漏值的樣本或特徵

- `df.dropna(axis=0)` 刪除含任何 NaN 的**列**
- `df.dropna(axis=1)` 刪除含任何 NaN 的**欄**
- 進階用法：
    - `how='all'` 只刪除全為 NaN 的列
    - `thresh=k` 保留至少有 k 個非 NaN 值的列
    - `subset=['C']` 只依據特定欄位判斷

```python

# 刪除所有數據中相對應特徵(行)或樣本(列)
df.dropna(axis=0)

```
```python

# 刪除所有至少包含一個NaN的特徵行
df.dropna(axis=1)

```
""")

st.image("pages/04/03.png", use_container_width=True)

st.markdown("""
```python

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN)
df.dropna(how='all')

# drop rows that have less than 4 real values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

```
""")

st.image("pages/04/04.png", use_container_width=True)
st.image("pages/04/05.png", use_container_width=True)
st.image("pages/04/06.png", use_container_width=True)
st.markdown("""
## 填補遺漏值

- **平均值插補**：適合連續數值特徵
- **中位數/眾數**：對抗離群值或分類特徵

```python

# 平均插補 (meanimputation)
# 計算每個特徵"行的平均值"，並將所有NaN換成對應"平均值"
# strategy 其他選項 median(中位數) 或 most_frequent(最頻繁的值或分類特徵做補值)
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

```
""")

st.image("pages/04/07.png", use_container_width=True)

st.markdown("""
```python

# 利用 pandas的 fillna方法
df.fillna(df.mean())

```
""")

st.image("pages/04/08.png", use_container_width=True)

st.markdown("""
**提示**：在 Pipeline 中，`SimpleImputer` 屬於 Transformer，可與其他步驟串接。

## 了解 scikit-learn 的估計器API

SimpleImputer類別在scikit-learn中屬於 transformer類別(轉換器)

估計器(estimator):

fit方法從訓練數據集中做參數學習

transform方法利用參數做 數據轉換

predict方法對新的數據樣本做預測

# 處理分類數據

分類數據 (categorical data)

- 名目特徵 (nominal feature)
- 有序特徵 (ordinal feature)

## 使用 pandas 進行分類數據編碼

```python

# 新建立 DataFrame包含三行特徵:
# 名目特徵(color)，有序特徵(size)，數字特徵(price)
import pandas as pd

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
    ])
df.columns = ['color', 'size', 'price', 'classlabel']
df

```
""")

st.image("pages/04/09.png", use_container_width=True)

st.markdown("""
## 對應有序特徵

```python

# 手動定義size的 對應字典 (mapping dictionary)
size_mapping = {
                'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
df

```
""")

st.image("pages/04/10.png", use_container_width=True)

st.markdown("""
```python

# 整數值轉換回原字串，定義一個 反向對應字典 (reverse-mapping dictionary)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

```
""")

st.image("pages/04/11.png", use_container_width=True)

st.markdown("""
## 類別標籤編碼(Label Encoding)

```python

# 類別標籤 應對字典
import numpy as np

class_mapping = {label:idx for idx, label in 
                 enumerate(np.unique(df['classlabel']))}
class_mapping

```

```python

# 類別標籤 轉換為 整數
df['classlabel'] = df['classlabel'].map(class_mapping)
df

```
""")

st.image("pages/04/12.png", use_container_width=True)

st.markdown("""
```python

# 對應字典中的 鍵-值 對調來完成 反向對應字典
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

```
""")

st.image("pages/04/13.png", use_container_width=True)

st.markdown("""
```python

# scikit-learn中 LabelEncoder類別可直接完成上述工作
# fit_transform方法會先後呼叫fit 與 transform方法
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

```
```python

# inverse_transform方法將整數對應回原始 類別標籤
class_le.inverse_transform(y)

```
## 對名目特徵執行獨熱編碼

### 名目特徵獨熱編碼 (One-Hot Encoding)

- 避免單純用 LabelEncoder 帶來“大小”誤解
- 可用 Scikit-Learn 或 Pandas 兩種方式

```python

# 使用LabelEncoder轉換 color 名目特徵
# 但這結果將不是最佳的
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

```
""")

st.image("pages/04/14.png", use_container_width=True)

st.markdown("""
```python

# 獨熱編碼 (one-hot encoding)
# 對名目特徵中的每個值建立新的 虛擬特徵 (dummy feature)
from sklearn.preprocessing import OneHotEncoder

X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

```
""")

st.image("pages/04/15.png", use_container_width=True)

st.markdown("""
```python

# 使用 ColumnTransformer
# 透過 passthrough參數指定修改行而不修改其他行
# 在 Pipeline 中直接指定要編碼或保留哪些欄
from sklearn.compose import ColumnTransformer

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
        ('onehot', OneHotEncoder(), [0]),
        ('nothing', 'passthrough', [1, 2])
])
c_transf.fit_transform(X).astype(float)

```
```python

# 另一種方法新增 獨熱編碼的虛擬特徵
# 用pandas的 get_dummies方法
pd.get_dummies(df[['price', 'color', 'size']])

```
""")

st.image("pages/04/16.png", use_container_width=True)

st.markdown("""
```python

# 移除特徵 color_blue，藍色資訊事實上仍存在
pd.get_dummies(df[['price', 'color', 'size']],
               drop_first=True)

```
""")

st.image("pages/04/17.png", use_container_width=True)

st.markdown("""
```python

# 透過 OneHotEncoder刪除多餘的行
# 需設置 drop='first' 以及 categories='auto'
color_ohn = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
            ('onehot', color_ohe, [0]),
            ('nothing', 'passthrough', [1, 2])
])
c_transf.fit_transform(X).astype(float)

```
""")

st.image("pages/04/18.png", use_container_width=True)

st.markdown("""
## 小結

- **遺漏值處理**：刪除 vs 填補
- **分類特徵編碼**：有序 → Mapping；名目 → One-Hot
- **Scikit-Learn Pipeline**：`SimpleImputer`、`ColumnTransformer` 幫助保持流程一致且易重複
- **核心原則**：先仔細檢查、再選擇最合適的策略，不同特徵類型需不同處理方法
""")

st.markdown("""
# 將數據集區分為訓練用與測試用

```python

# 葡萄酒數據集 從 UCI機器學習儲存庫 中下載
import numpy as np
import pandas as pd

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()
            
```
## 讀取 葡萄酒 數據集

- 來源：UCI Wine Data
- 無標題 (header=None)，因此手動指定欄位名稱
- `Class label` 為目標 (1,2,3 三類)

""")

st.image("pages/04/19.png", use_container_width=True)


st.markdown("""

```python

# 使用 train_test_split函數隨機劃分數據集為 訓練數據集 與 測試數據集
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

```

- `test_size=0.3`：30% 留作測試
- `stratify=y`：保持各類別比例一致

# 縮放特徵令其具相同比例

特徵縮放 (Feature scaling)

決策樹 和 隨機森林 不需做特徵縮放

正規化:

- MinMaxScaler : 常態化 (normalization) 最小 - 最大縮放，壓縮到 [0,1]
- StandardScaler : 標準化 (standardization) 有限的範，轉為 **零均值、單位標準差**

其他更高階特徵縮放方法，例如 RobustScaler（對離群值更魯棒）

```python

# 用 scikit-learn的 最小-最大縮放 程序(MinMaxScaler) 完成 特徵縮放
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)  # fit on train, transform train
X_test_norm = mms.transform(X_test)   # transform test (注意：用同一個 Scaler)

```

```python

# 手動將數據(值由0到5) 做 標準化 和 常態化
ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean()) / ex.std())

print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

```
""")

st.image("pages/04/20.png", use_container_width=True)

st.markdown("""
```python

# 用 scikit-learn 實作一個 標準化 類別 (StandardScaler)
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```

# 選取有意義的特徵

過度擬合（overfitting）時，模型在訓練資料上表現很好，卻在新資料上錯誤率高。降低**泛化誤差**（generalization error）的常見策略：

- **收集更多訓練資料**
- **正規化**：對過於複雜的模型引入懲罰項
- **減少參數**：選擇簡單模型
- **特徵選擇／降維**：去除多餘特徵

## L1 正規化與 L2 正規化當作模型複雜度的逞罰

- **L2 正規化 (Ridge)**：將所有權重平方和作為懲罰，能抑制單個權重過大，適合均勻收縮。
- **L1 正規化 (Lasso)**：將權重絕對值和作為懲罰，能使部分權重變為零，達到特徵篩選（稀疏解）。

## L2 正規化的幾何解釋

在參數空間上，L2 懲罰相當於在圓形（或超球面）範圍內尋找最小化損失的點；圓形邊界鼓勵權重均勻分布。

## L1 正規化的稀疏解

L1 相當於在菱形（或超稜錐）邊界上尋找最小化損失的點，容易在棱角處產生零值，達成特徵選擇。

```python

# 對 LogisticRegression 套用 L1 正規化
# penalty='l1' 會產生稀疏解（部分 coef_ 為 0）
# solver='liblinear' 支援 L1，multi_class='ovr' 用一對多策略
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')

```

```python

# 將"L1正規化" 的 邏輯斯迴歸應用於"標準化"後 "葡萄酒數據集"，會產生以下稀疏解
# 預測正確率都非常好
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear',
                        multi_class='ovr')
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regularization effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))

print('Test accuracy:', lr.score(X_test_std, y_test))

```
""")

st.image("pages/04/21.png", use_container_width=True)

st.markdown("""
```python

# LogisticReegression於 多類別數據集會使用 一對餘 (one-vs.-rest，OvR)
# 截距項回傳3個值
lr.intercept_

```
""")

st.image("pages/04/22.png", use_container_width=True)

st.markdown("""
```python

# lr.coef_ 屬性讀取 "加權陣列"
# 包含三列 "加權係數";每一列代表某個類別的"加權向量"
# 在 scikit-learn中 w0相對於intercept_，而wj(j>0)則相對於coef_的值
lr.coef_

```

```python

# 繪製不同正規化強度 C（C=1/λ）下的係數路徑
# 它們是不同特徵對不同 "正規化強度"的 "加權係數"路徑
# 當模型以正規化參數(C<0.01)的方式設定所有加權都會為零;C是 正規化參數 λ的倒數
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink','lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1',
                            C=10.**c, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

```
""")

st.image("pages/04/23.png", use_container_width=True)

st.markdown("""
## 循序特徵選擇演算法

```python

# 循序向後選擇 (Sequential Backward Selection，SBS)
# 從所有特徵開始，每次移除一個使驗證分數最小下降的特徵，直到達到目標特徵數。
# f_features參數定義演算法 "最後保留多少個特徵"
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=self.test_size,
                                            random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test,
                    indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

```

```python

# 用 KNN 與 SBS 篩選出最佳子集
# 演算法會將每個階段精確度最好的 "特徵子集合" 所得分數收集、記錄起來
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

```

```python

# 繪製各階段特徵數量 vs. 驗證集準確度
# k={3, 7, 8, 9, 10, 11, 12}時，KNN分類器的準確度是 100%
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

```
""")

st.image("pages/04/24.png", use_container_width=True)

st.markdown("""
```python

# 選出 k=3 時表現最佳的特徵
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

```
""")

st.image("pages/04/25.png", use_container_width=True)

st.markdown("""
```python

# 對 原始"訓練數據集"使用全部特徵正確率97%，對 "測試數據集"正確率97%
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))

print('Test accuracy:', knn.score(X_test_std, y_test))

```
""")

st.image("pages/04/26.png", use_container_width=True)

st.markdown("""
```python

# 只使用"葡萄酒數據集"中不到四分之一的原始特徵，特徵數據集的預測正確率只略微下降
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:',
      knn.score(X_train_std[:, k3], y_train))

print('Test accuracy:',
      knn.score(X_test_std[:, k3], y_test))

```
""")

st.image("pages/04/27.png", use_container_width=True)

st.markdown("""
# 以隨機森林評估特徵的重要性

```python

# 樹模型可自然計算 feature_importances_，評估每個特徵的貢獻度。
# 將13個特徵依重要性來排名
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d % -*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

```
""")

st.image("pages/04/28.png", use_container_width=True)
st.image("pages/04/29.png", use_container_width=True)

st.markdown("""
```python

# 用 SelectFromModel 自動挑出重要性大於門檻的特徵
# 將特徵重要性的"門檻"設為0.1來做特徵選擇
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold',
      'ctiterion:', X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print("%2d % -*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

```
""")

st.image("pages/04/30.png", use_container_width=True)

st.markdown("""
## 小結

1. **正規化**：L1 產生稀疏解、L2 均勻收縮；可搭配交叉驗證選最佳 λ（或 C）。
2. **Wrapper 方法**：SBS 等逐步篩選算法，對少量特徵組合有效，但計算成本高。
3. **Embedded 方法**：利用模型自帶的 feature_importances_ 或 L1 懲罰，一步到位做特徵選擇。
""")