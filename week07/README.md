# 第 7 週｜迴歸分析：線性迴歸與羅吉斯迴歸

## 週次資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch4 簡單線性迴歸、Ch5 多元迴歸、Ch6 羅吉斯迴歸 |
| 日期 | 4/9（三）09:10–12:00 |
| Colab 程式 | [Ch4](https://colab.research.google.com/drive/18nr_4pOkjzvs-5uJoPWfKW5cU97KH0qP)、[Ch5](https://colab.research.google.com/drive/1iXifxI-AoJlkVMravR5rXP_2pDn8yYtS)、[Ch6](https://colab.research.google.com/drive/1gaq4UpvRb4CXUrgt_k6lJhMUAsakd1Ly) |

---

# 教學內容

## 教學主題 1：簡單線性迴歸（Ch4，約 50 分鐘）

### 核心觀念

- 什麼是迴歸？用一條直線描述 X 與 Y 的關係
- 迴歸公式：$\hat{y} = \beta x + \alpha$（斜率 × 特徵 + 截距）
- 最小平方法（Least Squares）：讓所有預測誤差的平方和最小
- 評估指標：R²（決定係數）、RMSE（均方根誤差）

### 課堂範例：廣告花費 vs 銷售額

用一組簡單的廣告資料，示範從資料到模型的完整流程。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === 建立範例資料：廣告花費 vs 銷售額 ===
np.random.seed(42)
ad_spend = np.random.uniform(10, 100, 50)  # 廣告花費（萬元）
sales = 2.5 * ad_spend + np.random.normal(0, 15, 50) + 30  # 銷售額（萬元）
df = pd.DataFrame({'ad_spend': ad_spend, 'sales': sales})

# === Step 1：資料切割 ===
X = df[['ad_spend']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Step 2：訓練模型 ===
model = LinearRegression()
model.fit(X_train, y_train)

print(f'斜率 (coefficient): {model.coef_[0]:.4f}')
print(f'截距 (intercept):   {model.intercept_:.4f}')
print(f'迴歸公式：sales = {model.coef_[0]:.2f} × ad_spend + {model.intercept_:.2f}')

# === Step 3：預測與評估 ===
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f'\n訓練 R²: {r2_score(y_train, y_pred_train):.4f}')
print(f'測試 R²: {r2_score(y_test, y_pred_test):.4f}')
print(f'訓練 RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}')
print(f'測試 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}')

# === Step 4：視覺化 ===
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Predicted')
plt.xlabel('Ad Spend (萬元)')
plt.ylabel('Sales (萬元)')
plt.title('簡單線性迴歸：廣告花費 vs 銷售額')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 課堂重點提問

1. 斜率 2.5 代表什麼？→ 廣告每多花 1 萬，銷售額增加約 2.5 萬
2. R² = 0.85 代表什麼？→ 模型解釋了 85% 的銷售變異
3. 訓練 R² > 測試 R² 正常嗎？→ 正常，但差太多就是過擬合

---

## 教學主題 2：多元迴歸（Ch5，約 40 分鐘）

### 核心觀念

- 多元迴歸：多個特徵同時預測 Y → $\hat{y} = \beta_1 x_1 + \beta_2 x_2 + ... + \alpha$
- 為什麼要用多元？單一特徵可能不夠，多個特徵能提供更多資訊
- PolynomialFeatures：當關係不是直線時，加入多項式特徵

### 課堂範例：廣告花費（擴展為三管道）

從主題 1 的單一特徵擴展為三個廣告管道，示範多元迴歸的優勢。

```python
# === 擴展資料：三個廣告管道 ===
np.random.seed(42)
n = 100
df_multi = pd.DataFrame({
    'tv_spend': np.random.uniform(10, 300, n),
    'radio_spend': np.random.uniform(5, 50, n),
    'web_spend': np.random.uniform(1, 80, n),
})
df_multi['sales'] = (
    0.05 * df_multi['tv_spend'] +
    0.1 * df_multi['radio_spend'] +
    0.08 * df_multi['web_spend'] +
    np.random.normal(0, 2, n) + 5
)

X = df_multi[['tv_spend', 'radio_spend', 'web_spend']]
y = df_multi['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 簡單迴歸（只用 tv_spend）===
lr_simple = LinearRegression()
lr_simple.fit(X_train[['tv_spend']], y_train)
r2_simple = r2_score(y_test, lr_simple.predict(X_test[['tv_spend']]))

# === 多元迴歸（三個管道全用）===
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
r2_multi = r2_score(y_test, lr_multi.predict(X_test))

print('=== 簡單迴歸（只用 tv_spend）===')
print(f'測試 R²: {r2_simple:.4f}')

print('\n=== 多元迴歸（三管道）===')
print(f'測試 R²: {r2_multi:.4f}')
print(f'\n各特徵迴歸係數：')
for name, coef in zip(X.columns, lr_multi.coef_):
    print(f'  {name}: {coef:.4f}')
print(f'  截距: {lr_multi.intercept_:.4f}')
```

### 課堂重點提問

1. 多元的 R² 比簡單高多少？→ 多了其他管道的資訊
2. 哪個管道影響最大？→ 看係數大小（注意要標準化後才能比）
3. 加更多特徵一定更好嗎？→ 不一定，可能過擬合或加入雜訊

---

## 教學主題 3：羅吉斯迴歸（Ch6，約 50 分鐘）

### 核心觀念

- 羅吉斯迴歸：用迴歸的方法做二元分類（是/否、故障/正常）
- Sigmoid 函數：把迴歸值壓縮到 0~1，當作機率
- 混淆矩陣：TP、TN、FP、FN 四格的意義
- ROC 曲線與 AUC：衡量分類模型的整體表現

### 課堂範例：鐵達尼號生存預測

用經典的 Titanic 資料，示範從特徵處理到模型評估的完整流程。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)

# === 載入 Titanic 資料 ===
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# 選取特徵並處理
titanic = titanic[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp']].dropna()
X = titanic[['Pclass', 'Age', 'Fare', 'SibSp']]
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 建立 Pipeline：標準化 + 羅吉斯迴歸 ===
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# === 迴歸係數 ===
print('=== 各特徵迴歸係數（標準化後）===')
for name, coef in zip(X.columns, pipe.named_steps['lr'].coef_[0]):
    print(f'  {name}: {coef:.4f}')

# === 分類報告 ===
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred))

# === 混淆矩陣 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
axes[0].set_title('Confusion Matrix')

# === ROC 曲線 ===
y_prob = pipe.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 課堂重點提問

1. Pclass 係數為負代表什麼？→ 艙等越高（數字越小）存活率越高
2. 混淆矩陣的 FN（漏報）在這裡代表什麼？→ 實際生還但預測死亡
3. AUC = 0.80 算好嗎？→ 0.7-0.8 可接受，0.8-0.9 好，>0.9 很好

---

# 作業

> **設計原則**：每題作業都是對應教學主題的**小修改**，換一組資料、換一個情境，但程式碼結構和步驟完全相同。

## 第 1 題：簡單迴歸 + 多元迴歸（40 分）

> 對應教學主題 1 + 2。
> 課堂用「廣告花費 vs 銷售額」，作業換成「加州房價」。
> 程式碼結構一模一樣，只需換資料集和欄位名。

### 任務說明

使用 California Housing 資料集，先用單一特徵做簡單迴歸，再用全部特徵做多元迴歸，比較差異。

### 資料載入

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
```

### 程式要求（對照課堂範例修改）

| 步驟 | 課堂做的 | 作業要做的（只需改的地方） |
|------|---------|-------------------------|
| 資料 | 廣告花費（自建） | California Housing（sklearn 內建） |
| 簡單迴歸特徵 | `ad_spend` | `MedInc`（收入中位數） |
| 多元迴歸特徵 | `tv, radio, web`（3 個） | 全部 8 個特徵 |
| 其餘 | 完全相同 | 完全相同 |

### 作答內容

請建立 `week07/q1_regression.txt`：

```
姓名：
學號：

=== 完整程式碼 ===

=== 簡單迴歸結果 ===
使用特徵：MedInc
斜率：???   截距：???
迴歸公式：Price = ??? × MedInc + ???
訓練 R²：???   測試 R²：???
訓練 RMSE：??? 測試 RMSE：???

=== 多元迴歸結果 ===
各特徵迴歸係數：（列出 8 個）
訓練 R²：???   測試 R²：???
訓練 RMSE：??? 測試 RMSE：???

=== 簡單 vs 多元比較 ===
R² 差了多少？RMSE 差了多少？為什麼？
```

---

## 第 2 題：羅吉斯迴歸 — 船舶設備故障預測（40 分）

> 對應教學主題 3。
> 課堂用「鐵達尼號生存預測」，作業換成「船舶引擎故障預測」。
> Pipeline 結構一模一樣（StandardScaler + LogisticRegression），只需換資料。

### 任務說明

使用船舶引擎感測器資料，預測引擎是否故障。

### 資料建立

```python
np.random.seed(42)
n = 400
data = {
    'engine_hours': np.random.randint(100, 20000, n),
    'vibration_level': np.round(np.random.uniform(0.1, 8.0, n), 2),
    'oil_temp': np.round(np.random.uniform(60, 120, n), 1),
    'rpm': np.random.randint(500, 3000, n),
}
df = pd.DataFrame(data)
prob = 1 / (1 + np.exp(-(
    0.0002 * df['engine_hours'] + 0.3 * df['vibration_level'] +
    0.02 * df['oil_temp'] - 0.001 * df['rpm'] - 2.5
)))
df['is_fault'] = (np.random.random(n) < prob).astype(int)
```

### 程式要求（對照課堂範例修改）

| 步驟 | 課堂做的（Titanic） | 作業要做的（只需改的地方） |
|------|---------------------|-------------------------|
| 資料 | Titanic CSV | 船舶引擎（上方程式碼） |
| 目標欄位 | `Survived` | `is_fault` |
| 特徵 | `Pclass, Age, Fare, SibSp` | `engine_hours, vibration_level, oil_temp, rpm` |
| Pipeline | StandardScaler + LogisticRegression | 完全相同 |
| 評估 | classification_report + 混淆矩陣 + ROC | 完全相同 |

### 作答內容

請建立 `week07/q2_logistic.txt`：

```
姓名：
學號：

=== 完整程式碼 ===

=== 各特徵迴歸係數 ===
engine_hours:    ???
vibration_level: ???
oil_temp:        ???
rpm:             ???

=== 影響最大的特徵 ===
特徵名稱：???
原因：???

=== 分類報告 ===
（貼上 classification_report 輸出）

=== AUC 值 ===
???

=== 混淆矩陣判讀 ===
TP：???  FP：???
FN：???  TN：???
在船舶故障預測中，FN（漏報故障）比 FP（誤報故障）更嚴重，因為：???
```

---

## 第 3 題：迴歸觀念題（20 分）

> 對應教學主題 1-3 的課堂提問延伸。

請建立 `week07/q3_concept.txt`：

```
姓名：
學號：

Q1：課堂上我們用單一特徵（ad_spend）和三個特徵（tv, radio, web）
    分別做了迴歸。多元迴歸的 R² 比簡單迴歸高。
    但加更多特徵一定會讓模型更好嗎？什麼情況下反而會變差？
A1：???

Q2：R²（決定係數）和 RMSE（均方根誤差）都可以評估迴歸模型。
    它們的差異是什麼？在什麼情境下你會更看重 RMSE 而非 R²？
A2：???

Q3：課堂上 Titanic 的混淆矩陣中，FN 代表「實際生還但預測死亡」。
    作業中船舶故障預測的 FN 代表「實際故障但預測正常」。
    在不同的應用情境中（例如：醫療診斷、垃圾郵件過濾、自駕車障礙偵測），
    FN 和 FP 哪個更嚴重？請任選一個情境說明。
A3：???
```

---

## 教學與作業對應總覽

| 教學主題 | 課堂範例 | 作業題目 | 改了什麼 |
|---------|---------|---------|---------|
| 主題 1：簡單線性迴歸 | 廣告花費 → 銷售額 | Q1 前半：MedInc → 房價 | 換資料集 |
| 主題 2：多元迴歸 | 三管道廣告 → 銷售額 | Q1 後半：8 特徵 → 房價 | 換資料集、多特徵 |
| 主題 3：羅吉斯迴歸 | Titanic 生存預測 | Q2：船舶引擎故障預測 | 換海事情境 |
| 課堂提問延伸 | 三個主題的重點問答 | Q3：觀念題（3 題） | 從提問延伸為書面 |

---

## 繳交 Checklist

- [ ] week07/q1_regression.txt 包含完整程式碼、簡單/多元迴歸結果與比較
- [ ] week07/q2_logistic.txt 包含完整程式碼、係數分析、混淆矩陣、ROC 曲線
- [ ] week07/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week07
