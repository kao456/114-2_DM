# 第 10 週｜分類模板、交叉驗證與網格搜尋 + Prompt Engineering 入門

> 對應教科書：Ch10 分類模板、Ch11 交叉驗證、Ch12 網格搜尋
> 進度：期中考後第一週，從本週起逐步導入 AI 輔助學習工具

---

## 學習目標

1. 看懂並改寫分類預測的標準流程（資料前處理 → 切分 → 模型訓練 → 預測 → 評估）
2. 用 K-Fold 交叉驗證評估模型穩定性，避免單次切分誤差
3. 用 `GridSearchCV` 自動找最佳超參數組合
4. **學會用 Prompt Engineering 五要素寫出有效的 AI 提示**
5. **學會用 AI 輔助開發三大原則保留學習成效**

---

## 一、本週課程主軸（Ch10–Ch12）

### 1. 分類預測模板（Ch10）

把前幾週學的分類器（KNN、SVM、決策樹）整理成共用的預測模板：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Colab：[10 分類預測模版](https://colab.research.google.com/drive/1OqudZ0PDJ3YaUQPiOwilPG9vcCX2O9jt)

### 2. K-Fold 交叉驗證（Ch11）

單次 train/test 切分容易因運氣好壞影響結果，**K-Fold 把資料切 K 份輪流當測試集**，得到 K 個分數，取平均更穩定。

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_s, y, cv=5, scoring='accuracy')
print(f"5-Fold mean = {scores.mean():.3f}, std = {scores.std():.3f}")
```

Colab：[11 交叉驗證](https://colab.research.google.com/drive/1YvHf8e4V5-OFlAClYlfgaE6xBJRvxNvo?usp=sharing)

### 3. GridSearchCV 網格搜尋（Ch12）

超參數（如 KNN 的 `n_neighbors`、SVM 的 `C`/`gamma`）會大幅影響表現，`GridSearchCV` **自動枚舉所有組合 + 交叉驗證 + 給最佳組合**。

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_s, y)
print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
```

Colab：[12 模型參數挑選和網格搜尋](https://colab.research.google.com/drive/1o-I1M7RAbANMsawstOypcshUuDmNBaQ2?usp=sharing)

---

## 二、Prompt Engineering 入門 — 好 Prompt 的五個要素

從本週起，課程逐步引入 AI 工具（ChatGPT、Claude、Copilot）輔助學習。**寫好 prompt 是 AI 時代的核心技能**。一個有效的 prompt 通常包含五個要素：

| # | 要素 | 說明 | 反例 → 正例 |
|---|---|---|---|
| 1 | **角色（Role）** | 設定 AI 的身份/視角 | ❌「教我 KNN」<br>✅「你是資料探勘課程的助教，請用大二學生能懂的方式解釋 KNN」 |
| 2 | **任務（Task）** | 明確要做什麼動作 | ❌「KNN 怎麼用」<br>✅「請**比較** KNN 與 SVM 在這份資料上的優缺點」 |
| 3 | **背景（Context）** | 提供脈絡：資料、已試過什麼 | ❌「我跑不出來」<br>✅「我用 sklearn 的 iris 資料集，KNN n=5 跑出 accuracy 0.6，比預期低，附上我的 code …」 |
| 4 | **格式（Format）** | 指定輸出形式 | ❌「給我答案」<br>✅「請以表格列出三種模型的 accuracy / f1 / 計算時間」 |
| 5 | **限制（Constraints）** | 明確邊界與禁止 | ❌（無）<br>✅「**不要直接給我完整答案**，請用問題引導我自己找出 bug；以 200 字以內回答」 |

### 範例對比

❌ **沒有結構的 prompt**：
> 教我交叉驗證

✅ **有五要素的 prompt**：
> 你是資料探勘課程的家教（**角色**）。請**解釋**為什麼要用 5-Fold 交叉驗證而不是單次 train/test split（**任務**）。我已經學過 train_test_split，但不懂為什麼結果不穩定（**背景**）。請以「先回答 why、再給一個範例 code、最後列出三個常見錯誤」的順序回答（**格式**）。**不要超過 300 字**，**程式碼用 Python sklearn**（**限制**）。

---

## 三、AI 輔助開發三大原則

只會寫 prompt 還不夠 — **怎麼用 AI 才不會讓自己變笨**才是關鍵。研究顯示（Shen & Tamkin 2026），把 AI 當「直接寫答案的工具」會損害學習；當作「幫你思考的對話夥伴」才能保留學習成效。請守住三大原則：

### 原則 1：理解優先（Understand Before Generate）

**不要叫 AI 直接寫，先請 AI 解釋你不懂的概念。**

| 反例 | 正例 |
|---|---|
| 「幫我寫一個 KNN 分類器」 | 「請用簡單範例解釋為什麼 KNN 對 feature scale 敏感，再告訴我 StandardScaler 在做什麼」 |

理由：你不懂的概念若沒搞清楚，貼上 AI 給的 code 也只是抄作業，下次遇到變形題仍不會。

### 原則 2：小步驗證（Small Steps, Verify Each）

**每個 AI 給的 snippet 都要在 Colab 跑一次、看輸出對不對。**

不要把 AI 給的 100 行 code 整段貼上去說「跑不出來」。改用：

1. 拿前 10 行貼上 → 跑 → 看 print 的東西對不對
2. 對的話再貼下 10 行 → 跑 → 驗證
3. 任一段不對 → 把錯誤訊息 + 你貼的 code 拿回去問 AI

理由：100 行 code 出錯時，你不知道哪行有問題；分段驗證才能精準定位。

### 原則 3：質疑與重述（Challenge and Rephrase）

**拿到答案後，先用自己一句話重述，再問「為什麼這樣寫」。**

| 收到 AI 回答後的兩個動作 |
|---|
| **重述**：「所以 GridSearchCV 是把參數所有組合都跑一次，挑分數最高的對嗎？」 |
| **質疑**：「為什麼用 5-fold 而不是 10-fold？資料量小的時候會不會 overfit？」 |

理由：能用自己的話講出來才代表懂了；能質疑代表開始有自己的判斷。

---

## 四、課堂演練（建議步驟）

> 老師會帶大家做一遍，作業則自己跑一次

1. 開啟 Colab [12 模型參數挑選和網格搜尋](https://colab.research.google.com/drive/1o-I1M7RAbANMsawstOypcshUuDmNBaQ2?usp=sharing)
2. 把範例 code 完整跑一次，看 best_params_ 是什麼
3. 用**五要素 prompt** 問 AI：「為什麼 best_params_ 是這個值？換成另一組會怎樣？」
4. 用**原則 2（小步驗證）**改一個參數重跑，看 best_score 變化
5. 用**原則 3（質疑與重述）**寫一句話總結「GridSearchCV 在做什麼」貼到課堂討論區

---

## 五、課後作業

| # | 內容 | 繳交方式 |
|---|---|---|
| 1 | 把 Ch10 / Ch11 / Ch12 三個 Colab 跑完（自己的 Google 帳號 copy 一份）| 連結貼到作業 issue |
| 2 | 寫一個**含五要素的 prompt**，用來請 AI 解釋你期中考最不會的一題；把 prompt 與 AI 回答貼出來 | issue 內貼文 |
| 3 | 用**三大原則**做一次完整對話，把對話截圖（至少 3 輪）貼出來 | issue 內貼文 |

繳交期限：下週上課前

---

## 六、本週重點觀念複習卡

| 觀念 | 一句話記憶 |
|---|---|
| 分類模板 | `train_test_split → fit → predict → evaluate` 五步驟 |
| K-Fold 交叉驗證 | 切 K 份輪流當測試集，取平均分數比較穩 |
| GridSearchCV | 暴力枚舉所有超參數組合 + 交叉驗證 = 最佳組合 |
| Prompt 五要素 | 角色 / 任務 / 背景 / 格式 / 限制 |
| AI 輔助三原則 | 理解優先 → 小步驗證 → 質疑與重述 |

---

*本週內容是模型評估與優化的核心技術；下週起進入組合預測器（Ch13），請先熟悉本週三個 Colab，組合預測器會以此為基礎。*
