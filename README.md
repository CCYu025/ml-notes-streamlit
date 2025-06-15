# 🤖 機器學習筆記網站（Streamlit 多分頁）

這是一個以 **Streamlit** 製作的互動式網站，作為我學習《Python 機器學習》各章節的知識整理與展示。透過多分頁設計，清楚劃分每章內容，幫助理解分類模型、資料處理與模型訓練流程。

👉 **線上展示**（可選）：[https://ccyu025-ml-notes.streamlit.app](https://ccyu025-ml-notes.streamlit.app/)

---

## 🧠 網站功能簡介

- 📖 **首頁介紹**：簡單說明網站內容與導覽方式
- 🧭 **側邊欄分頁導航**：每一章為獨立分頁，方便閱讀與維護
- 📊 **圖表展示**：搭配 `matplotlib`、`scikit-learn` 等視覺化學習成果
- 📝 **逐步程式碼展示**：部分章節附有模型訓練過程與程式輸出

---

## 📁 專案結構

ml-notes-streamlit/

├── app.py               # 主首頁與頁面導航邏輯

├── pages/

│   ├── 01.py            # 第1章：讓機器從數據中學習

│   ├── 02.py            # 第2章：訓練簡單的分類器

│   ├── 03.py            # 第3章：使用 scikit-learn

│   └── 04.py            # 第4章：數據預處理

├── requirements.txt     # 所需套件

└── .gitignore

---

## 🔧 安裝與執行

```

# 安裝環境
pip install -r requirements.txt

# 執行網站
streamlit run app.py

```

---

## 📚 使用到的技術

- Python 3.11+
- Streamlit
- Pandas / NumPy
- Matplotlib
- scikit-learn

---

## 🎯 開發目的與應用場景

本專案不僅是學習紀錄，也是為了訓練自己將複雜的機器學習流程**視覺化、模組化與簡化**。適合：

- 自學者建立筆記型應用
- 講師作為教學展示平台
- 求職面試時展示技術與理解力

---

## 👨‍💻 作者資訊

> 製作者：余振中 (Yu Chen Chung)
> 
> 
> GitHub: [github.com/CCYu025](https://github.com/CCYu025)
> 
> Portfolio: (可加入 Notion、Blog 或其他展示網站)
>