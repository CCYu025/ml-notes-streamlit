# 余振中 (Yu Chen Chung)
import streamlit as st

st.set_page_config(
    page_title="我的機器學習網站",
    page_icon="🤖",
    layout="wide"
)  # 網頁設置 開頭 圖示 頁面寬窄



# st.title("")  # 開頭

st.write("# 歡迎來到我的機器學習網站！ 👋")

##  st.sidebar.success("從上方選擇一個頁面。")  # 側邊欄的歡迎訊息

st.markdown(
    """
    這個網站將帶你探索機器學習的基礎概念和主要類型。

    請從左側的導覽列選擇您感興趣的主題。
    """
)

# 定義各分頁，第一個參數是檔案路徑，title 自訂在側邊顯示的標籤
pages = [
    st.Page("pages/01.py",
            title="第一章：賦予電腦從數據中學習的能力",
            icon="📖"),
    st.Page("pages/02.py",
            title="第二章：訓練簡單的機器學習分類演算法",
            icon="🛠️"),
    st.Page("pages/03.py",
            title="第三章：使用 scikit-learn 巡覽機器學習分類器",
            icon="🛠️"),
    st.Page("pages/04.py",
            title="第四章：建置良好的訓練數據集 - 數據預處理",
            icon="🛠️"),

    # ... 其他分頁
]

# 建立側邊選單並跑起來
pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()