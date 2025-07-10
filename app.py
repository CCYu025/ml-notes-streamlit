# 余振中 (Yu Chen Chung)
import streamlit as st

st.set_page_config(
    page_title="我的機器學習網站",
    page_icon="🤖",
    layout="wide"
)  # 網頁設置 開頭 圖示 頁面寬窄



# st.title("")  # 開頭



# 定義各分頁，第一個參數是檔案路徑，title 自訂在側邊顯示的標籤
# pages = [
#     st.Page("pages/01.py",
#             title="第一章",
#             icon="📖"),
#     st.Page("pages/02.py",
#             title="第二章",
#             icon="🛠️"),
#     st.Page("pages/03.py",
#             title="第三章",
#             icon="🛠️"),
#     st.Page("pages/04.py",
#             title="第四章",
#             icon="🛠️"),
#     st.Page("pages/05.py",
#             title="第五章",
#             icon="🛠️"),
#
#     # ... 其他分頁
# ]

with st.sidebar:
    # 這是你會在側邊欄看到的第一個標題
    st.title("導航選單")

    st.write("---")  # 增加一條分隔線讓視覺更清晰
    pages = [
         st.Page("pages/01.py",
                 title="Python 機器學習",
                 icon="📖"),

         st.Page("pages/02.py",
                 title="第一章",
                 icon="🛠️"),
         st.Page("pages/03.py",
                 title="第二章",
                 icon="🛠️"),
         st.Page("pages/04.py",
                 title="第三章",
                 icon="🛠️"),
         st.Page("pages/05.py",
                 title="第四章",
                 icon="🛠️"),
         st.Page("pages/06.py",
                 title="第五章",
                 icon="🛠️"),
        st.Page("pages/07.py",
                title="第六章",
                icon="🛠️"),
        st.Page("pages/08.py",
                title="第七章",
                icon="🛠️"),
        st.Page("pages/09.py",
                title="第八章",
                icon="🛠️"),

         # ... 其他分頁
     ]


    # 建立側邊選單並跑起來
    pg = st.navigation(pages, expanded=True)

pg.run()


