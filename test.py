# 余振中 (Yu Chen Chung)
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)  # 網頁設置 開頭 圖示 頁面寬窄

st.title("Sales Streamlit Dashboard")  # 開頭
st.markdown(" Prototype v0.4.1 ")  # markdown格式

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

with st.sidebar:  # 側邊欄位
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info(" Upload a file through config", )
    st.stop()

df = load_data(uploaded_file)

with st.expander("Data Preview"):  # 資料顯示折疊
    st.dataframe(
        df,
        column_config={"Year":st.column_config.NumberColumn(format="%d")
        },  # 將 "Year" 欄位設定為整數格式顯示

    )


# MARKDOWN
st.markdown("""

""")


# Image
st.image("pages/04/01.PNG", use_container_width=True)
