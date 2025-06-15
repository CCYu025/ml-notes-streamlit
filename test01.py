# 余振中 (Yu Chen Chung)
import streamlit as st
import io

st.set_page_config(layout="wide")  # 設定頁面佈局為寬版

st.title("文本檔案預覽器 (.txt 和 .md)")
st.write("請上傳一個 .txt 或 .md 檔案來預覽其內容。")

# 允許使用者上傳 .txt 或 .md 檔案
uploaded_file = st.file_uploader("選擇一個文本檔案 (.txt 或 .md)", type=["txt", "md"])

if uploaded_file is not None:
    # 讀取檔案內容
    # uploaded_file.getvalue() 返回的是 bytes
    # .decode("utf-8") 將 bytes 轉換為 utf-8 編碼的字串
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")

        st.subheader("檔案內容預覽：")

        # 根據檔案類型選擇顯示方式
        if uploaded_file.type == "text/plain":  # .txt 檔案的 MIME 類型
            st.text(file_content)  # 使用 st.text 顯示純文本，保留原始格式
        elif uploaded_file.type == "text/markdown":  # .md 檔案的 MIME 類型
            st.markdown(file_content)  # 使用 st.markdown 顯示 Markdown 內容
        else:
            st.write(file_content)  # 預設使用 st.write

    except UnicodeDecodeError:
        st.error("檔案編碼錯誤，請確認檔案是 UTF-8 編碼或嘗試其他編碼。")
    except Exception as e:
        st.error(f"讀取文件時發生錯誤：{e}")