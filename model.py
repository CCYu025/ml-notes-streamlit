# 余振中 (Yu Chen Chung)
import streamlit as st


def mathod01():

    if st.session_state.load_error:
        st.error(f"❌ 資料載入失敗: {st.session_state.load_error}")
    else:
        # 按鈕觸發資料載入並顯示
        if st.button("資料載入"):
            st.success("資料載入成功 ✅")
            st.subheader("📋 資料預覽")
            st.dataframe(st.session_state.df.head())
            st.markdown(f"- 資料筆數: {st.session_state.df.shape[0]}  \\  - 欄位數: {st.session_state.df.shape[1]}")
            st.subheader("欄位資料型別")
            st.dataframe(st.session_state.df.dtypes)
            st.subheader("缺失值概覽")
            st.dataframe(st.session_state.df.isnull().sum())
        else:
            st.info("按下按鈕以顯示資料載入後資料")


def mathod02():
    df = st.session_state.df
    if st.session_state.load_error:
        st.error("無法處理，因為資料載入時發生錯誤。請先確認資料是否可用。")
    else:
        if st.button("顯示處理後資料"):
            # 刪除缺失值並重置索引
            df_processed = df.dropna().reset_index(drop=True)
            # 轉換 CHAS、RAD 欄位為浮點數
            df_processed["CHAS"] = df_processed["CHAS"].astype(float)
            df_processed["RAD"] = df_processed["RAD"].astype(float)
            # 分割特徵矩陣 X (所有欄位除 MEDV) 與目標變數 y (MEDV)
            X = df_processed.drop(columns=["MEDV"])
            y = df_processed[["MEDV"]]
            st.session_state.df_processed = df_processed
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.processed = True

        # 如果已處理，顯示結果
        if st.session_state.get('processed', False):
            df_processed = st.session_state.df_processed
            X = st.session_state.X
            y = st.session_state.y
            st.subheader("📋 處理後資料預覽")
            st.dataframe(df_processed.head())
            st.markdown(
                f"- 原始資料筆數: {df.shape[0]}  \\  - 處理後資料筆數: {df_processed.shape[0]}"
            )

            st.subheader("🔢 特徵矩陣 X 預覽")
            st.dataframe(X.head())
            st.markdown(f"- X 維度: {X.shape}")
            st.subheader("🏷️ 目標變數 y 預覽")
            st.dataframe(y.head())
            st.markdown(f"- y 維度: {y.shape}")
        else:
            st.info("按下按鈕以顯示處理後資料")