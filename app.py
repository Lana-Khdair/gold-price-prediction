
# streamlit run app.py

import streamlit as st
import pickle
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- FULL PIPELINE FUNCTION ---
def run_pipeline():
    # -----------------------
    # LOAD PIPELINE
    # -----------------------
    pipeline = joblib.load("sgd_pipeline.pkl")
    feature_cols = ["High","Low","Open","Volume","Brent_Price","DXY_Price","Fed_Rate"]
    pipeline.last_train_date = getattr(pipeline, "last_train_date", datetime(2025, 11, 30).date())

    pipeline.last_fed_rate = getattr(pipeline, "last_fed_rate", None)
    
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    # -----------------------
    # DATA FETCH FUNCTIONS
    # -----------------------
    def fetch_gold(start, end):
        df = yf.download("GC=F", start=start, end=end + timedelta(days=1),
                         interval="1d", auto_adjust=False, progress=False).dropna()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Date","High","Low","Open","Volume","Close"]].rename(columns={"Close":"Gold_Price"})
    
    def fetch_brent(start, end):
        df = yf.download("BZ=F", start=start, end=end + timedelta(days=1),
                         interval="1d", progress=False, auto_adjust=False).dropna()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Date","Close"]].rename(columns={"Close":"Brent_Price"})
    
    def fetch_dxy(start, end):
        df = yf.download("DX-Y.NYB", start=start, end=end + timedelta(days=1),
                         interval="1d", auto_adjust=False, progress=False).dropna()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Date","Close"]].rename(columns={"Close":"DXY_Price"})
    
    def fetch_fed(start, end):
        df = DataReader("EFFR","fred", start, end).reset_index().rename(columns={"DATE":"Date","EFFR":"Fed_Rate"})
        df["Date"] = pd.to_datetime(df["Date"])
        daily_index = pd.date_range(start, end, freq="D")
        df = df.set_index("Date").reindex(daily_index).ffill().reset_index().rename(columns={"index":"Date"})
        return df
    
    def merge_features(gold, brent, dxy, fed):
        df = gold.merge(brent, on="Date", how="left")\
                 .merge(dxy, on="Date", how="left")\
                 .merge(fed, on="Date", how="left")
        df[["Brent_Price","DXY_Price","Fed_Rate"]] = df[["Brent_Price","DXY_Price","Fed_Rate"]].ffill().bfill()
        df[["High","Low","Open","Volume","Gold_Price"]] = df[["High","Low","Open","Volume","Gold_Price"]].interpolate()
        return df.dropna(subset=["High","Low","Open","Volume"]).sort_values("Date")
    
    def next_trading_day(date):
        d = date + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        return d

    # -----------------------
    # ONLINE LEARNING
    # -----------------------
    # -----------------------
# ONLINE LEARNING
# -----------------------
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    if pipeline.last_train_date >= yesterday:
      st.info("📌 Model already updated for today. Skipping online learning.")
    else:
     learn_start = pipeline.last_train_date + timedelta(days=1)
     df_temp_gold = fetch_gold(learn_start, yesterday)
     learn_end = None if df_temp_gold.empty else df_temp_gold["Date"].max().date()

     if learn_end is not None and learn_start <= learn_end:
        df_learn = merge_features(fetch_gold(learn_start, learn_end),
                                  fetch_brent(learn_start, learn_end),
                                  fetch_dxy(learn_start, learn_end),
                                  fetch_fed(learn_start, learn_end))
        if pipeline.last_fed_rate is not None:
            df_learn["Fed_Rate"] = df_learn["Fed_Rate"].fillna(pipeline.last_fed_rate)
        df_learn_complete = df_learn.dropna(subset=feature_cols)
        if not df_learn_complete.empty:
            X_new = df_learn_complete[feature_cols]
            y_new = df_learn_complete["Gold_Price"]
            pipeline.named_steps['sgd'].partial_fit(pipeline.named_steps['scaler'].transform(X_new), y_new)
            pipeline.last_train_date = df_learn_complete["Date"].max().date()
            pipeline.last_fed_rate = df_learn_complete["Fed_Rate"].iloc[-1]
            joblib.dump(pipeline, "sgd_pipeline.pkl")


    # -----------------------
    # NEXT-DAY PREDICTION
    # -----------------------
    df_pred = merge_features(fetch_gold(pipeline.last_train_date - timedelta(days=10), pipeline.last_train_date),
                             fetch_brent(pipeline.last_train_date - timedelta(days=10), pipeline.last_train_date),
                             fetch_dxy(pipeline.last_train_date - timedelta(days=10), pipeline.last_train_date),
                             fetch_fed(pipeline.last_train_date - timedelta(days=10), pipeline.last_train_date))
    df_pred = df_pred.dropna(subset=feature_cols)
    last_row = df_pred.iloc[-1]
    next_date = next_trading_day(last_row["Date"].date())
    X_pred_df = pd.DataFrame([last_row[feature_cols]], columns=feature_cols)
    predicted_price = pipeline.predict(X_pred_df)[0]

    # -----------------------
    # EVALUATION
    # -----------------------
    start_date = datetime(2024,11,30).date()
    end_date = pipeline.last_train_date
    df_show = merge_features(fetch_gold(start_date, end_date),
                             fetch_brent(start_date, end_date),
                             fetch_dxy(start_date, end_date),
                             fetch_fed(start_date, end_date))
    df_show = df_show.dropna(subset=feature_cols + ["Gold_Price"]).reset_index(drop=True)
    df_show["Predicted"] = pipeline.predict(df_show[feature_cols])

    # Save results to pickle
    data_to_save = {
        "df_show": df_show,
        "next_day_price": predicted_price,
        "prediction_date": next_date,
        "last_trained_date": pipeline.last_train_date
    }
    with open("gold_data_all.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    return df_show, predicted_price, next_date, pipeline.last_train_date

# -----------------------
# STREAMLIT APP
# -----------------------
st.title("📈 Gold Price Prediction System")

# Button to run/update pipeline
if st.button("🔄 Update"):
    with st.spinner("Running full pipeline, please wait..."):
        df_show, predicted_price, next_date, last_trained_date = run_pipeline()
   
else:
    # Load last saved data
    with open("gold_data_all.pkl", "rb") as f:
        data = pickle.load(f)
    df_show = data["df_show"]
    predicted_price = data["next_day_price"]
    next_date = data["prediction_date"]
    last_trained_date = data["last_trained_date"]

# Show last trained date
st.info(f"🛠️ Model last trained up to: {last_trained_date}")

# Show next-day prediction
st.subheader("🔮 Next-Day Prediction")
st.success(f"Predicted Gold Price for {next_date}: ${predicted_price:.2f}")

# Show Actual vs Predicted chart (from 30/11/2024)
start_plot_date = pd.to_datetime("2024-11-30")
df_plot = df_show[df_show["Date"] >= start_plot_date].copy()
df_plot["Date"] = pd.to_datetime(df_plot["Date"])

st.subheader("📊 Actual vs Predicted")
st.line_chart(df_plot.set_index("Date")[["Gold_Price", "Predicted"]])



