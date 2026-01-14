# 📈 Gold Price Prediction Using Machine Learning

An end-to-end machine learning system for predicting daily gold prices using market and macroeconomic indicators.  
The project includes data collection, preprocessing, model training, online learning, and deployment via a Streamlit interface.

---

## 🚀 Demo

🎥 **UI Demo Video:**  https://drive.google.com/file/d/1ULj6APPOjkeLSyJEjbMylsTE244YyuWn/view?usp=sharing

---

## 📌 Features

- Daily gold price prediction
- Time-series–aware training and evaluation
- Online learning using **SGD Regressor**
- Real-time data updates
- Actual vs Predicted visualization
- Interactive Streamlit UI

---

## 🧠 Models Used

- Linear Regression  
- Support Vector Regression (SVR)  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- **Stochastic Gradient Descent (SGD)** *(used for deployment due to incremental learning)*

---

## 📊 Data Sources

- **Yahoo Finance**
  - Gold Futures (GC=F)
  - Brent Crude Oil (BZ=F)
  - US Dollar Index (DXY)
- **FRED**
  - Effective Federal Funds Rate (EFFR)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- yfinance, pandas_datareader
- Streamlit
- Matplotlib

---

## ⚙️ Installation & Usage

### Clone the repository
```bash
git clone https://github.com/your-username/gold-price-prediction.git
cd gold-price-prediction

# Run by :

streamlit run app.py
