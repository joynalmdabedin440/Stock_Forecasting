# 📈 Stock Price Forecasting Web App

This is a complete **Stock Price Forecasting** project that uses **LSTM (Long Short-Term Memory)** and **ARIMA** models to predict future stock prices.  
The project includes **data collection from Tiingo API**, preprocessing, model training, comparison of performance metrics, and deployment of the better-performing model via **Gradio** to Hugging Face Spaces.

---

## 🔍 Project Overview

The goal of this project is to build a robust forecasting system for stock prices using time series analysis and deep learning techniques.

**Main steps include:**
1. Collecting stock price data from **Tiingo API**
2. Preprocessing the data for model training
3. Building and training **LSTM** and **ARIMA** models
4. Comparing their performance using **MAPE** and **RMSE**
5. Selecting the best model (**LSTM** in this case)
6. Deploying the forecasting system as a **Gradio web app** on Hugging Face Spaces

---

## ⚙️ Methodology

### **Data Collection**
- Stock price data is collected using the [Tiingo API](https://api.tiingo.com/).
- Data includes historical closing prices for the chosen stock symbol.

### **Preprocessing**
- Missing values are handled appropriately.
- Data is normalized using `MinMaxScaler` for LSTM training.
- Time series structure is created for sequential modeling.

### **Model Building**
- **ARIMA model** — statistical time-series model for baseline forecasting.
- **LSTM model** — deep learning model designed to capture time dependencies in sequential data.

### **Model Comparison**
Both models were evaluated using:
- **MAPE (Mean Absolute Percentage Error)**
- **RMSE (Root Mean Squared Error)**

**Result:**  
LSTM outperformed ARIMA in terms of accuracy and was selected for deployment.

---

## 📊 Features of the App

- Predicts stock prices for the next 30 days using past data.
- Interactive web interface using **Gradio**.
- Displays both:
  - **Graph of forecasted prices**
  - **List of forecasted values**
- Handles varying lengths of input data.

---

## 🛠 Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** — LSTM model training
- **statsmodels** — ARIMA model
- **scikit-learn** — MinMaxScaler
- **Matplotlib** — Plotting
- **Gradio** — Web interface
- **Tiingo API** — Data collection
- **Hugging Face Spaces** — Deployment platform

---

## 📂 Repository Structure
├── model_training.ipynb # Notebook for training LSTM and ARIMA models
├── lstm_stock_forecast.h5 # Trained LSTM model
├── app.py # Gradio web app
├── requirements.txt # Required Python packages
├── README.md # Project description
├── utils/ # Helper functions (optional)
└── data/ # Raw and processed data

---

# 📈 LSTM Stock Forecasting App

## 🚀 Live Demo
**Try the app:** [View on Hugging Face Spaces](https://huggingface.co/spaces/amdjoynal443/DataSynthis_ML_JobTask)

---

## 📦 Quick Deploy to Hugging Face

### Required Files:
- `app.py` - Main Gradio application
- `lstm_stock_forecast.h5` - Trained LSTM model
- `requirements.txt` - Python dependencies

### Dependencies:
Create a `requirements.txt` file with:
```txt
gradio
tensorflow
scikit-learn
matplotlib
numpy
pandas
statsmodels
