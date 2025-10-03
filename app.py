import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model("lstm_stock_forecast.h5")

# Define scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def forecast(prices):
    if isinstance(prices, str):
        prices = [float(x.strip()) for x in prices.split(",") if x.strip()]

    data = np.array(prices).reshape(-1, 1)
    scaled = scaler.fit_transform(data)

    # Pad if less than 100
    if len(scaled) < 100:
        pad_length = 100 - len(scaled)
        scaled = np.vstack([np.full((pad_length, 1), scaled[0]), scaled])

    temp_input = list(scaled[-100:].flatten())
    output = []

    for i in range(30):
        x = np.array(temp_input[-100:]).reshape(1, 100, 1)
        yhat = model.predict(x, verbose=0)
        temp_input.append(yhat[0][0])
        output.append(yhat[0][0])

    forecasted = scaler.inverse_transform(np.array(output).reshape(-1, 1)).flatten().tolist()

    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(range(len(prices)), prices, label="Original Prices")
    ax.plot(range(len(prices), len(prices) + 30), forecasted, label="Forecasted Prices", color="red")
    ax.set_title("Stock Price Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    return fig, forecasted


demo = gr.Interface(
    fn=forecast,
    inputs=gr.Textbox(label="Enter prices (comma-separated) 100days recommended"),
    outputs=[
        gr.Plot(label="Forecasted Prices Graph"),
        gr.Textbox(label="Forecasted Next 30days Prices List")
    ],
    title="Stock Price Forecasting with LSTM"
)

if __name__ == "__main__":
    demo.launch()
