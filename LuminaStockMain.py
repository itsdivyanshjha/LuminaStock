import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Function to download data
def download_data(stock_symbols, start_date, end_date):
    stock_data = {}
    for symbol in stock_symbols:
        data = yf.download(symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime here
        stock_data[symbol] = data
    return stock_data


# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to preprocess data and train LSTM model
def preprocess_and_train(stock_data):
    df = stock_data
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)  # Ensure there are no NAs
    
    # Normalize the entire dataset
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Open', 'Close', 'Low', 'High', 'Volume']])
    
    # Create sequences for LSTM
    seq_length = 50
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build LSTM Model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(25),
        Dense(5) 
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
    
    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_split=0.1, callbacks=[early_stopping], verbose=0)
    
    return model, history.history

# Function to make predictions
def predict_stock_prices(stock_data, model):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data[['Open', 'Close', 'Low', 'High', 'Volume']])
    
    # Create sequences for LSTM
    seq_length = 50
    X, y = create_sequences(scaled_data, seq_length)
    
    # Predict using the model
    predictions = model.predict(X)
    
    # Inverse scaling for predictions if necessary
    # predictions = scaler.inverse_transform(predictions)
    
    return predictions, y


# Visualize historical stock prices
def visualize_stock_prices(stock_data):
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
    plt.plot(stock_data['Date'], stock_data['Open'], label='Open Price')
    plt.plot(stock_data['Date'], stock_data['High'], label='High Price')
    plt.plot(stock_data['Date'], stock_data['Low'], label='Low Price')
    plt.title('Historical Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Visualize volume trends
def visualize_volume(stock_data):
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data['Date'], stock_data['Volume'], label='Volume')
    plt.title('Volume Trend')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    st.pyplot(plt)

# Visualize model loss and metrics
def visualize_model_metrics(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

# Visualize residuals
def visualize_residuals(actual_prices, predictions):
    residuals = actual_prices - predictions
    plt.figure(figsize=(12, 8))
    plt.plot(residuals, marker='o', linestyle='None')
    plt.title('Residuals Plot')
    plt.xlabel('Data Point')
    plt.ylabel('Residual')
    st.pyplot(plt)

# Calculate Daily Returns
def calculate_daily_returns(stock_data):
    returns = stock_data['Close'].pct_change()
    return returns

# Calculate Moving Average
def calculate_moving_average(stock_data, window=20):
    moving_avg = stock_data['Close'].rolling(window=window).mean()
    return moving_avg

# Calculate Risk (Standard Deviation)
def calculate_risk(returns):
    risk = returns.std()
    return risk

# Inverse Scaling Function
def inverse_scale_data(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

# Streamlit App
def main():
    st.title("Welcome to LuminaStock: Illuminate Your Investments")
    st.write("LuminaStock leverages cutting-edge LSTM machine learning models to provide precise stock price predictions, helping you navigate the complexities of the financial markets with confidence. Our tool empowers investors by offering deep insights and actionable forecasts, so you can see smarter and invest better. Whether you're a seasoned trader or just starting out, LuminaStock is your go-to platform for enhancing your investment strategy and optimizing your portfolio's performance.")
    
    # User inputs
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL):")
    start_date = st.date_input("Enter Start Date:")
    end_date = st.date_input("Enter End Date:")
    
    if st.button("Predict"):
        if ticker and start_date and end_date:
            # Download data
            stock_data = download_data([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))[ticker]

            # Train LSTM model
            model, history = preprocess_and_train(stock_data)

            # Fit scaler
            scaler = MinMaxScaler()
            scaler.fit(stock_data[['Open', 'Close', 'Low', 'High', 'Volume']])

            # Make predictions
            predictions, actual_prices = predict_stock_prices(stock_data, model)

            # Calculate R^2
            r2 = r2_score(actual_prices, predictions)

            # Display performance metrics
            st.subheader("Performance Metrics")
            st.write("Mean Absolute Error:", history['val_mean_absolute_error'][-1])
            st.write("Mean Squared Error:", history['val_mean_squared_error'][-1])
            st.write("R^2 Score:", r2)

            # Visualize actual prices and predictions
            st.subheader("Actual Prices vs. Predicted Prices")
            df = pd.DataFrame({'Actual': actual_prices.flatten(), 'Predicted': predictions.flatten()})
            st.line_chart(df)

            # Display table of predicted values
            st.subheader("Table of Predicted Values")
            st.write("The table showcases predicted values for various stock market indicators, including open, close, low, high prices, and volume. Each row provides insights into forecasted trends, aiding investors in decision-making and strategic planning.")
            st.write(pd.DataFrame(predictions, columns=['Predicted Open', 'Predicted Close', 'Predicted Low', 'Predicted High', 'Predicted Volume']))
            st.write("-----------------------------------------------------------------------")
            st.write("")

            # Additional visualizations
            st.subheader("Historical Stock Prices")
            st.write("This section presents visualizations of historical stock prices, offering a graphical representation of the stock's performance over time. By analyzing past trends, users gain valuable insights into price fluctuations and potential future patterns, aiding in informed investment decisions.")
            visualize_stock_prices(stock_data)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            st.subheader("Stock Volumes")
            st.write("This section highlights stock trading volumes through visualizations, illustrating the quantity of shares exchanged over time. By examining volume trends alongside price movements, users can better gauge market activity and potential shifts in investor sentiment, enhancing their understanding of stock dynamics for strategic decision-making.")
            visualize_volume(stock_data)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            st.subheader("Model Metrics")
            st.write("Stock Data Model Metrics: This section provides insights into the performance of the predictive model through visual representations of key metrics. By examining model metrics such as loss and accuracy over training epochs, users can assess the model's effectiveness in forecasting stock prices, aiding in the evaluation and refinement of predictive strategies.")
            visualize_model_metrics(history)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            st.subheader("Model Residuals")
            st.write("Stock Data Model Residuals: This section displays the residuals of the predictive model, showcasing the differences between actual and predicted stock prices. By visualizing these residuals, users can evaluate the model's performance in capturing the variability in stock prices and identifying any patterns or trends in prediction errors, facilitating further model refinement and enhancement.")
            visualize_residuals(actual_prices, predictions)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            # Calculate and display additional metrics
            daily_returns = calculate_daily_returns(stock_data)
            st.subheader("Daily Returns")
            st.write("This section computes and visualizes the daily returns of the selected stock, representing the percentage change in price from one trading day to the next. Analyzing daily returns helps investors understand the volatility and potential profitability of their investment over time.")
            st.line_chart(daily_returns)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            moving_avg = calculate_moving_average(stock_data)
            st.subheader("Moving Average")
            st.write("This segment calculates the moving average of the stock's closing prices, providing a smoothed representation of its price trend over a specified time window. Visualizing the moving average aids investors in identifying long-term trends and potential buy or sell signals in the stock's performance.")
            st.line_chart(moving_avg)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            risk = calculate_risk(daily_returns)
            st.subheader("Risk (Standard Deviation)")
            st.write("This section calculates the risk associated with the investment by determining the standard deviation of daily returns. A higher standard deviation implies greater volatility, indicating the level of uncertainty or potential fluctuations in investment returns over time.")
            st.write(risk)
            st.write("-----------------------------------------------------------------------")
            st.write("")

            # Inverse scale the predicted values
            inverse_scaled_predictions = inverse_scale_data(predictions, scaler)

            # Display table of inverse scaled predicted values
            st.subheader("Table of Inverse Scaled Predicted Values")
            st.write(" ")
            st.write(pd.DataFrame(inverse_scaled_predictions, columns=['Predicted Open', 'Predicted Close', 'Predicted Low', 'Predicted High', 'Predicted Volume']))

        else:
            st.write("Please fill in all the fields.")


if __name__ == "__main__":
    main()