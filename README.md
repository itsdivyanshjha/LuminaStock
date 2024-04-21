# LuminaStock

# Illuminate Your Investments: See Smarter, Invest Better

LuminaStock is a Streamlit web application that leverages cutting-edge LSTM machine learning models to provide precise stock price predictions. Whether you're a seasoned trader or just starting out, LuminaStock empowers investors by offering deep insights and actionable forecasts, helping you navigate the complexities of the financial markets with confidence.

## Features

- **Stock Price Prediction**: Utilizes LSTM models to predict future stock prices based on historical data.
- **Performance Metrics**: Provides key performance metrics such as Mean Absolute Error, Mean Squared Error, and R^2 Score to evaluate model accuracy.
- **Visualizations**: Offers interactive visualizations of historical stock prices, volume trends, model metrics, and residuals to aid in analysis and decision-making.
- **Additional Metrics**: Computes and displays daily returns, moving averages, and risk (standard deviation) to further enhance understanding of investment performance.

## Installation

To run the LuminaStock app locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
    ```
    pip install -r requirements.txt
    ```
3. Run the Streamlit app using the following command:
    ```
    streamlit run LuminaStockMain.py
    ```

## Usage

1. Launch the app by running `streamlit run LuminaStockMain.py` in your terminal.
2. Enter the stock ticker symbol (e.g., AAPL) and select the start and end dates for the desired historical data.
3. Click on the "Predict" button to generate stock price predictions and visualize the results.
4. Explore the different sections of the app, including performance metrics, visualizations, and additional metrics.

## Contributors

- [Divyansh Jha](https://github.com/itsdivyanshjha) 

## License

This project is licensed under the [MIT License](LICENSE).
