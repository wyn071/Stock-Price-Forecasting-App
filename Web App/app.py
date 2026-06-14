from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.json.get("ticker", "GOOG").upper()

    # Fetch the live stock data
    stock = yf.Ticker(ticker)
    data = stock.history(start="2019-01-01")

    if data.empty:
        return jsonify({"error": f"No data found for ticker '{ticker}'. Please check the symbol and try again."}), 400

    # Prepare the fetched data for the Prophet model
    data = data.reset_index()
    data = data.rename(columns={"Date": "ds", "Close": "y"})
    data["ds"] = data["ds"].dt.tz_localize(None)
    data = data[["ds", "y"]]

    # Train the Prophetmodel and then make predictions
    m = Prophet(daily_seasonality=False)
    m.fit(data)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Send back the dates and predicted prices
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(365)
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

    return jsonify({
        "ticker": ticker,
        "forecast": result.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)