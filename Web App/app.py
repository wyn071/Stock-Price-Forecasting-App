from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

# Horizons we support, in days
HORIZONS = [7, 30, 90, 180, 365]


def fetch_and_clean(ticker):
    """Fetch live data for a ticker and run it through the cleaning pipeline."""
    stock = yf.Ticker(ticker)
    data = stock.history(start="2019-01-01")

    if data.empty:
        return None

    data = data.reset_index()
    data = data.rename(columns={"Date": "ds", "Close": "y"})
    data["ds"] = data["ds"].dt.tz_localize(None)
    data = data[["ds", "y"]]

    # Cleaning pipeline
    data = data.dropna()
    data = data.drop_duplicates(subset=["ds"])
    data = data.sort_values("ds").reset_index(drop=True)

    return data


def evaluate_horizon(data, horizon_days):
    """Train on everything except the last `horizon_days`, test on that holdout window."""
    cutoff = data["ds"].max() - pd.DateOffset(days=horizon_days)
    train = data[data["ds"] <= cutoff]
    test = data[data["ds"] > cutoff]

    if len(train) < 60 or len(test) == 0:
        return None

    m = Prophet(daily_seasonality=False)
    m.fit(train)

    future = m.make_future_dataframe(periods=horizon_days)
    forecast = m.predict(future)
    forecast = forecast[["ds", "yhat"]]

    results = test.merge(forecast, on="ds")
    if results.empty:
        return None

    mae = (results["y"] - results["yhat"]).abs().mean()
    mape = ((results["y"] - results["yhat"]).abs() / results["y"]).mean() * 100
    rmse = ((results["y"] - results["yhat"]) ** 2).mean() ** 0.5

    return {"horizon": horizon_days, "mae": round(mae, 2), "mape": round(mape, 2), "rmse": round(rmse, 2)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json or {}
    ticker = payload.get("ticker", "GOOG").upper()
    horizon = int(payload.get("horizon", 30))

    if horizon not in HORIZONS:
        return jsonify({"error": f"Horizon must be one of {HORIZONS}"}), 400

    data = fetch_and_clean(ticker)
    if data is None:
        return jsonify({"error": f"No data found for ticker '{ticker}'. Please check the symbol and try again."}), 400

    if len(data) < 90:
        return jsonify({"error": f"Not enough history for '{ticker}' to make a reliable forecast."}), 400

    # --- Main forecast for the chosen horizon ---
    m = Prophet(daily_seasonality=False)
    m.fit(data)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

    # Recent actual history to plot alongside the forecast (last 180 days for context)
    history = data.tail(180).copy()
    history["ds"] = history["ds"].dt.strftime("%Y-%m-%d")

    # --- Accuracy metrics across all horizons (backtested) ---
    accuracy_table = []
    for h in HORIZONS:
        metrics = evaluate_horizon(data, h)
        if metrics:
            accuracy_table.append(metrics)

    # --- Plain-language summary: current price vs target date price ---
    current_price = round(float(data["y"].iloc[-1]), 2)
    current_date = data["ds"].iloc[-1].strftime("%Y-%m-%d")

    target_row = result.iloc[-1]
    target_price = round(float(target_row["yhat"]), 2)
    target_date = target_row["ds"]
    target_low = round(float(target_row["yhat_lower"]), 2)
    target_high = round(float(target_row["yhat_upper"]), 2)

    change_amount = round(target_price - current_price, 2)
    change_percent = round((change_amount / current_price) * 100, 2)

    summary = {
        "current_price": current_price,
        "current_date": current_date,
        "target_price": target_price,
        "target_date": target_date,
        "target_low": target_low,
        "target_high": target_high,
        "change_amount": change_amount,
        "change_percent": change_percent
    }

    return jsonify({
        "ticker": ticker,
        "horizon": horizon,
        "history": history.to_dict(orient="records"),
        "forecast": result.to_dict(orient="records"),
        "accuracy_table": accuracy_table,
        "summary": summary
    })


if __name__ == "__main__":
    app.run(debug=True)