# Stock Price Forecasting App

A machine learning web app that forecasts stock prices using **Facebook Prophet** with backtested accuracy metrics across multiple forecast horizons (7 to 365 days).

**Try it here:** https://wyn071-stock-price-forecaster.hf.space

---

## What it does

- Fetches live stock data from Yahoo Finance
- Trains a Facebook Prophet time series model on historical closing prices
- Forecasts prices across 5 horizons: 1 week, 30, 90, 180, and 365 days
- Displays backtested accuracy metrics (MAPE, MAE, RMSE) for each horizon
- Automatically selects the best-performing model (Baseline vs Volume-enhanced) per horizon based on backtesting results

---

## Important note

- Forecast accuracy degrades predictably with horizon length (from ~4% MAPE at 7 days, to ~31% at 365 days)

---


# Screenshots

![alt text](Screenshots/image.png)
![alt text](Screenshots/image-1.png)
![alt text](Screenshots/image-2.png)
![alt text](Screenshots/image-4.png)
![alt text](Screenshots/image-5.png)
![alt text](Screenshots/image-6.png)
![alt text](Screenshots/image-7.png)
