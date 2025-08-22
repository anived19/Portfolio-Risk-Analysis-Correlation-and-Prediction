# Portfolio-Risk-Analysis-Correlation-and-Prediction
This project dives into statistical modeling to perform correlation and prediction of stocks which can be dynamically input as well as the no. of shares for the stocks, i used geometric brownian motion to simulate the stock paths and monte carlo simulations for the outputs, used yfinance to procure stock data

Monte Carlo Portfolio Simulation & Risk Analysis
This Python script provides a tool for analyzing the potential future performance and risk of a stock portfolio using a Monte Carlo simulation. It fetches real historical stock data, models the behavior of individual assets and their correlations, and simulates thousands of possible future scenarios to provide a probabilistic forecast.


Features
Interactive Portfolio Input: Allows users to dynamically enter the number of assets, their stock tickers, and the number of shares for each.

Live Financial Data: Fetches the last two years of historical stock data from Yahoo Finance using the yfinance library.

Statistical Analysis: Automatically calculates key financial metrics from historical data for each asset:

Annualized Drift: The average annual rate of return.

Annualized Volatility: The measure of the stock's price fluctuation.

Correlation Modeling: Calculates the correlation matrix for the assets in the portfolio, ensuring the simulation realistically models how stocks move together.

Monte Carlo Simulation: Runs 1,000 simulations to forecast the portfolio's value over the next year (252 trading days).

Rich Visualizations: Generates two plots using matplotlib for easy interpretation:

Simulated Portfolio Paths: A "spaghetti" plot showing a sample of possible future paths for the portfolio's value, including the overall expected (mean) path.

Distribution of Final Values: A histogram showing the frequency of different outcomes at the end of the one-year period.

Key Risk Metrics: Calculates and displays critical statistics for risk assessment:

Expected Return: The average expected return on the portfolio.

Value at Risk (VaR): The potential loss in value for a given confidence level (provides 5% and 1% VaR).

Probability of Loss: The percentage of simulations that resulted in a final value lower than the initial value.
