# **Trading Algorithms**

## Table of Contents

- [Introduction](#introduction)
- [Why have I made this?](#why-have-i-made-this)
- [Optimisations](#optimisations)
- [Features](#features)
- [What have I learnt from this?](#what-have-i-learnt-from-this)
- [Improvements to be made](#improvements-to-be-made)
- [File Structure](#file-structure)
- [How to use this?](#how-to-use-this)

## Introduction
This repository contains a detailed look into how one can go about connecting all the projects we have made thus far to build and backtest a trading algorithm. The aim is to build a strategy that uses the same stocks as the S&P500 index, but by calculating the features, returns, risks and etc, and then by clustering our stocks (using our K-Means Clustering Algorithm project), we will only take the top stocks based off their sharpe ratios and then create a monthly portfolio out of these at the start of every month. Then we will apply Portfolio Optimisation to find the optimal allocated shares of each stock we should invest in each month by maximising our sharpe ratios. Finally, we will Backtest this against a strategy of simply buying and holding the S&P500 index from the start of our investment period to the end.

To best understand how this works, I will firstly give a flowchart showing how our trading algorithm will work.

1) Firstly, we will get the historial OHLCV data for all S&P500 stocks from the past 8 years
2) Then, we will calculate features for all these stocks, namely:
    -  MACD
    -  RSI
    -  MFI
    -  ATR
    -  Bollinger Bands
    -  Returns with various monthly lookbacks
    -  Fama-French five-factor model
3) Now we will use our K-Means Clustering Algorithm to effectively cluster each month's stocks based on all the above features. We will also use the Silhouette Score method to choose the best number of clusters to use for each month
4) After clustering, we will choose the best cluster, and all the stocks in said cluster, each month based off the Sharpe ratios to invest in at the start of the subsequent month
5) Then for each group of stocks we would like to invest in each month, we will perform Portfolio Optimisation to find the optimal allocated number of integer shares to invest in each stock
6) Finally, we will backtest this strategy by calculating the returns, volatility, Sharpe ratio, and other metrics and compare this against a simple benchmark of buying and holding the S&P500 index

## Why have I made this?
This project was created to extend the foundational ideas of Markowitz’s portfolio optimisation into a practical unsupervised learning trading strategy. Specifically, I wanted to apply these ideas to a dynamic, monthly rebalanced portfolio built using K-Means clustering on S&P 500 constituent stocks and their technical features we derived in another project. This project bridges the gap between theory and practice, demonstrating how unsupervised learning and portfolio optimisation can work together to potentially outperform a passive investment in the market.

## Optimisations

I have implemented the following optimisations:
- Dynamic clustering to allow the number of clusters ($k$) to adapt based on Silhouette Score. Re-calculate clusters every month instead of fixing them, to capture evolving market conditions
- Used various technical indicators (RSI, MACD, ATR, Bollinger Bands and etc) instead of just simply returns/volatility
- Used shrinkage for covariance matrices in portfolio optimisation to avoid overfitting
- Experimented with different rebalancing periods (monthly, quarterly) to see what works best
- Used walk-forward analysis rather than a single backtest to avoid lookahead bias

## Features

This project inclues the following features:
- Displays other metrics: maximum drawdown, volatility, alpha and beta against S&P500
- Uses a benchmark of S&P500

## What have I learnt from this?

Throughout this project, I’ve deepened my understanding of how unsupervised learning techniques like K-Means clustering can complement financial strategies, specifically in dynamically selecting high-performing stocks from the S&P 500. By building and testing a monthly rebalanced portfolio, I learned the practical challenges of translating theoretical portfolio optimisation, often limited to static weight derivation, into a real-world, adaptive trading strategy.

Working with financial data at this scale also taught me the importance of robust feature engineering and the role of different risk/return metrics like the Sharpe ratio in guiding decision making. I realised how crucial it is to go beyond simple historical returns and incorporate diverse features, such as volatility and risk measures, to create more resilient stock groupings.

Moreover, I saw first-hand how backtesting frameworks can reveal gaps between expected and actual performance, reinforcing the need for incorporating transaction costs, turnover constraints, and risk controls. Finally, this project has shown me that iterative refinement, whether in selecting features, optimising clusters, or tweaking rebalancing intervals, is key to evolving an algorithmic strategy that can adapt to changing market environments.

## Improvements to be made

There is one big improvement to be made that will significantly elevate this strategy. Although we find good times to enter the market with investing in profitable stocks, we simply hold our stocks for a full month and then close our position everywhere and start again. Clearly, our exit strategy is not good since anything could have happened within that month. So in order to truly elevate this, we must incorporate an exit strategy, whether that be a simple Stop Loss/Take Profit exit strategy dependent on the ATR and closing prices, or a more complex Hawkes Process exit strategy.

Some smaller improvements are:
- Incorporate estimated transaction costs and slippage into the portfolio rebalancing step
- Use PCA or t-SNE to visualise the feature space and clusters

## File Structure
This repository is split into 2 folders, one containing all the prerequiste files needed to be imported and in the file directory for our trading algorithm to run. The second contains the actual trading algorithm in the form of a Jupyter Notebook.

## How to use this?

Please ensure all necessary libraries at the top of the trading algorithm are either in the same file directory or already installed in Python. Then, I would recommend going through the notebook cell by cell, to best understand the flow of how it works and to then go and apply this thinking to your own trading algorithm.

