# Importing necessary libraries
import numpy as np
import pandas as pd
import portfolio_optimisers as opt
import allocations as alloc

"""
Full results for each strategy
"""

def full_results(df, ret, cov, weight_bounds, risk_free_rate, total_investment, short_ratio=0.5, results=False):
    # Calculate returns, volatilities, weights and sharpe ratios of all strategies
    opts = opt.optimisers(ret, cov, weight_bounds=weight_bounds)
    ret_max, risk_max, w_max = opts.maximise_return()
    sr_max = (ret_max - risk_free_rate)/risk_max
    ret_mr, risk_mr, w_mr = opts.minimise_risk(target_return=0)
    sr_mr = (ret_mr - risk_free_rate)/risk_mr
    ret_msr, risk_msr, w_msr = opts.maximise_sharpe_ratio(risk_free_rate=risk_free_rate)
    sr_msr = (ret_msr - risk_free_rate)/risk_msr

    # Create our metrics result dataframe
    metrics = pd.DataFrame({'Strategy': ['Max Return', 'Min Risk', 'Max Sharpe Ratio'],
                            'Return (%)': [ret_max*100, ret_mr*100, ret_msr*100],
                            'Volatility (%)': [risk_max*100, risk_mr*100, risk_msr*100],
                            'Sharpe Ratio': [sr_max, sr_mr, sr_msr]})
    metrics.set_index("Strategy", inplace=True)

    # Create multi index of strategies and tickers
    tickers = list(df.columns)*3
    length = len(df.columns)
    strats = ['Max Return']*length + ['Min Volatility']*length + ['Max Sharpe Ratio']*length
    arrays = [strats,tickers]

    # Calculate allocations and leftover amount of all strategies
    alloc_max, leftover_max = alloc.allocation(df.iloc[-1], w_max, total_investment=total_investment,short_ratio=short_ratio, remove_zero_investments=False)
    alloc_mr, leftover_mr = alloc.allocation(df.iloc[-1], w_mr, total_investment=total_investment,short_ratio=short_ratio, remove_zero_investments=False)
    alloc_msr, leftover_msr = alloc.allocation(df.iloc[-1], w_msr, total_investment=total_investment,short_ratio=short_ratio, remove_zero_investments=False)

    # Create our distribution dataframe
    distribution = pd.DataFrame({'Weights': np.concatenate([w_max, w_mr, w_msr]), 'Allocations': np.concatenate([alloc_max, alloc_mr, alloc_msr])}, index=arrays)
    distribution = distribution.rename_axis(index=['Strategy', 'Ticker'])

    # Create our leftovers dataframe
    leftovers = pd.DataFrame({'Strategy': ['Max Return', 'Min Risk', 'Max Sharpe Ratio'],
                              'Leftover Amount': [leftover_max, leftover_mr, leftover_msr]})
    leftovers.set_index("Strategy", inplace=True)

    # Display dataframes if results=True, else simply return dataframes
    if results:
        print(metrics)
        print(distribution)
        print(leftovers)
    
    return metrics, distribution, leftovers