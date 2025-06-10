# Importing necessary libraries
import numpy as np
import pandas as pd
import cvxpy as cp

"""
Optimal allocation of each asset
"""

# Function to find the optimal allocation of our assets given a set of weights
# Please refer to the previous notebook for the explanation of the ideas behind this, especially when dealing with short positions
def allocation(prices, weights, total_investment, short_ratio=0.5, reinvest=False, remove_zero_investments=True, results=False):
    
    # Handle case where weights include short positions
    if (weights < 0).any():
        longs = weights[weights >= 0]
        shorts = weights[weights < 0]

        # Normalize both the long and short weights
        longs = longs / longs.sum()
        shorts = shorts / shorts.sum()

        # Calculate how much we have to allocate for both the long and short positions
        short_val = total_investment * short_ratio
        long_val = total_investment + short_val if reinvest else total_investment

        # Recursively apply function for long and short positions separately
        long_alloc, long_leftover = allocation(prices[longs.index], longs, long_val, reinvest=False, remove_zero_investments=remove_zero_investments, results=False)
        short_alloc, short_leftover = allocation(prices[shorts.index], shorts, short_val, reinvest=False, remove_zero_investments=remove_zero_investments, results=False)

        # Negate shorts to turn them into longs and get combined results
        short_alloc = -short_alloc
        combined_alloc = pd.concat([long_alloc, short_alloc]).groupby(level=0).sum()
        # if remove_zero_investments:
        #     combined_alloc = combined_alloc[combined_alloc != 0]
        combined_leftover = long_leftover + short_leftover

        # Print results if results=True, else return allocation and leftover amount
        if results:
            print(f"Remaining cash: {combined_leftover:.2f}\n")
            print(f"Allocation:\n{combined_alloc.to_string(dtype=False)}")

        # Return allocations and leftover amount
        return combined_alloc, combined_leftover

    # Long only positions
    # Define variables
    tickers = weights.index
    p = prices.to_numpy()
    w = weights.to_numpy()
    n = len(w)

    # Create cvxpy variables and ensure each allocation is an integer
    x = cp.Variable(n, integer=True)
    u = cp.Variable(n)

    # Calculate the amount of money spent from buying shares and how much we have leftover
    spent = p @ x
    leftover = total_investment - spent

    # Calculate the absolute deviation from target allocation
    target = w * total_investment
    error = target - cp.multiply(x, prices)

    # Construct the objective function and constraints
    objective = cp.Minimize(cp.sum(u) + leftover)
    constraints = [error <= u,
                   error >= -u,
                   x >= 0,
                   leftover >= 0]

    # Solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="ECOS_BB")

    # Find final allocation by removing assets with no investment into them if remove_zero_investments is trueinto them and set as a pandas series
    allocation_final = pd.Series(np.rint(x.value).astype(int), index=tickers)
    if remove_zero_investments:
        allocation_final = allocation_final[allocation_final != 0]

    # Print results if results=True, else return allocation and leftover amount
    if results and (weights > 0).all():
        print(f"Leftover cash: {leftover.value:.2f}\n")
        print(f"Allocation:\n{allocation_final.to_string(dtype=False)}")

    return allocation_final, leftover.value