# Importing necessary libraries
import numpy as np
import pandas as pd
import cvxpy as cp

# Small function to make our weights more readable
# We simply use a cutoff approach to set negligible weights to 0 and round all weights to 7dp
def final_weights(weights, cutoff=1e-5, rounding=7):
    weights[np.abs(weights) < cutoff] = 0
    final_weights = np.round(weights, rounding)
    return final_weights

"""
Class to find optimal weights for various strategies:
- Maximum Returns
- Minimal Risk
- Maximum Sharpe Ratio
"""

class optimisers:
    # Initialise class
    def __init__(self, ret, cov, weight_bounds):
        self.ret = ret
        self.cov = cov
        self.weight_bounds = weight_bounds

    # Function to find a set of weights that maximise our expected returns
    def maximise_return(self, results=False):
	    
	    # Create cvxpy variable to minimize
	    w = cp.Variable(len(self.ret))
	    
	    # Construct the objective function and constraints
	    obj = cp.Minimize(-w.T @ self.ret)
	    const = [cp.sum(w) == 1,
	             w >= self.weight_bounds[0],
	             w <= self.weight_bounds[1]]
	    
	    # Solve problem
	    prob = cp.Problem(obj, const)
	    opt_v = prob.solve()

	    # Check to see if our optimisation failed
	    if w.value is None:
	        raise RuntimeError("Optimization failed: no solution found. Please ensure appropriate expected returns and weight bounds were provided.")
	        
	    # Get our optimal weights and corresponding portfolio return and risk
	    w = w.value.round(10) + 0.0
	    returns = float(w.T @ self.ret.to_numpy())
	    risk = float(np.sqrt(w.T @ self.cov @ w))
	    # Put weights in a pandas series
	    weights = pd.Series(final_weights(w), index=self.ret.index.to_list())

	    # Print results if needed otherwise just return results
	    if results:
	        print(f'For our return maximising optimisation, we achieve results of:\n')
	        print('Return (%):', np.round(returns*100, 5))
	        print('Risk (%):', np.round(risk*100, 5))
	        print('Weights:\n')
	        print(weights.to_string(dtype=False))
	        return returns, risk, weights
	    else:
	        return returns, risk, weights

	# Function to find a set of weights that minimise our risk
	# Please refer to previous notebook on derivations and explanations behind this optimisation
    def minimise_risk(self, target_return, results=False):
	    # Ensure that the target return provided is actually attainable by ensuring that is below the maximum possible return
	    max_ret = self.maximise_return()[0]
	    if target_return > max_ret:
	        print(f'The target return of {target_return*100}% provided exceeds the maximum achievable return {max_ret*100:.3f}%.')
	        print(f'We will proceed with a target return of {np.floor(max_ret*1e3)/10}%.\n')
	        target_return = np.floor(max_ret*1e3)/1e3
	    
	    # Create cvxpy variable to minimize
	    w = cp.Variable(len(self.ret))
	    
	    # Construct the objective function and constraints
	    obj = cp.Minimize(w.T @ self.cov @ w)
	    const = [cp.sum(w) == 1,
	         (w.T @ self.ret) - target_return >= 0,
	         w >= self.weight_bounds[0],
	         w <= self.weight_bounds[1]]
	    
	    # Solve problem
	    prob = cp.Problem(obj, const)
	    try:
	        opt_v = prob.solve(solver=cp.OSQP)
	    except cp.SolverError:
	        try:
	            opt_v = prob.solve(solver=cp.ECOS)
	        except cp.SolverError:
	            opt_v = prob.solve(solver=cp.SCS)

	    # Check to see if our optimisation failed
	    if w.value is None:
	        raise RuntimeError("Optimization failed: no solution found. Please ensure appropriate expected returns, covariance matrix and weight bounds were provided.")
	    
	    # Get our optimal weights and corresponding portfolio return and risk
	    w = w.value.round(10) + 0.0
	    returns = float(w.T @ self.ret)
	    risk = float(np.sqrt(opt_v))
	    weights = pd.Series(final_weights(w), index=self.ret.index.to_list())

	    # Print results if needed otherwise just return results
	    if results:
	        print(f'For our efficient risk minimising optimisation with a target return of {np.floor(target_return*1e3)/10}%, we achieve results of:\n')
	        print('Return (%):', np.round(returns*100, 5))
	        print('Risk (%):', np.round(risk*100, 5))
	        print('Weights:\n')
	        print(weights.to_string(dtype=False))
	        return returns, risk, weights
	    else:
	        return returns, risk, weights

	# Function to find a set of weights that maximise our sharpe ratio
	# Please refer to previous notebook on derivations and explanations behind this optimisation
    def maximise_sharpe_ratio(self, risk_free_rate, results=False):
	    
	    # Create cvxpy variables
	    w = cp.Variable(len(self.ret))
	    k = cp.Variable()
	    
	    # Construct the objective function and constraints
	    obj = cp.Minimize(w.T @ self.cov @ w)
	    const = [w.T @ (self.ret - risk_free_rate) == 1,
	             cp.sum(w) == k,
	             w >= self.weight_bounds[0],
	             w <= self.weight_bounds[1],
	             k >= 0]
	    
	    # Solve problem
	    prob = cp.Problem(obj, const)
	    prob.solve()

	    # Check to see if our optimisation failed
	    if w.value is None:
	        raise RuntimeError("Optimization failed: no solution found. Please ensure appropriate expected returns, covariance matrix and weight bounds were provided.")
	        
	    # Get our optimal weights and corresponding portfolio return and risk
	    w = (w.value/k.value).round(10) + 0.0
	    returns = float(w.T @ self.ret)
	    risk = float(np.sqrt(w.T @ self.cov @ w))
	    weights = pd.Series(final_weights(w), index=self.ret.index.to_list())

	    # Print results if needed otherwise just return results
	    if results:
	        print('For our sharpe ratio maximising optimisation, we achieve results of:\n')
	        print('Return (%):', np.round(returns*100, 5))
	        print('Risk (%):', np.round(risk*100, 5))
	        print('Sharpe Ratio:', np.round(((returns - risk_free_rate)/ risk), 5))
	        print('Weights:\n')
	        print(weights.to_string(dtype=False))
	        return returns, risk, weights
	    else:
	        return returns, risk, weights