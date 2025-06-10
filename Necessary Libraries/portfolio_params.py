# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import covariance
from sklearn.covariance import ledoit_wolf

"""
Class to find:
- Expected returns using CAPM
- Covariance matrix using Lediot-Wolf Shrinkage Estimator
- Correlation matrix
"""

class parameters:
	# Initialise class
    def __init__(self, price_df, frequency, risk_free_rate):
        self.price_df = price_df
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate

    # Function to calculate expected returns using CAPM
	# Please refer to previous notebook for further information
    def expected_returns_capm(self):
        # Make a copy of returns dataframe to calculate the mean of all returns
        temp = self.price_df.copy()
        temp['mean_ret'] = self.price_df.mean(axis=1)
        
        # Calculate covariance and betas of assets
        cov = temp.cov()
        beta = cov["mean_ret"] / cov.loc["mean_ret", "mean_ret"]
        beta = beta.drop('mean_ret')
        mean_mkt_ret = (1 + temp['mean_ret']).prod() ** (self.frequency / len(temp['mean_ret'])) - 1
        
        # Calculate expected returns for each asset
        mu = self.risk_free_rate + beta * (mean_mkt_ret - self.risk_free_rate)

        return mu

    # Function to calculate covariances between each pair of assets using Ledoit-Wolf Shrinkage Estimator
	# Please refer to previous notebook for further information
    def covariance_matrix_ledoit_wolf(self):
        # Use sklearn's ledoit_wolf function
        cov, shrinkage = ledoit_wolf(self.price_df)
    
        # Multiply by time period to annualise covariances
        cov = cov * self.frequency
    
        # Return results in a dataframe
        cov_df = pd.DataFrame(cov, columns=self.price_df.columns.to_numpy(), index=self.price_df.columns.to_numpy())

        return cov, cov_df

    # Function to calculate the correlations between any pair of assets based on our earlier calculated covariances
    def correlation_matrix(self):
        cov, _ = self.covariance_matrix_ledoit_wolf()
        V = np.diag(1 / np.sqrt(np.diag(cov)))
        corr = V @ cov @ V
        corr_df = pd.DataFrame(corr, index=self.price_df.columns.to_numpy(), columns=self.price_df.columns.to_numpy())

        return corr_df