# Importing necessary libraries
import numpy as np
import pandas as pd
import cvxpy as cp
import portfolio_optimisers as opt
import allocations as alloc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to plot our covariances/correlations as a heatmap
def heatmap(df):

    sns.set_theme(style="white")
    # Generate a mask for the upper triangle since our results are symmetrical so we only need one diagonal of results
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Initialise matplotlib figure
    f, ax = plt.subplots(figsize=(12, 8))
    
    # Draw the heatmap
    sns.heatmap(df, ax=ax, mask=mask, cmap='RdYlGn', vmax=.3, center=0,
                annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Covariances")

    plt.show()
    return

"""
Efficient Frontier Plot
"""

# Function to plot our efficient frontier with random portfolios, where our minimum risk and maximum sharpe ratio portfolios are and Capital Market Line
# For certain weight bounds, mostly when we include negative short weights, our efficient frontier is extremely large, so we will produce another plot that is a zoomed in version
def efficient_frontier(ret, cov, risk_free_rate, weight_bounds, ef_points=500, n=50000):
    
    # Initialise a Plotly figure with subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=["Full Efficient Frontier", "Zoomed Efficient Frontier"], vertical_spacing=0.1)
    """
    Plot random portfolios
    """
    # Create random portfolios to calculate expected returns, risks and sharpe ratios for our plot
    random_w = []
    while len(random_w) < n:
        # Create a mask to ensure we get a set of weights where some assets have a weight of 0
        mask = np.random.binomial(1, 0.8, size=len(ret))
        w = np.random.uniform(weight_bounds[0], weight_bounds[1], size=len(ret)) * mask
        
        # Ensure weights do not sum to 0, so that we can normalise our weights to make them sum to 1
        w_sum = np.sum(w)
        if w_sum == 0:
            continue
        w = w / w_sum
        
        # Check all weights are still within bounds and append to final list of all weights
        if np.all(w >= weight_bounds[0]) and np.all(w <= weight_bounds[1]):
            random_w.append(w)

    # Calculate each portfolio's return, risk and sharpe ratio
    # Note how instead of using a massive for loop for each set of weights, we combine them all into one big array and only require one calculation to find each set of results
    random_w = np.array(random_w)
    random_rets = random_w @ ret
    random_risks = np.sqrt(np.einsum('ij,jk,ik->i', random_w, cov, random_w))
    random_sharpes = (random_rets-risk_free_rate) / random_risks

    # Plot our random portfolios (risk against return) and including a colour scale dependent on sharpe ratio
    Random_Portfolios = go.Scatter(x=random_risks*100,y=random_rets*100,
                                    mode="markers",
                                    name="Random Portfolios",
                                    marker=dict(size=6, opacity=0.5,
                                                color=random_sharpes,colorscale='Viridis',colorbar=dict(title='Sharpe Ratio', len=0.45, y=1.0, yanchor='top'),
                                                showscale=True),
                                    showlegend=True)
    Random_Portfolios2 = go.Scatter(x=random_risks*100,y=random_rets*100,
                                    mode="markers",
                                    name="Random Portfolios",
                                    marker=dict(size=6, opacity=0.5,
                                                color=random_sharpes,colorscale='Viridis',colorbar=dict(title='Sharpe Ratio', len=0.45, y=0.0, yanchor='bottom'),
                                                showscale=True),
                                    showlegend=False)
    fig.add_trace(Random_Portfolios, row=1, col=1)
    fig.add_trace(Random_Portfolios2, row=2, col=1)

    # Calculate the minimum and maximum risks and returns possible with the given set of assets
    opts = opt.optimisers(ret, cov, weight_bounds=weight_bounds)
    min_ret, min_risk = opts.minimise_risk(target_return=0)[:2]
    max_ret, max_risk = opts.maximise_return()[:2]
    """
    Plot Efficient Frontier
    """
    # Create a set of returns that are attainable 
    rets = np.linspace(min_ret, max_ret, ef_points)

    # For each return, we find the corresponding minimum risk 
    ret_ef=[]
    risk_ef=[]
    for i in rets:
        i_ret,i_risk = opts.minimise_risk(target_return=i)[:2]
        ret_ef.append(i_ret*100)
        risk_ef.append(i_risk*100)

    # Plot our calculated efficient frontier points
    Efficient_Frontier = go.Scatter(x=risk_ef,y=ret_ef,
                                    mode="lines",
                                    name="Efficient frontier",
                                    line=dict(width=3, color="lightskyblue"),
                                    showlegend=True)
    Efficient_Frontier2 = go.Scatter(x=risk_ef,y=ret_ef,
                                    mode="lines",
                                    name="Efficient frontier",
                                    line=dict(width=3, color="lightskyblue"),
                                    showlegend=False)
    fig.add_trace(Efficient_Frontier, row=1, col=1)
    fig.add_trace(Efficient_Frontier2, row=2, col=1)
    """
    Plot individual assets' returns and risks using our expected returns and covariance matrix
    """
    Assets = go.Scatter(x=np.sqrt(np.diag(cov))*100,y=ret*100,
                        mode="markers",
                        name="Assets",
                        marker=dict(size=8, symbol="star", color="black"),
                        text=ret.index.tolist(),
    					hoverinfo='text+x+y', 
                        showlegend=True)
    Assets2 = go.Scatter(x=np.sqrt(np.diag(cov))*100,y=ret*100,
                        mode="markers",
                        name="Assets",
                        marker=dict(size=8, symbol="star", color="black"),
                        text=ret.index.tolist(),
    					hoverinfo='text+x+y', 
                        showlegend=False)
    fig.add_trace(Assets, row=1, col=1)
    fig.add_trace(Assets2, row=2, col=1)
    """
    Plot the point on our efficient frontier where the portfolio with the minimal risk is
    """
    Min_Risk = go.Scatter(x=[min_risk*100],y=[min_ret*100],
                          mode="markers",
                          name="Minimum Volatility",
                          marker=dict(size=10, symbol="x", color="green"),
                          showlegend=True)
    Min_Risk2 = go.Scatter(x=[min_risk*100],y=[min_ret*100],
                          mode="markers",
                          name="Minimum Volatility",
                          marker=dict(size=10, symbol="x", color="green"),
                          showlegend=False)
    fig.add_trace(Min_Risk, row=1, col=1)
    fig.add_trace(Min_Risk2, row=2, col=1)
    """
    Plot the point on our efficient frontier where the portfolio with the maximal sharpe ratio is
    """
    ret_msr, risk_msr = opts.maximise_sharpe_ratio(risk_free_rate=0.041)[:2]
    Max_Sharpe_Ratio = go.Scatter(x=[risk_msr*100],y=[ret_msr*100],
                                  mode="markers",
                                  name="Maximum Sharpe Ratio",
                                  marker=dict(size=10, symbol="x", color="red"),
                                  showlegend=True)
    Max_Sharpe_Ratio2 = go.Scatter(x=[risk_msr*100],y=[ret_msr*100],
                                  mode="markers",
                                  name="Maximum Sharpe Ratio",
                                  marker=dict(size=10, symbol="x", color="red"),
                                  showlegend=False)
    fig.add_trace(Max_Sharpe_Ratio, row=1, col=1)
    fig.add_trace(Max_Sharpe_Ratio2, row=2, col=1)
    """
    Plot our Capital Market Line (CML)
    Refer to previous notebook where we proved that this line is:
    - Tangent to the Efficient Frontier
    - Intersects at the maximum sharpe ratio portfolio
    - Has a y-intercept at the risk free rate
    """
    # Our 2 points that will form the line
    x1, y1 = 0, risk_free_rate
    x2, y2 = risk_msr, ret_msr
    # Limit the range of the CML
    y_vals = np.array([risk_free_rate, max_ret])
    # Calculate slope (intercept is just simply the risk free rate) 
    m = (y2 - y1) / (x2 - x1)
    # Get corresponding x-values for the 2 y values
    x_vals = (y_vals - y1) / m
    # Plot CML
    CML = go.Scatter(x=x_vals*100, y=y_vals*100,
                     mode='lines',
                     name='Capital Market Line',
                     line=dict(color='brown', width=4, dash='dot'),
                     showlegend=True)
    CML2 = go.Scatter(x=x_vals*100, y=y_vals*100,
                     mode='lines',
                     name='Capital Market Line',
                     line=dict(color='brown', width=4, dash='dot'),
                     showlegend=False)
    fig.add_trace(CML, row=1, col=1)
    fig.add_trace(CML2, row=2, col=1)
    """
    Plot horizontal line representing risk free rate
    """
    rf_x = np.array([0,max_risk])
    rf_y = np.array([risk_free_rate,risk_free_rate])
    RF = go.Scatter(x=rf_x*100, y=rf_y*100,
                    mode='lines',
                    name='Risk Free Rate',
                    line=dict(color='orange', width=4, dash='dot'),
                    showlegend=True)
    RF2 = go.Scatter(x=rf_x*100, y=rf_y*100,
                    mode='lines',
                    name='Risk Free Rate',
                    line=dict(color='orange', width=4, dash='dot'),
                    showlegend=False)
    fig.add_trace(RF, row=1, col=1)
    fig.add_trace(RF2, row=2, col=1)
    """
    Update plot size, legend, titles and axes
    """
    fig.update_layout(width=1400,height=1800,
                     legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.6)', bordercolor='black', borderwidth=1),
                     margin=dict(l=100,r=100),
                     title=dict(text='Efficient Frontier Plots', font=dict(size=40)))
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=30)
    fig.update_xaxes(title='Annualised Volatility (%)', title_font=dict(size=30), row=1, col=1)
    fig.update_yaxes(title='Annualised Return (%)', title_font=dict(size=30), row=1, col=1)
    fig.update_xaxes(title='Annualised Volatility (%)', range=[0, random_risks.max()*100 +5], title_font=dict(size=30), row=2, col=1)
    fig.update_yaxes(title='Annualised Return (%)', range=[random_rets.min()*100 +5, random_rets.max()*100 +5], title_font=dict(size=30), row=2, col=1)
    
    # Show plot
    fig.show()
    return

"""
Plot the weights of each asset for each strategy
"""

def weights(ret, cov, weight_bounds, risk_free_rate=0.041):
    # Get an array of tickers in our portfolio
    tickers = ret.index.to_numpy()

    # Calculate the optimal weights for each strategy
    opts = opt.optimisers(ret, cov, weight_bounds=weight_bounds)
    _, _, w_max = opts.maximise_return()
    _, _, w_mr = opts.minimise_risk(target_return=0)
    _, _, w_msr = opts.maximise_sharpe_ratio(risk_free_rate=risk_free_rate)

    # Initialise plotly figure
    fig = go.Figure()

    # Plot bar charts, for our max return strategy we will make it invisible since the weights are extreme and not realisitc
    fig.add_trace(go.Bar(name='Max Return', x=tickers, y=w_max, visible='legendonly'))
    fig.add_trace(go.Bar(name='Min Risk', x=tickers, y=w_mr))
    fig.add_trace(go.Bar(name='Max Sharpe Ratio', x=tickers, y=w_msr))

    # Group bars and edit axes and title
    fig.update_layout(width=1400,height=900,
                      barmode='group',
                      title=dict(text='Asset Weights for Each Strategy', font=dict(size=40)))
    fig.update_xaxes(title='Assets', title_font=dict(size=30), tickangle=-45)
    fig.update_yaxes(title='Weight', title_font=dict(size=30))
    
    fig.show()
    return

"""
Plot the allocated shares of each asset for each strategy
"""

def allocations(df, ret, cov, weight_bounds, risk_free_rate=0.041, total_investment=10_000, short_ratio=0.5):
    # Get an array of tickers in our portfolio
    tickers = ret.index.to_numpy()

    # Calculate the optimal weights for each strategy
    opts = opt.optimisers(ret, cov, weight_bounds=weight_bounds)
    _, _, w_max = opts.maximise_return()
    _, _, w_mr = opts.minimise_risk(target_return=0)
    _, _, w_msr = opts.maximise_sharpe_ratio(risk_free_rate=risk_free_rate)

    alloc_max, _ = alloc.allocation(df.iloc[-1], w_max, total_investment=total_investment, short_ratio=short_ratio, remove_zero_investments=False)
    alloc_mr, _ = alloc.allocation(df.iloc[-1], w_mr, total_investment=total_investment, short_ratio=short_ratio, remove_zero_investments=False)
    alloc_msr, _ = alloc.allocation(df.iloc[-1], w_msr, total_investment=total_investment, short_ratio=short_ratio, remove_zero_investments=False)

    # Initialise plotly figure
    fig = go.Figure()

    # Plot bar charts
    fig.add_trace(go.Bar(name='Max Return', x=tickers, y=alloc_max))
    fig.add_trace(go.Bar(name='Min Risk', x=tickers, y=alloc_mr))
    fig.add_trace(go.Bar(name='Max Sharpe Ratio', x=tickers, y=alloc_msr))

    # Group bars and edit axes and title
    fig.update_layout(width=1400,height=900,
                      barmode='group',
                      title=dict(text='Number of Allocated Shares of each Asset for each Strategy', font=dict(size=40)))
    fig.update_xaxes(title='Assets', title_font=dict(size=30), tickangle=-45)
    fig.update_yaxes(title='Number of Shares', title_font=dict(size=30))
    
    fig.show()
    return