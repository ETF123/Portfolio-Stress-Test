import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Beta
def calculate_beta(stock_symbol, market_symbol='^GSPC', period='1y'):
    # Download stock and market data
    stock_data = yf.download(stock_symbol, period=period)['Adj Close']
    market_data = yf.download(market_symbol, period=period)['Adj Close']

    # Calculate daily returns
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

    # Ensure both series have the same length
    min_length = min(len(stock_returns), len(market_returns))
    stock_returns = stock_returns[-min_length:]
    market_returns = market_returns[-min_length:]

    # Calculate covariance between stock and market
    covariance = np.cov(stock_returns, market_returns)[0][1]
    
    # Calculate variance of market
    market_variance = np.var(market_returns)

    # Calculate beta
    beta = covariance / market_variance

    return beta

# List of stocks
stocks = ['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN']

# Calculate Beta for each stock
betas = {}
for stock in stocks:
    beta = calculate_beta(stock)
    betas[stock] = beta
    print(f"Beta of {stock}: {beta:.2f}")

# Convert to DataFrame
df_betas = pd.DataFrame(list(betas.items()), columns=['Stock', 'Beta'])


#import matplotlib.pyplot as plt

# Assign calculated Betas to the portfolio and define weights
portfolio = {
    'Asset': ['AAPL', 'NVDA', 'MSFT', 'GOOG', 'AMZN'],
    'Beta': [betas['AAPL'], betas['NVDA'], betas['MSFT'], betas['GOOG'], betas['AMZN']],
    'Weight': [0.25, 0.20, 0.20, 0.15, 0.20]  # Adjusted portfolio weights
}

# Convert to DataFrame
df_portfolio = pd.DataFrame(portfolio)

# Calculate portfolio's weighted Beta
df_portfolio['Weighted Beta'] = df_portfolio['Beta'] * df_portfolio['Weight']
portfolio_beta = df_portfolio['Weighted Beta'].sum()

# Simulate random market stress scenarios
np.random.seed(42)  # For reproducibility
market_changes = np.random.normal(loc=0, scale=0.15, size=1000)  # Simulate 1000 market changes

# Calculate estimated portfolio change for each scenario
portfolio_changes = market_changes * portfolio_beta

# Plot histogram of estimated portfolio changes
plt.figure(figsize=(8, 6))
plt.hist(portfolio_changes * 100, bins=30, edgecolor='black', alpha=0.7, color='blue')

# Add labels and title
plt.title('Frequency Distribution of Estimated Portfolio Changes')
plt.xlabel('Estimated Portfolio Change (%)')
plt.ylabel('Frequency')

# Display the plot
plt.grid(True)
plt.show()

# Output results
print("Portfolio Beta: {:.2f}".format(portfolio_beta))

# Plot the market changes vs. portfolio changes
plt.figure(figsize=(8, 6))
plt.plot(market_changes * 100, portfolio_changes * 100, label=f'Portfolio Beta: {portfolio_beta:.2f}', color='blue')

# Add labels and title
plt.title('Simulated Portfolio Returns Under Market Stress')
plt.xlabel('Market Change (%)')
plt.ylabel('Estimated Portfolio Change (%)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
