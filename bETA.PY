import numpy as np
import matplotlib.pyplot as plt

# Define market stress scenarios (e.g., -5%, -10%, -15%, -20%, etc.)
market_stress_scenarios = np.array([-0.05, -0.10, -0.15, -0.20, -0.25])

# Define portfolio betas to test
portfolio_betas = np.array([0.5, 1.0, 1.5, 2.0])

# Calculate portfolio returns for each beta under different stress scenarios
portfolio_returns = np.outer(market_stress_scenarios, portfolio_betas)

# Plot the results
plt.figure(figsize=(8, 6))
for i, beta in enumerate(portfolio_betas):
    plt.plot(market_stress_scenarios * 100, portfolio_returns[:, i] * 100, label=f'Beta = {beta}')

# Add labels and title
plt.title('Portfolio Returns Under Stress Scenarios for Different Betas')
plt.xlabel('Market Change (%)')
plt.ylabel('Portfolio Return (%)')
plt.axhline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
