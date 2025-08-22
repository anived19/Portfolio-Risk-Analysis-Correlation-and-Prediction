import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

#Interactive inputs
n = int(input("Enter number of assets: "))
tickers = []
shares = []

try:
    for i in range(n):
        t = str(input(f"Enter stock ticker {i+1}: ")).upper()
        tickers.append(t)
        s = float(input(f"Enter number of shares for {t}: "))
        shares.append(s)
except:
    # Fallback values
    tickers = ["AAPL", "TSLA", "NVDA"]
    shares = [10, 5, 8]
    n = len(tickers)
#get actual stock data(using yfinance)
def get_stock_data(symbol, period="2y"):
    
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    
    # Calculate daily returns
    returns = hist['Close'].pct_change().dropna()
    
    # Get latest price
    latest_price = hist['Close'].iloc[-1]
    
    return returns, latest_price

#calculate annual drift and vollatility using returns
def calculate_drift_volatility(returns):
    #Annualize the mean return (drift)
    mu = returns.mean() * 252  # 252 trading days per year
    
    #Annualize the volatility
    sigma = returns.std() * np.sqrt(252)
    
    return mu, sigma

def normalize_shares_to_weights(shares, prices):
    #Calculate total portfolio value
    portfolio_values = np.array(shares) * np.array(prices)
    total_value = np.sum(portfolio_values)
    
    #Calculate weights (normalized to sum to 1)
    weights = portfolio_values / total_value
    
    return weights, portfolio_values, total_value

#Get real stock data and calculate parameters
print("\nFetching stock data and calculating parameters...")

S0 = []  # Starting prices
mu = []  # Drift rates
sigma = []  # Volatilities
all_returns = []

for ticker in tickers:
    try:
        returns, price = get_stock_data(ticker)
        drift, vol = calculate_drift_volatility(returns)
        
        S0.append(price)
        mu.append(drift)
        sigma.append(vol)
        all_returns.append(returns)
        
        print(f"{ticker}: Price=${price:.2f}, Drift={drift:.3f}, Volatility={vol:.3f}")
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        #Use fallback values
        S0.append(100)
        mu.append(0.08)
        sigma.append(0.20)

#Convert to numpy arrays
S0 = np.array(S0)
mu = np.array(mu)
sigma = np.array(sigma)
shares = np.array(shares)

#Calculate portfolio weights from share quantities
weights, individual_values, total_portfolio_value = normalize_shares_to_weights(shares, S0)

print(f"\nPortfolio Composition:")
for i in range(n):
    print(f"{tickers[i]}: {shares[i]} shares @ ${S0[i]:.2f} = ${individual_values[i]:.2f} ({weights[i]*100:.1f}%)")
print(f"Total Portfolio Value: ${total_portfolio_value:.2f}")

#Calculate correlation matrix from historical returns
print("\nCalculating correlation matrix from historical data...")
returns_matrix = np.column_stack([returns.values for returns in all_returns])

#Ensure all return series have the same length by taking the minimum
min_length = min(len(returns) for returns in all_returns)
aligned_returns = np.column_stack([returns.values[-min_length:] for returns in all_returns])

corr_matrix = np.corrcoef(aligned_returns.T)
print("Correlation Matrix:")
print(corr_matrix)

#Create covariance matrix and Cholesky decomposition
cov_matrix = np.outer(sigma, sigma) * corr_matrix
L = np.linalg.cholesky(cov_matrix)

#Time setup
T = 1  # 1 year
N = 252  # trading days
dt = T / N

#Monte Carlo simulation parameters
M = 1000  #number of simulations

#Initialize price paths
price_paths = np.zeros((M, N + 1, n))
price_paths[:, 0, :] = S0

#Simulate GBM with correlated shocks
print(f"\nRunning {M} Monte Carlo simulations...")
for m in range(M):
    Z = np.random.normal(size=(N, n))
    correlated_Z = Z @ L.T
    for t in range(1, N + 1):
        price_paths[m, t, :] = price_paths[m, t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * correlated_Z[t - 1]
        )

#Compute portfolio values using actual share quantities
#Multiply each price path by corresponding number of shares
portfolio_values = np.sum(price_paths * shares, axis=2)  #shape (M, N+1)

#Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

#Left plot: Simulation paths
#Plot subset of paths for clarity
sample_paths = min(100, M)
for i in range(sample_paths):
    axs[0].plot(portfolio_values[i], alpha=0.3, color='lightblue')

mean_path = portfolio_values.mean(axis=0)
axs[0].plot(mean_path, color="darkblue", linewidth=2, label="Expected Value")

axs[0].set_title("Sample Simulated Portfolio Paths")
axs[0].set_xlabel("Time Step (Day)")
axs[0].set_ylabel("Portfolio Value ($)")
axs[0].grid(True)
axs[0].legend()

#Add portfolio info as text box
portfolio_info = "Portfolio Composition:\n" + '\n'.join(
    f"{tickers[i]}: {shares[i]} shares ({weights[i]*100:.1f}%)" 
    for i in range(n)
) + f"\nTotal Initial Value: ${total_portfolio_value:.2f}"

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axs[0].text(0.05, 0.95, portfolio_info, transform=axs[0].transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

#Right plot: Histogram of final portfolio values
final_values = portfolio_values[:, -1]
axs[1].hist(final_values, bins=50, edgecolor='black', alpha=0.7)
axs[1].set_title("Distribution of Final Portfolio Values")
axs[1].set_xlabel("Portfolio Value at T ($)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

#Calculate statistics
VaR_95 = np.percentile(final_values, 5)
VaR_99 = np.percentile(final_values, 1)
expected_final = np.mean(final_values)
pct_loss = np.mean(final_values < total_portfolio_value) * 100

#Plot reference lines
axs[1].axvline(VaR_95, color='red', linestyle='--', label=f'VaR (5%): ${VaR_95:,.0f}')
axs[1].axvline(VaR_99, color='darkred', linestyle='--', label=f'VaR (1%): ${VaR_99:,.0f}')
axs[1].axvline(total_portfolio_value, color='orange', linestyle='--', 
               label=f'Initial Value: ${total_portfolio_value:,.0f}')
axs[1].axvline(expected_final, color='green', linestyle='--', 
               label=f'Expected Final: ${expected_final:,.0f}')

axs[1].legend()

#Add statistics box
stats_text = f"Expected Return: {((expected_final/total_portfolio_value - 1)*100):.1f}%\n" \
             f"Chance of Loss: {pct_loss:.1f}%\n" \
             f"VaR (5%): ${VaR_95:,.0f}\n" \
             f"VaR (1%): ${VaR_99:,.0f}"

axs[1].text(0.97, 0.97, stats_text, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props)

plt.tight_layout()

#Print summary statistics
print(f"\n=== PORTFOLIO SIMULATION RESULTS ===")
print(f"Initial Portfolio Value: ${total_portfolio_value:,.2f}")
print(f"Expected Final Value: ${expected_final:,.2f}")
print(f"Expected Return: {((expected_final/total_portfolio_value - 1)*100):.2f}%")
print(f"Standard Deviation of Final Values: ${np.std(final_values):,.2f}")
print(f"Value at Risk (5% worst case): ${VaR_95:,.2f}")
print(f"Value at Risk (1% worst case): ${VaR_99:,.2f}")
print(f"Probability of Loss: {pct_loss:.2f}%")

plt.show()