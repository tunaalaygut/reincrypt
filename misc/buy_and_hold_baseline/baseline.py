import pandas as pd
import matplotlib.pyplot as plt

# Define a function to read currency data and return a DataFrame
def read_currency_data(file_path, currency_name):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    # Only keep 'Date' and 'Adj Close', and rename 'Adj Close' to the currency name
    return df[['Date', 'Adj Close']].rename(columns={'Adj Close': currency_name})

# List of file names for each currency
file_names = ['BTC', 'DGC', 'FTC', 'GLC', 'LTC', 'NMC', 'PPC', 'TRC']
file_paths = [f'{name}.csv' for name in file_names]  # Replace with your actual path

# Read the data for all currencies
all_data = [read_currency_data(file, name) for file, name in zip(file_paths, file_names)]

# Use functools.reduce to iteratively merge the DataFrames on the 'Date' column
from functools import reduce
portfolio_daily = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), all_data)

# Calculate the mean of the adjusted close values for each date
portfolio_daily['Mean Adj Close'] = portfolio_daily.iloc[:, 1:].mean(axis=1)

# Normalize the daily closing values to start with a cumulative asset of 1
portfolio_daily['Normalized Close'] = portfolio_daily['Mean Adj Close'] / portfolio_daily['Mean Adj Close'].iloc[0]

# Sort the DataFrame by date to ensure the plot is in chronological order
portfolio_daily.sort_values('Date', inplace=True)

# Plot the normalized daily closing values
plt.figure(figsize=(10, 6))
plt.plot(portfolio_daily['Date'], portfolio_daily['Normalized Close'])
plt.title('Buy-and-Hold Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Asset Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
