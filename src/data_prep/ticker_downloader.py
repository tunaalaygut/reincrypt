import os
import yfinance as yf

TICKERS_LIST_PATH = "../../crypto_tickers.txt"
TICKERS = open(TICKERS_LIST_PATH, 'r+').read().splitlines()

OUTPUT_DIRECTORY = "../../crypto_data"
PERIOD = "max"
INTERVAL = "1d"


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    for ticker_name in TICKERS:
        ticker_data = yf.download(tickers=ticker_name, 
                                  period=PERIOD,
                                  interval=INTERVAL)

        ticker_data = ticker_data.reset_index()
        ticker_data = ticker_data.sort_values(by="Date", 
                                              ascending=True)
        ticker_data.to_csv(
            f'{OUTPUT_DIRECTORY}/{len(ticker_data):04d}_{ticker_name}.csv', 
            index=False)

        print(f"{ticker_name} downloaded and written to file.")


if __name__ == "__main__":
    main()
