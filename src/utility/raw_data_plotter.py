import os
import sys
import pandas as pd
from plotter import plot_daily
from datetime import datetime

RAW_DATA_PATH = sys.argv[1]
FEATURE = sys.argv[2]
CURRENCY = os.path.basename(RAW_DATA_PATH).split("_")[1].split(".")[0]


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    plot_daily(daily_data=df[FEATURE], date_begin=df.Date.values[0],
               date_end=df.Date.values[-1],
               title=f"{FEATURE} by Date for {CURRENCY}", y_label=FEATURE,
               output_dir=".", output_fname=f"{CURRENCY}_raw_data_{FEATURE}",
               file_suffix=int(datetime.now().timestamp()), color="green",
               mark_anomalies=True)


if __name__ == "__main__":
    main()
