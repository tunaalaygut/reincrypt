import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from util import get_anomalies



def plot_daily(daily_data: list, date_begin: str, date_end: str, title: str,
               y_label: str, output_dir: str, output_fname: str, 
               file_suffix: int, color: str, extension="png", 
               mark_anomalies=False, figsize=(10, 6), dpi=600):
    # Create x and y data
    date_begin = datetime.strptime(date_begin, "%Y-%m-%d").date()
    date_end = datetime.strptime(date_end, "%Y-%m-%d").date()
    days = pd.date_range(date_begin, date_end, freq='d')
    
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(days, daily_data, color=color, label=y_label)

    if mark_anomalies:
        df = pd.DataFrame(columns=['daily_data', 'is_anomaly', 'date'])
        df.daily_data = daily_data
        df.date = days
        df = get_anomalies(df, columns=["daily_data"], window=30)
        anomaly_df = df[df["daily_data_is_anomaly"]] 
        ax.scatter(anomaly_df.date, anomaly_df['daily_data'], color='red',
                   label='Anomaly', s=3)
    plt.title(title)
    plt.xlabel('Date')
    plt.xticks(rotation=90)  # To rotate dates
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname=f"{output_dir}/{output_fname}_{file_suffix}.{extension}",
                bbox_inches="tight")
