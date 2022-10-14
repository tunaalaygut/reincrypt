import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_daily(daily_data: list, date_begin: str, date_end: str, title: str,
               y_label: str, output_dir: str, output_fname: str, 
               file_suffix: int, color: str, extension="png"):
    # Create x and y data
    date_begin = datetime.strptime(date_begin, "%Y-%m-%d").date()
    date_end = datetime.strptime(date_end, "%Y-%m-%d").date()
    days = pd.date_range(date_begin, date_end, freq='d')
    
    plt.plot(days, daily_data, color=color)
    plt.title(title)
    plt.xlabel('Date')
    plt.xticks(rotation=90)  # To rotate dates
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(fname=f"{output_dir}/{output_fname}_{file_suffix}.{extension}",
                bbox_inches="tight")
