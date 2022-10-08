import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_daily(daily_returns, date_begin, date_end):
    #TODO: Make title, output and y-label parametric
    date_begin = datetime.strptime(date_begin, "%Y-%m-%d").date()
    date_end = datetime.strptime(date_end, "%Y-%m-%d").date()
    days = pd.date_range(date_begin, date_end, freq='d')
    
    plt.plot(days, daily_returns, color='red')
    plt.title('Average Return per Date', fontsize=14)
    plt.xlabel('Date', fontsize=10)
    plt.xticks(rotation=90)
    plt.ylabel('Average Daily Return', fontsize=10)
    plt.grid(True)
    plt.savefig(fname="../dqn/output/deneme.png")
