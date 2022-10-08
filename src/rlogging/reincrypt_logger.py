import os
import sys
sys.path.append("..")
from utility.plotter import plot_daily
import json
from datetime import datetime
from utility.util import get_sharpe_ratio


class ReincryptLogger:
    def __init__(self, config:dict, tickers:list, output_dir: str):
        self.name = config['experiment_name']
        self.start = datetime.now()
        self.end = None
        self.duration = 0
        self.tickers = tickers
        self.config = config
        self.output_dir = f"{output_dir}/{self.name}"
        self.date_begin = None
        self.date_end = None

    def finish(self):
        self.end = datetime.now()
        self.duration = self.end - self.start
        
    def log_2_file(self, result, file_prefix, file_suffix=None):
        os.makedirs(f"{self.output_dir}", exist_ok=True)

        if not file_suffix:
            file_suffix = int(self.start.timestamp())

        with open(
            f"{self.output_dir}/{file_prefix}_{file_suffix}.json", "w+") as js:
            json.dump(result, js, indent=2)
            
    def set_dates(self, date_begin=None, date_end=None):
        self.date_begin = date_begin
        self.date_end = date_end


class TrainingLogger(ReincryptLogger):
    def __init__(self, config:dict, tickers:list, output_dir: str):
        super(TrainingLogger, self).__init__(config, tickers, output_dir)
        print("Training logging initialized.")
        self.losses = list()

    def save(self):
        super(TrainingLogger, self).finish()

        result = {
            "training_start": str(self.start),
            "training_end": str(self.end),
            "training_duration (m)": self.duration.total_seconds()//60,
            "tickers": self.tickers,
            "config": self.config,
            "losses": self.losses,
            "num_days": self.config["num_days"],
            "date_begin": self.date_begin,
            "date_end": self.date_end 
        }

        super(TrainingLogger, self).log_2_file(result=result, 
                                               file_prefix="training")
        print("Training logging finalized.")

    def add_loss(self, loss):
        self.losses.append(loss)


class VerificationLogger(ReincryptLogger):
    def __init__(self, config:dict, tickers:list, output_dir: str):
        super(VerificationLogger, self).__init__(config, tickers, output_dir)
        print("Verification logging initialized.")
        self.num_currencies = None
        self.position_change = None
        self.final_cumulative_asset = None
        self.cumulative_assets = []
        self.avg_daily_returns = []

    def save(self):
        super(VerificationLogger, self).finish()
        
        result = {
            "verification_start": str(self.start),
            "verification_end": str(self.end),
            "verification_duration (m)": self.duration.total_seconds()//60,
            "verification_tickers": self.tickers,
            "config": self.config,
            "results": {    
                "num_currencies": self.num_currencies,
                "position_change": self.position_change,
                "cumulative_asset": self.final_cumulative_asset,
                "sharpe_ratio": get_sharpe_ratio(self.avg_daily_returns),
                "num_days": self.config["num_days"],
                "date_begin": self.date_begin,
                "date_end": self.date_end,
                "daily_results": self.create_daily_results()
            }
        }

        #TODO: Calculate suffix here and pass it to both plot and result JSON
        plot_daily(self.avg_daily_returns, date_begin=self.date_begin, 
                           date_end=self.date_end)
        super(VerificationLogger, self).log_2_file(result=result,
                                                   file_prefix="verification")
        print("Verification logging finalized.")

    def add_daily_results(self, cumulative_asset, avg_daily_return):
        self.cumulative_assets.append(cumulative_asset)
        self.avg_daily_returns.append(avg_daily_return)

    def create_daily_results(self) -> list:
        daily_results = []

        for day_idx, (cumulative_asset, avg_daily_return) \
            in enumerate(zip(self.cumulative_assets, self.avg_daily_returns)):
            daily_results.append(
                {
                    "day_index": day_idx,
                    "cumulative_asset": cumulative_asset,
                    "avg_daily_return": avg_daily_return
                }
            )

        return daily_results
