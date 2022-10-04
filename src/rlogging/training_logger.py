from abc import abstractmethod
import os
import json
from datetime import datetime


class ReincryptLogger:
    def __init__(self, config:dict, tickers:list, output_dir: str):
        self.name = config['experiment_name']
        self.start = datetime.now()
        self.end = None
        self.duration = 0
        self.tickers = tickers
        self.config = config
        self.output_dir = f"{output_dir}/{self.name}"

    def finish(self):
        self.end = datetime.now()
        self.duration = self.end - self.start
        
    def log_2_file(self, result, file_prefix):
        os.makedirs(f"{self.output_dir}", exist_ok=True)
        timestamp = int(self.start.timestamp())
        with open(
            f"{self.output_dir}/{file_prefix}_{timestamp}.json", "w+") as js:
            json.dump(result, js, indent=2)


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
            "num_days": self.config["num_days"] 
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
        self.cumulative_asset = None

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
                "cumulative_asset": self.cumulative_asset,
                "num_days": self.config["num_days"]
            }
        }

        super(VerificationLogger, self).log_2_file(result=result,
                                                   file_prefix="verification")
        print("Verification logging finalized.")
