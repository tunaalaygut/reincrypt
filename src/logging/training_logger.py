import os
import json
from datetime import datetime


class TrainingLogger:
    def __init__(self, config:dict, tickers:list, output_dir: str):
        self.name = config['experiment_name']
        self.training_start = datetime.now()
        self.training_end = None
        self.training_duration = 0
        self.tickers = tickers
        self.config = config
        self.losses = list()
        self.output_dir = f"{output_dir}/{self.name}"

    def save(self):
        self.__finish_training()
        result = {
            "training_start": str(self.training_start),
            "training_end": str(self.training_end),
            "training_duration (m)": self.training_duration.total_seconds()//60,
            "tickers": self.tickers,
            "config": self.config,
            "losses": self.losses 
        }
        os.makedirs(f"{self.output_dir}", exist_ok=True)
        
        timestamp = int(self.training_start.timestamp())
        
        with open(
            f"{self.output_dir}/log_{timestamp}.json", "w+") as js:
            json.dump(result, js, indent=2)

    def add_loss(self, loss):
        self.losses.append(loss)

    def __finish_training(self):
        self.training_end = datetime.now()
        self.training_duration = self.training_end - self.training_start
