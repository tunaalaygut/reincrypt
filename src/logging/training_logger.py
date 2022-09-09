import os
import json
from datetime import datetime


class TrainingLogger:
    def __init__(self, name:str, hyperparameters:dict, tickers:list):
        self.name = name
        self.training_start = datetime.now()
        self.training_end = None
        self.training_duration = 0
        self.tickers = tickers
        self.hyperparameters = hyperparameters
        self.losses = list()

    def save(self):
        self.__finish_training()
        result = {
            "name": self.name,
            "training_start": str(self.training_start),
            "training_end": str(self.training_end),
            "training_duration (m)": self.training_duration.total_seconds()//60,
            "tickers": self.tickers,
            "hyperparameters": self.hyperparameters,
            "losses": self.losses 
        }
        os.makedirs("log", exist_ok=True)
        with open(
            f"log/{self.name}_training_log_{int(self.training_start.timestamp())}.json", 
            "w+") as js:
            json.dump(result, js, indent=2)

    def add_loss(self, loss):
        self.losses.append(loss)

    def __finish_training(self):
        self.training_end = datetime.now()
        self.training_duration = self.training_end - self.training_start
