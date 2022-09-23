import os
import sys
import json
from data_reader import DataReader
from train import Agent
sys.path.append("../logging")
from training_logger import TrainingLogger


# TODO: Implement arg parser instead of this
DATA_DIR = sys.argv[1]
CONFIG_FILENAME = sys.argv[2]
CONFIG = dict()
OUTPUT_DIR = "output"


def main():
    read_config()
    tickers = os.listdir(DATA_DIR)
    logger = TrainingLogger(config=CONFIG, tickers=tickers, 
                            output_dir=OUTPUT_DIR)

    agent = Agent(CONFIG)

    data_dirs = [os.path.join(DATA_DIR, curr_data) for curr_data in tickers]
    data_reader = DataReader()
    X, y = data_reader.read(data_dirs)

    agent.set_data(X, y)
    agent.train(CONFIG, logger=logger)

    logger.save()


def read_config():
    global CONFIG

    with open(f"config/{CONFIG_FILENAME}.json", "r+") as f:
        CONFIG = json.load(f)
    CONFIG["experiment_name"] = CONFIG_FILENAME

    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
