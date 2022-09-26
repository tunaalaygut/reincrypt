import os
import sys
import json
from data_reader import DataReader
from train import Agent
sys.path.append("../logging")
from training_logger import TrainingLogger


# Command line arguments
# TODO: Implement arg parser instead of this
DATA_DIR = sys.argv[1]
CONFIG_FILENAME = sys.argv[2]

# Globals
TICKERS = os.listdir(DATA_DIR)
DATA_DIRS = [os.path.join(DATA_DIR, curr_data) for curr_data in TICKERS]
CONFIG = dict()
OUTPUT_DIR = "output"


def main():
    read_config()

    agent = Agent(CONFIG)
    data_reader = DataReader()

    X, y = data_reader.read(DATA_DIRS)
    agent.set_data(X, y, CONFIG)

    logger = TrainingLogger(config=CONFIG, tickers=TICKERS, 
                            output_dir=OUTPUT_DIR)
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
