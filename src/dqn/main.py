import os
import sys

from verification import test_market_neutralized_portfolio
sys.path.append("..")
from rlogging.training_logger import TrainingLogger
from utility.util import read_config
from data_reader import DataReader
from train import Agent

# Command line arguments
# TODO: Implement arg parser instead of this
DATA_DIR = sys.argv[1]
CONFIG_FILENAME = sys.argv[2]
MODEL_PATH = ""

try:
    MODEL_PATH = sys.argv[3]
except IndexError as e:
    pass

# Globals
TICKERS = os.listdir(DATA_DIR)
DATA_DIRS = [os.path.join(DATA_DIR, curr_data) for curr_data in TICKERS]
OUTPUT_DIR = "output"


def main():
    config = read_config(config_filename=CONFIG_FILENAME, output_dir=OUTPUT_DIR)

    agent = Agent(config)
    data_reader = DataReader()

    X, y = data_reader.read(DATA_DIRS)
    agent.set_data(X, y, config)

    logger = TrainingLogger(config=config, tickers=TICKERS, 
                            output_dir=OUTPUT_DIR)
    if MODEL_PATH == "":
        agent.train(config, logger=logger)
        logger.save()
    else:
        test_market_neutralized_portfolio(MODEL_PATH, config, X, y)


if __name__ == "__main__":
    main()
