import sys
sys.path.append("..")
import os
from utility.util import read_config
from rlogging.training_logger import TrainingLogger, VerificationLogger
import argparse
from train import Agent
from data_reader import DataReader
from verification import test_mnp


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', required=True, type=str,
                    help="Path to input data.")
parser.add_argument('-c', '--config', required=True, type=str,
                    help="Name of the config file.")
parser.add_argument('-v', '--verification', action="store_true",
                    help="Whether to work in training or verification mode.")
parser.add_argument("-m", "--model", required=False, type=str,
                    help="Path to pretrained model. Required if -v is set.")
args = vars(parser.parse_args())

# Globals
DATA_DIR = args["input_path"]
CONFIG_FILENAME = args["config"]
IS_TRAINING = not args["verification"]
MODEL_PATH = args["model"] if args["verification"] else None
if args["verification"] and args["model"] is None:
    parser.error("-v requires -m")

TICKERS = os.listdir(DATA_DIR)
DATA_DIRS = [os.path.join(DATA_DIR, curr_data) for curr_data in TICKERS]
OUTPUT_DIR = "output"


def main():
    config = read_config(config_filename=CONFIG_FILENAME,
                         output_dir=OUTPUT_DIR)
    data_reader = DataReader()
    X, y = data_reader.read(DATA_DIRS, limit=20)
    config["height"], config["width"] = X[0][0].shape[0], X[0][0].shape[1]

    if IS_TRAINING:
        agent = Agent(config)
        agent.set_data(X, y, config)
        logger = TrainingLogger(config=config, tickers=TICKERS,
                                output_dir=OUTPUT_DIR)
        agent.train(config, logger=logger)
        logger.save()
    else:
        logger = VerificationLogger(config=config, tickers=TICKERS,
                                    output_dir=OUTPUT_DIR)
        test_mnp(X, y, config, MODEL_PATH, logger)
        logger.save()


if __name__ == "__main__":
    main()
