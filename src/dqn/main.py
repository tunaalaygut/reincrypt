import sys
sys.path.append("..")
import os
from utility.util import read_config, populate_config
from rlogging.reincrypt_logger import TrainingLogger, VerificationLogger
import argparse
from train import Agent
from data_reader import DataReader
from verification import test_mnp, test_tbk


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
parser.add_argument("-dl", "--data-limit", required=False, type=int,
                    help="Limit data being read to this number.")
args = vars(parser.parse_args())

# Globals
DATA_DIR = args["input_path"]
CONFIG_FILENAME = args["config"]
IS_TRAINING = not args["verification"]
MODEL_PATH = args["model"] if args["verification"] else None
if args["verification"] and args["model"] is None:
    parser.error("-v requires -m")
DATA_LIMIT = args["data_limit"]

TICKERS = os.listdir(DATA_DIR)
DATA_DIRS = [os.path.join(DATA_DIR, curr_data) for curr_data in TICKERS]
OUTPUT_DIR = "output"


def main():
    config = read_config(config_filename=CONFIG_FILENAME,
                         output_dir=OUTPUT_DIR)
    data_reader = DataReader()
    X, y, date_begin, date_end = data_reader.read(DATA_DIRS, limit=DATA_LIMIT)
    populate_config(config, X)

    logger = None

    if IS_TRAINING:
        agent = Agent(config)
        agent.set_data(X, y, config)
        logger = TrainingLogger(config=config, tickers=TICKERS,
                                output_dir=OUTPUT_DIR)
        agent.train(config, logger=logger)
    else:
        logger = VerificationLogger(config=config, tickers=TICKERS,
                                    output_dir=OUTPUT_DIR)
        # TODO: Make type of portfolio to run parametric
        test_tbk(X, y, 0.2, config, MODEL_PATH, logger)
        # test_mnp(X, y, config, MODEL_PATH, logger)

    logger.set_dates(date_begin, date_end)
    logger.save()


if __name__ == "__main__":
    main()
