import sys
sys.path.append("..")
import os
from utility.r_utils import read_config, populate_config
from rlogging.reincrypt_logger import TrainingLogger, VerificationLogger
import argparse
from train import Agent
from data_reader import DataReader
from verification import test_portfolio


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', required=True, type=str,
                    help="Path to input data.")
parser.add_argument('-c', '--config', required=True, type=str,
                    help="Name of the config file.")
parser.add_argument('-v', '--verification', action="store_true",
                    help="Whether to work in training or verification mode.")
parser.add_argument('-s', '--verification-start', type=int, required=False,
                    help="Verification data date start index. Inclusive.")
parser.add_argument('-e', '--verification-end', type=int, required=False,
                    help="Verification data date end index. Inclusive.")
parser.add_argument("-m", "--model", required=False, type=str,
                    help="Path to pretrained model. Required if -v is set.")
parser.add_argument("-k", "--topbottomk", required=False, type=float,
                    help="K for top/bottom K portfolio")
parser.add_argument("-r", "--random", action="store_true",
                    help="Whether to use random actions in portfolio")
args = vars(parser.parse_args())

# Globals
DATA_DIR = args["input_path"]
CONFIG_FILENAME = args["config"]
IS_TRAINING = not args["verification"]
RANDOM_ACTIONS = args["random"]
MODEL_PATH = args["model"] if args["verification"] else None
if args["verification"] and args["model"] is None:
    parser.error("-v requires -m")
VERIFICATION_START_IDX = args["verification_start"] \
    if args["verification_start"] else 0
VERIFICATION_END_IDX = args["verification_end"]

TICKERS = os.listdir(DATA_DIR)
DATA_DIRS = [os.path.join(DATA_DIR, curr_data) for curr_data in TICKERS]
OUTPUT_DIR = "output"
K = args["topbottomk"]


def main():
    config = read_config(config_filename=CONFIG_FILENAME,
                         output_dir=OUTPUT_DIR)
    data_reader = DataReader()
    X, y, date_begin, date_end = data_reader.read(
        DATA_DIRS, 
        start_idx=VERIFICATION_START_IDX, 
        end_idx=VERIFICATION_END_IDX)
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
        test_portfolio(X, y, config, MODEL_PATH, logger, K, RANDOM_ACTIONS)

    logger.set_dates(date_begin, date_end)
    logger.save()


if __name__ == "__main__":
    main()
