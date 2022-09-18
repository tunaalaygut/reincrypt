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
    logger = TrainingLogger(name=CONFIG["experiment_name"], 
                            config=CONFIG,
                            tickers=tickers,
                            output_dir=OUTPUT_DIR)

    data_reader = DataReader()

    model = Agent(1.0, CONFIG["epsilon_min"], CONFIG["max_iterations"], 
                  CONFIG["batch_size"], CONFIG["B"], CONFIG["C"], 
                  CONFIG["learning_rate"], CONFIG["penalty"])

    data_dirs = [os.path.join(DATA_DIR, curr_data) for curr_data in tickers]
 
    X, y = data_reader.read(data_dirs)

    model.set_data(X, y)

    model.train(CONFIG["width"], CONFIG["width"], CONFIG["num_actions"], 
                CONFIG["memory_size"], CONFIG["gamma"], CONFIG["learning_rate"],
                CONFIG["patch_size"], logger)

    logger.save()


def read_config():
    global CONFIG

    with open(f"config/{CONFIG_FILENAME}.json", "r+") as f:
        CONFIG = json.load(f)
    CONFIG["experiment_name"] = CONFIG_FILENAME

    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
