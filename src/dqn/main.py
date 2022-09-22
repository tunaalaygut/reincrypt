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

    data_reader = DataReader()

    model = Agent(epsilon_init=CONFIG["epsilon_init"],
                  epsilon_min=CONFIG["epsilon_min"], 
                  max_iterations=CONFIG["max_iterations"], 
                  batch_size=CONFIG["batch_size"], B=CONFIG["B"], C=CONFIG["C"], 
                  penalty=CONFIG["penalty"])

    data_dirs = [os.path.join(DATA_DIR, curr_data) for curr_data in tickers]
 
    X, y = data_reader.read(data_dirs)

    model.set_data(X, y)

    model.train(height=CONFIG["height"], width=CONFIG["width"], 
                num_actions=CONFIG["num_actions"], 
                memory_size=CONFIG["memory_size"], gamma=CONFIG["gamma"], 
                learning_rate=CONFIG["learning_rate"],
                patch_size=CONFIG["patch_size"], 
                projection_dim=CONFIG["projection_dim"], 
                mlp_head_units=CONFIG["mlp_head_units"], 
                transformer_units=CONFIG["transformer_units"], 
                num_heads=CONFIG["num_heads"], 
                transformer_layers=CONFIG["transformer_layers"], logger=logger)

    logger.save()


def read_config():
    global CONFIG

    with open(f"config/{CONFIG_FILENAME}.json", "r+") as f:
        CONFIG = json.load(f)
    CONFIG["experiment_name"] = CONFIG_FILENAME

    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
