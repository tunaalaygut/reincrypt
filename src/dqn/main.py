import os
import sys
from data_reader import DataReader
from train import Agent
sys.path.append("../logging")
from training_logger import TrainingLogger


# TODO: Implement arg parser instead of this
DATA_DIR = sys.argv[1]
EXPERIMENT_NAME = sys.argv[2]


def main():
    num_actions = 3  # buy, hold, sell
    max_iterations = 50000
    learning_rate = 0.001
    epsilon_min = 0.1
    width = 18  
    memory_size = 1000 
    B = 10  # Parameter theta update interval (online network)
    C = 1000  # Parameter theta^* update interval (target network)
    gamma = 0.99  # Discount factor
    batch_size = 32 
    penalty = 0.05
    patch_size = 6
    resized_image_size = 72

    # initialize logger
    hyperparameters = {
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "epsilon_min": epsilon_min,
        "width": width,
        "memory_size": memory_size,
        "B": B,
        "C": C,
        "gamma": gamma,
        "batch_size": batch_size,
        "penalty": penalty
    }

    logger = TrainingLogger(name=EXPERIMENT_NAME, 
                            hyperparameters=hyperparameters,
                            tickers=os.listdir(DATA_DIR))

    data_reader = DataReader()
    model = Agent(1.0, epsilon_min, max_iterations, batch_size, B, C, 
                  learning_rate, penalty)

    data_dirs = [os.path.join(DATA_DIR,
                            curr_data) for curr_data in os.listdir(DATA_DIR)] 
    X, y = data_reader.read(data_dirs)

    model.set_data(X, y)
    model.train(width, width, num_actions, memory_size, gamma, learning_rate,
                patch_size, resized_image_size, logger)

    logger.save()


if __name__ == "__main__":
    main()
