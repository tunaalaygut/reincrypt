import os
import sys
from data_reader import DataReader
from train import Agent
from model import ViT 
sys.path.append("../logging")
from training_logger import TrainingLogger


# TODO: Implement arg parser instead of this
DATA_DIR = sys.argv[1]
EXPERIMENT_NAME = sys.argv[2]

num_actions = 3  # buy, hold, sell


# hyper parameters described in the paper
# max_iterations = 5000000       # maxmimum iteration number
max_iterations = 50000       # maxmimum iteration number
learning_rate = 0.001       # learning rate
epsilon_min = 0.1           # minimum epsilon

width = 18  # input matrix size
memory_size = 1000  # memory buffer capacity
B = 10  # parameter theta  update interval
C = 1000  # parameter theta^* update interval ( TargetQ )
gamma = 0.99  # discount factor
batch_size = 32            # batch size
# transaction penalty while training.  0.05 (%) for training, 0 for testing
penalty = 0.05
patch_size = 6
resized_image_size = 72

# initialize
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
model = Agent(1.0, 
              epsilon_min,
              max_iterations, 
              batch_size,
              B,
              C,
              learning_rate,
              penalty)

network = ViT(width, 
                 width,
                 num_actions,
                 learning_rate,
                 patch_size,
                 resized_image_size)

######## Test Model ###########

# # folder list for testing 
# folderlist = data_reader.get_filelist(  '../Sample_Testing/')
# sess, saver, state, isTrain, rho_eta = model.TestModel_ConstructGraph(width, 
#                                                                       width,
#                                                                       filter_size,
#                                                                       pool_size,
#                                                                       stride,
#                                                                       num_actions)



# for i in range (len( folderlist)):
#     print(folderlist[i])
   
#     filepathX = folderlist[i] + 'inputX.txt'
#     filepathY = folderlist[i] + 'inputY.txt' 

#     XData = data_reader.read_X(filepathX, width, width)
#     YData = data_reader.read_y(filepathY, len(XData), len(XData[0]))   

#     model.set_data(XData, YData)
#     print("\n\n")
#     model.Test_Neutralized_Portfolio(
#         network, 
#         width,
#         width,
#         num_actions)
#     print("\n\n")
#     model.Test_TopBottomK_Portfolio(
#         network,
#         width,
#         width,
#         num_actions,
#         0.2)



# Uncomment following to train

# Train Model
# folder path for training
data_dirs = [os.path.join(DATA_DIR, curr_data) for curr_data in os.listdir(DATA_DIR)] 
X, y = data_reader.read(data_dirs)

model.set_data(X, y)
model.train(width, 
            width,
            num_actions,
            memory_size,
            gamma,
            learning_rate,
            patch_size,
            resized_image_size,
            logger)
logger.save()