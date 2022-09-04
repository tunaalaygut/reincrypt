import os
from data_reader import DataReader
from train import Agent
from model import ConvNN 


DATA_DIR = "../../output"

filter_size = 5
pool_size = 2
stride = 2
num_actions = 3


# hyper parameters described in the paper
# max_iterations = 5000000       # maxmimum iteration number
max_iterations = 50000       # maxmimum iteration number
learning_rate = 0.00001       # learning rate
epsilon_min = 0.1           # minimum epsilon

width = 18  # input matrix size
memory_size = 1000  # memory buffer capacity
B = 10  # parameter theta  update interval
C = 1000  # parameter theta^* update interval ( TargetQ )
gamma = 0.99  # discount factor
batch_size = 32            # batch size
# transaction panalty while training.  0.05 (%) for training, 0 for testing
penalty = 0.05

# initialize
data_reader = DataReader()
model = Agent(1.0, 
              epsilon_min,
              max_iterations, 
              batch_size,
              B,
              C,
              learning_rate,
              penalty)

network = ConvNN(width, 
                 width,
                 filter_size,
                 pool_size,
                 stride,
                 num_actions,
                 learning_rate)
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
            filter_size,
            pool_size,
            stride,
            num_actions,
            memory_size,
            gamma,
            learning_rate)
