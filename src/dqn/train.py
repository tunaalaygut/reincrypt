import sys
sys.path.append("..")
import random
import numpy as np
from model import ViT 
import tensorflow as tf
from experince_replay import Memory
from rlogging.training_logger import TrainingLogger


gpu_config = tf.compat.v1.ConfigProto()
# only use required resource(memory)
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # restrict to 50%


class Agent:
    def __init__(self, config: dict):
        self.X = list()
        self.y = list()

        self.epsilon = config["epsilon_init"]
        self.epsilon_min = config["epsilon_min"]

        self.max_iterations = config["max_iterations"]
        self.batch_size = config["batch_size"]
        self.penalty = config["penalty"]
        self.B = config["B"]
        self.C = config["C"]

    def set_data(self, X, y, config: dict):
        self.X = X
        self.y = y

        print(f'X: # Currencies = {len(self.X)}, # Days: {len(self.X[0])}')
        print(f'y: # Currencies = {len(self.y)}, # Days: {len(self.y[0])}')

    def train(self, config: dict, logger: TrainingLogger):
        online_network = ViT(config)
        target_network = ViT(config)
        target_network.model.set_weights(online_network.model.get_weights())

        height, width = config["height"], config["width"]
        num_actions = config["num_actions"]
        memory_size = config["memory_size"]
        gamma = config["gamma"]

        prev_state = np.empty((1, height, width), dtype=np.float64)
        prev_action = np.empty((num_actions), dtype=np.int64)
        cur_state = np.empty((1, height, width), dtype=np.float64)
        cur_action = np.empty((num_actions), dtype=np.int64)
        next_state = np.empty((height, width), dtype=np.float64)

        memory = Memory(memory_size, width, height)  # memory buffer
        b = 1  # iteration counter

        while True:
            c = random.randrange(0, len(self.X))
            t = random.randrange(1, len(self.X[c]) - 1)

            cur_state = self.X[c][t]
            prev_state = self.X[c][t-1]
            next_state = self.X[c][t+1]

            if(self.randf(0, 1) <= self.epsilon):
                prev_action = self.get_random_action(num_actions)
            else:
                prev_action = online_network.q_value(prev_state, False)[1]

            if(self.randf(0, 1) <= self.epsilon):
                cur_action = self.get_random_action(num_actions)
            else:
                cur_action = online_network.q_value(cur_state, False)[1]

            L = self.y[c][t]  # Next day return
            reward = self.get_reward(prev_action, cur_action, L, self.penalty)
            
            memory.remember(cur_state, cur_action, reward, next_state)

            # TODO: Should we make epsilon get smaller faster?
            if(self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon * 0.999999

            if(b % self.B == 0 and len(memory.current_state) >= memory_size):
                S, A, Y = memory.get_batch(target_network, 
                                           self.batch_size, num_actions, gamma)
                
                loss = online_network.optimize_q(S, A, Y, self.batch_size)
                
                if(b % (100 * self.B) == 0):
                    print(f"#{b} -> Loss:{loss}")
                    logger.add_loss(float(loss))

            if(b % (self.C * self.B) == 0):
                online_network.model.save(
                    f"{logger.output_dir}/{logger.name}_model")
                target_network.model.set_weights(
                    online_network.model.get_weights())
                print("Updated target network weights.")

            b += 1

            if(b >= self.max_iterations):
                online_network.model.save(
                    f"{logger.output_dir}/{logger.name}_model")
                print('Training finished!')
                return 0

    def get_random_action(self,  num_actions):
        rand_rho = tf.random.uniform((1, num_actions))
        return tf.one_hot(tf.argmax(rand_rho, 1),
                          num_actions,
                          on_value=1,
                          off_value=0,
                          dtype=tf.int64)

    def get_reward(self, prev_action, cur_action, L, penalty):
        # 1,0,-1 is assined to pre_act, cur_act
        # for action long, neutral, short respectively
        pre_act = 1 - np.argmax(prev_action)
        cur_act = 1 - np.argmax(cur_action)
        return (cur_act * L) - penalty * abs(cur_act - pre_act)
    
    def randf(self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s
