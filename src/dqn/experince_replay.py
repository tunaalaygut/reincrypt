import random 
import numpy as np
import tensorflow as tf

class Memory:
    def __init__(self, memory_size, width, height) :
        self.memory_size = memory_size
        self.W = width
        self.H = height

        self.current_state = list()
        self.current_action = list()
        self.reward = list()
        self.next_state = list()

    def remember(self, current_state, current_action, reward, next_state):
        # Save current experience
        self.current_state.append(current_state)
        self.current_action.append(current_action)
        self.reward.append(reward)
        self.next_state.append(next_state)

        # Delete oldest experience
        if(len(self.current_state) > self.memory_size):
            del self.current_state[0]
            del self.current_action[0]
            del self.reward[0]
            del self.next_state[0]

    def get_batch(self, target_network, is_training, batch_size, num_actions, gamma):
        cur_states = np.zeros((batch_size, self.H, self.W))
        cur_actions = np.zeros((batch_size, num_actions))
        targets = np.zeros((batch_size, num_actions))

        # Get `batch_size` random index from a memory sized list
        rand_idxs = random.sample(range(len(self.current_state)), batch_size)
        
        for k in range(batch_size):
            input_kth = self.current_state[rand_idxs[k]]
            action_kth = self.current_action[rand_idxs[k]]

            q_act_values = target_network.q_value(self.next_state[rand_idxs[k]], 
                                                  is_training)
            next_qs = q_act_values[0]

            target_kth = np.zeros(num_actions)
            target_kth[np.argmax(action_kth)] = self.reward[rand_idxs[k]] + gamma * next_qs[0][tf.argmax(next_qs[0])]

            cur_states[k] = input_kth
            cur_actions[k] = action_kth
            targets[k] = target_kth
            
            if np.isnan(target_kth).any():
                print("Stop")
    
        return cur_states, cur_actions, targets
