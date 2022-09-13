from math import ceil
import tensorflow as tf
import numpy as np
import random
from model import ViT 
from experince_replay import Memory
from time import time  # for experimenting purposes

gpu_config = tf.compat.v1.ConfigProto()
# only use required resource(memory)
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # restrict to 50%


class Agent:
    def __init__(self,
                 epsilon_init,
                 epsilon_min,
                 max_iterations,
                 batch_size,
                 B,
                 C,
                 learning_rate,
                 penalty):
        self.X = list()
        self.y = list()

        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min

        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.B = B
        self.C = C

    def set_data(self, X, y):
        self.X = X
        self.y = y

        print(f'X: # Currencies = {len(self.X)}, # Days: {len(self.X[0])}')
        print(f'y: # Currencies = {len(self.y)}, # Days: {len(self.y[0])}')

    def train(self, height, width, filter_size, 
              pool_size, stride, num_actions, 
              memory_size, gamma, learning_rate,
              patch_size, resized_image_size, logger):
        online_network = ViT(height, width, filter_size, 
                             pool_size, stride, num_actions,
                             learning_rate, patch_size, resized_image_size)
        target_network = ViT(height, width, filter_size,
                             pool_size, stride, num_actions,
                             learning_rate, patch_size, resized_image_size)

        target_network.model.set_weights(online_network.model.get_weights())

        prev_state = np.empty((1, height, width), dtype=np.float64)
        prev_action = np.empty((num_actions), dtype=np.int32)
        cur_state = np.empty((1, height, width), dtype=np.float64)
        cur_action = np.empty((num_actions), dtype=np.int32)
        next_state = np.empty((height, width), dtype=np.float64)

        memory = Memory(memory_size, width, height)  # memory buffer
        b = 1  # iteration counter

        while True:
            c = random.randrange(0, len(self.X))
            t = random.randrange(1, len(self.X[c]) - 1)

            cur_state = self.X[c][t]
            prev_state = self.X[c][t-1]

            if(self.randf(0, 1) <= self.epsilon):
                prev_action = self.get_randaction(num_actions)
            else:
                prev_action = online_network.q_value(prev_state, False)[1]

            if(self.randf(0, 1) <= self.epsilon):
                cur_action = self.get_randaction(num_actions)
            else:
                cur_action = online_network.q_value(cur_state, False)[1]

            L = self.y[c][t]  # Next day return
            reward = self.get_reward(prev_action, cur_action, L, self.penalty)
            next_state = self.X[c][t+1]
            
            memory.remember(cur_state, cur_action, reward, next_state)

            # TODO: Should we make epsilon get smaller faster?
            if(self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon * 0.999999

            if(b % self.B == 0 and len(memory.current_state) >= memory_size):
                S, A, Y = memory.get_batch(target_network, False, 
                                           self.batch_size, num_actions, gamma)
                
                loss = online_network.optimize_q(S, A, Y, self.batch_size)
                
                if(b % (100 * self.B) == 0):
                    print(f"#{b} -> Loss:{loss}")
                    logger.add_loss(float(loss))

            if(b % (self.C * self.B) == 0):
                online_network.model.save(f"{logger.name}_model")
                target_network.model.set_weights(
                    online_network.model.get_weights())
                print("Updated target network weights.")

            b += 1

            if(b >= self.max_iterations):
                online_network.model.save(f"{logger.name}_model")
                print('Training finished!')
                return 0

    def get_randaction(self,  num_actions):
        rand_rho = tf.random.uniform((1, num_actions))
        return tf.one_hot(tf.argmax(rand_rho, 1),
                          num_actions,
                          on_value=1,
                          off_value=0,
                          dtype=tf.int32)

    # TODO: Make sure this works as intended.
    def get_reward(self, prev_action, cur_action, L, penalty):
        # 1,0,-1 is assined to pre_act, cur_act
        # for action long, neutral, short respectively
        pre_act = 1 - np.argmax(prev_action)
        cur_act = 1 - np.argmax(cur_action)
        return (cur_act * L) - penalty * abs(cur_act - pre_act)
    
    def randf(self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s

#### Testing functions ####

    def validate_Neutralized_Portfolio(self, network, DataX, DataY,
                                       NumAction, H, W):

        # list
        N = len(DataX)
        Days = len(DataX[0])
        curA = np.zeros((N, NumAction))

        # alpha
        preAlpha_n = np.zeros(N)
        curAlpha_n = np.zeros(N)
        posChange = 0

        # reward
        curR = np.zeros(N)
        avgDailyR = np.zeros(Days)

        # cumulative asset:  initialize cumAsset to 1.0
        cumAsset = 1

        for t in range(Days - 1):

            for c in range(N):

                # 1: choose action from current state
                curS = DataX[c][t]
                QAValues = network.q_value(curS.reshape(1, H, W), False)
                curA[c] = np.round(QAValues[1])

            # set Neutralized portfolio for day t
            curAlpha_n = self.get_NeutralizedPortfolio(curA,  N)

            for c in range(N):

                # 1: get daily reward sum
                curR[c] = np.round(curAlpha_n[c] * DataY[c][t], 8)
                avgDailyR[t] = np.round(avgDailyR[t] + curR[c], 8)

                # 2: pos change sum
                posChange = np.round(
                    posChange + abs(curAlpha_n[c] - preAlpha_n[c]), 8)
                preAlpha_n[c] = curAlpha_n[c]

        # calculate cumulative return
        for t in range(Days):
            cumAsset = round(cumAsset + (cumAsset * avgDailyR[t] * 0.01), 8)

        print('cumAsset ',  cumAsset)
        return N, posChange, cumAsset

    def validate_TopBottomK_Portfolio(self, network, DataX, DataY, 
                                      NumAction, H, W, K):
        N = len(DataX)
        Days = len(DataX[0])

        print(N, Days)

        # alpha
        preAlpha_s = np.zeros(N)
        curAlpha_s = np.zeros(N)
        posChange = 0

        # reward
        curR = np.zeros(N)
        avgDailyR = np.zeros(Days)

        # cumulative asset: initialize curAsset to 1.0
        cumAsset = 1

        # action value for Signals and Threshold for Top/Bottom K
        curActValue = np.zeros((N, NumAction))
        LongSignals = np.zeros(N)

        UprTH = 0
        LwrTH = 0

        for t in range(Days - 1):

            for c in range(N):

                # 1: choose action from current state
                curS = DataX[c][t]
                QAValues = network.q_value(curS.reshape(1, H, W), False)
                curActValue[c] = np.round(QAValues[0], 4)
                LongSignals[c] = curActValue[c][0] - curActValue[c][2]

            # set Top/Bottom portfolio for day t
            UprTH, LwrTH = self.givenLongSignals_getKTH(LongSignals, K)
            curAlpha_s = self.get_TopBottomPortfolio(
                UprTH, LwrTH, LongSignals, N)

            for c in range(N):

                # 1: get daily reward sum
                curR[c] = np.round(curAlpha_s[c] * DataY[c][t], 8)
                avgDailyR[t] = np.round(avgDailyR[t] + curR[c], 8)

                # 2: pos change sum
                posChange = np.round(
                    posChange + abs(curAlpha_s[c] - preAlpha_s[c]), 8)
                preAlpha_s[c] = curAlpha_s[c]

        # calculate cumulative return
        for t in range(Days):
            cumAsset = round(cumAsset + (cumAsset * avgDailyR[t] * 0.01), 8)

        print('cumAsset ',  cumAsset)
        return N, posChange, cumAsset

    def TestModel_ConstructGraph(self, H, W, FSize, PSize, PStride,  NumAction):
        state = tf.compat.v1.placeholder(tf.float32, [None, H, W])
        isTrain = tf.compat.v1.placeholder(tf.bool, [])

        # construct Graph
        C = ViT(H, W, FSize, PSize, PStride, NumAction)
        rho_eta = C.q_value(state, isTrain)

        sess = tf.compat.v1.Session(config=gpu_config)
        saver = tf.train.Saver()

        return sess, saver, state, isTrain, rho_eta

    def Test_TopBottomK_Portfolio(self, network, H, W, NumAction, TopK):
        network.model = tf.keras.models.load_model('DeepQ')

        Outcome = self.validate_TopBottomK_Portfolio(
            network, self.X, self.y, NumAction, H, W, TopK)

        print('NumComp#: ',  Outcome[0],  'Transactions: ',
              Outcome[1]/2, 'cumulative asset', Outcome[2])
        self.writeResult_daily('TestResult.txt', Outcome,  len(self.X[0]) - 1)

    def Test_Neutralized_Portfolio(self,
                                   network,
                                   H,
                                   W,
                                   NumAction):
        network.model = tf.keras.models.load_model('DeepQ')
        outcome = self.validate_Neutralized_Portfolio(
            network, self.X, self.y, NumAction, H, W)

        print(f"outcome = {outcome}")

        print('NumComp#: ',  outcome[0],  'Transactions: ',
              outcome[1]/2, 'cumulative asset', outcome[2])
        self.writeResult_daily('TestResult.txt', outcome, len(self.X[0]) - 1)

    def get_NeutralizedPortfolio(self, curA, N):
        alpha = np.zeros(N)
        avg = 0

        # get average
        for c in range(N):
            alpha[c] = 1 - np.argmax(curA[c])
            avg = avg + alpha[c]

        avg = np.round(avg / N, 4)

        # set alpha
        sum_a = 0
        for c in range(N):
            alpha[c] = np.round(alpha[c] - avg, 4)
            sum_a = np.round(sum_a + abs(alpha[c]), 4)

        # set alpha
        if sum_a == 0:
            return alpha

        for c in range(N):
            alpha[c] = np.round(alpha[c] / sum_a, 8)

        return alpha

    def givenLongSignals_getKTH(self, LongSignals, K):
        Num = ceil(len(LongSignals) * K)
        SortedLongS = np.sort(LongSignals)

        return SortedLongS[len(LongSignals) - Num], SortedLongS[Num-1]

    def get_TopBottomPortfolio(self, UprTH, LwrTH, LongSignals, N):
        alpha = np.zeros(N)
        sum_a = 0

        for c in range(N):
            if LongSignals[c] >= UprTH:
                alpha[c] = 1
                sum_a = sum_a + 1
            elif LongSignals[c] <= LwrTH:
                alpha[c] = -1
                sum_a = sum_a+1
            else:
                alpha[c] = 0

        if sum_a == 0:
            return alpha

        for c in range(N):
            alpha[c] = np.round(alpha[c] / float(sum_a), 8)

        return alpha

    def writeResult_daily(self,  filename,  outcome, numDays):
        f = open(filename, 'a')

        f.write('Comp#,' + str(outcome[0]) + ',')
        f.write('Days#' + str(numDays-1) + ',')
        f.write('TR#,' + str(round(outcome[1]/2, 4)) + ',')
        f.write('FinalAsset,' + str(round(outcome[2], 4)))

        f.write("\n")
        f.close()
