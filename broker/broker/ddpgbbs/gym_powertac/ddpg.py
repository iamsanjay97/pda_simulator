import sys
sys.path.insert(1, './gym_powertac')

from powertac_wm import PowerTAC_WM
import gym
import datetime
import json
import numpy as np

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

model_storage_path = "../ddpg_v1.0" 

class DDPG(object):

    BUFFER_SIZE = 250000
    BATCH_SIZE = 32
    EPOCHS = 5
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    '''
    Action : [
               limitprice1 belongs to R
               limitprice2 belongs to R
             ] (TWO output nuerons)
    '''
    action_dim = 2

    '''
    State : [
              Proximity (1)
              Balancing_Price (1)
              Required_Quantity (1)
            ] (total 26 input nuerons)
    '''
    state_dim = 3

    np.random.seed(1337)
    EXPLORE = 100000.0

    step = 0
    epsilon = 1

    ou = OU()       #Ornstein-Uhlenbeck Process

    def __init__(self):

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    #Create replay buffer

        #Now load the weight
        print("Now we load the weight")
        try:
            self.actor.model.load_weights(model_storage_path + "/actormodel.h5")
            self.critic.model.load_weights(model_storage_path + "/criticmodel.h5")
            self.actor.target_model.load_weights(model_storage_path + "/actortargetmodel.h5")
            self.critic.target_model.load_weights(model_storage_path + "/critictargetmodel.h5")
            print("Weights loaded successfully")
        except Exception as e:
            print(e)
            print("Cannot find the weights")


    def choose_Action(self,states):

        actions = list()

        for state in states:

            try:

                s_t = np.array(state.form_network_input())

                # self.epsilon -= 1.0 / self.EXPLORE
                # a_t = np.zeros([self.action_dim])
                # noise_t = np.zeros([self.action_dim])

                a_t_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0].tolist()  # outputs a list of two limitprices
                # noise_t[0] = max(self.epsilon, 0) * self.ou.function(a_t_original[0],  0.0 , 0.60, 0.30)  # decide theta, sigma and mu for limitprice

                # a_t[0] = a_t_original[0] + noise_t[0]
                # a_t[1] = a_t_original[1] + noise_t[1]

                # print(a_t_original)
                actions.append(list(a_t_original))

            except Exception as e:
                print(e)

        return actions


    def add_to_reply_buffer(self, exeperiences):

        for exeperience in exeperiences:
            self.buff.add(exeperience)      #Add replay buffer


    def train_ddpg_network(self):

        for epoch in range(self.EPOCHS):

            print("Epoch ", (epoch+1))
            print("-"*12)
            loss = 0

            # for exeperience in exeperiences:
            #     self.buff.add(exeperience)      #Add replay buffer

            #Do the batch update
            batch = self.buff.getBatch(self.BATCH_SIZE)
            states = np.asarray([np.array(e[0].form_network_input()) for e in batch])
            actions = np.asarray([np.array([e[1]['key'], e[1]['value']]) for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([np.array(e[3].form_network_input()) for e in batch])
            terminals = np.asanyarray([e[4] for e in batch])

            y_t = np.asarray([np.array([e[1]['key'], e[1]['value']]) for e in batch])

            # print("States", states)
            # print("Actions", actions)
            # print("Rewards", rewards)
            # print("New_States", new_states)
            # print("Terminal", terminals)
            # print("Y_t", y_t)

            target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
            # print("Target_Q_Values", target_q_values)

            for k in range(len(batch)):
                if terminals[k] == 1:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + self.GAMMA*target_q_values[k]

            # print("Bellman Rewards", y_t)
            loss += self.critic.model.train_on_batch([states,actions], y_t)
            print("Loss", loss)
            a_for_grad = self.actor.model.predict(states)      # This may not be required, a_for_grad should be replaced by actions ##### Check PENDING #####
            # print("a_for_grad", a_for_grad)
            grads = self.critic.gradients(states, a_for_grad)       # a_for_grad is replaced by actions ##### Check PENDING #####   shape ERROR 
            # print("grads", grads)
            self.actor.train(states, grads)
            self.actor.target_train()
            self.critic.target_train()

            self.step += 1

        print("Training Completed !!!")


    def save_models(self):

        timestamp = int(datetime.datetime.now().timestamp())

        self.actor.model.save_weights(model_storage_path + "/actormodel.h5", overwrite=True)
        with open(model_storage_path + "/actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights(model_storage_path + "/criticmodel.h5", overwrite=True)
        with open(model_storage_path + "/criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)

        self.actor.target_model.save_weights(model_storage_path + "/actortargetmodel.h5", overwrite=True)
        with open(model_storage_path + "/actormodeltarget.json", "w") as outfile:
            json.dump(self.actor.target_model.to_json(), outfile)

        self.critic.target_model.save_weights(model_storage_path + "/critictargetmodel.h5", overwrite=True)
        with open(model_storage_path + "/criticmodeltarget.json", "w") as outfile:
            json.dump(self.critic.target_model.to_json(), outfile)

        ''' Save both models as backup '''
        self.actor.model.save_weights(model_storage_path + "/actormodel_" + str(timestamp) + ".h5", overwrite=True)
        with open(model_storage_path + "/actormodel_" + str(timestamp) + ".json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)

        self.critic.model.save_weights(model_storage_path + "/criticmodel_" + str(timestamp) + ".h5", overwrite=True)
        with open(model_storage_path + "/criticmodel_" + str(timestamp) + ".json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)

        print("Models Saved Successfully !!!")


    def test(self):

        proximity = 1               # One-Hot Encoded Proximity
        quantity = 0.4              # Normalized values

        state = PowerTAC_WM(proximity, quantity)
        next_state = PowerTAC_WM(proximity-1, quantity-0.2)

        print(self.choose_Action([state]))
        # self.train_ddpg_network()


# config = tf.ConfigProto(
#     device_count={'GPU': 1},
#     intra_op_parallelism_threads=1,
#     allow_soft_placement=True
# )

# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6

# session = tf.Session(config=config)

# ddpg = DDPG(session)
ddpg = DDPG() 
ddpg.test()
        
# states = list()
# for item in data:
#     if item is not None:

#         state = PowerTAC_WM(item['state']['proximity'], item['state']['balancingPrice'], item['state']['quantity'])
#         states.append(state)

# actions = ddpg.choose_Action(states)