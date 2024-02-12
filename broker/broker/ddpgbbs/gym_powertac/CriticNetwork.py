import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 30

class CriticNetwork(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        S = tf.keras.layers.Input(shape=[state_size])
        A = tf.keras.layers.Input(shape=[action_dim],name='action2')
        w1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = tf.keras.layers.add([h1,a1])
        h3 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = tf.keras.layers.Dense(action_dim,activation='linear')(h3)
        model = tf.keras.Model(inputs=[S,A],outputs=V)
        adam = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, A, S
