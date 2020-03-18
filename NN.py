import numpy as np
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import Adam

# A really simple NN that with a relu activation function and Adam Optimizer


class NN:

    def __init__(self, alpha):
        self.model = Sequential()
        # 8 continuous state space and 4 discrete action space
        self.model.add(Dense(128, input_shape=(8,)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(Dense(4))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=alpha))

    def train(self, x, y, batch_size):
        return self.model.fit(
            x, y,
            batch_size=batch_size,
            verbose=0
        )

    def predict(self, state):
        return self.model.predict(state)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
