import random
from collections import deque
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

DECAYING_EPSILON = "de"
FIXED_VALUE_EPSILON = "fve"


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = gamma    # discount rate
        self.learning_rate = lr
        self.model = self._build_model()
        self.target_model = self._build_model()

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        self.epsilon_fixed = 0.1
        self.epsilon_func = None
        self.epsilon_map = {
            DECAYING_EPSILON: self.decaying_epsilon,
            FIXED_VALUE_EPSILON: self.fixed_value_epsilon,
        }

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate, clipvalue=1))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)

        act_values = self.model.predict(state)
        for move in range(self.action_size):
            if move not in legal_moves:
                act_values[0][move] = np.NINF

        return np.argmax(act_values[0])  # returns action

    def act_policy(self, state, legal_moves):
        act_values = self.model.predict(state)
        for move in range(self.action_size):
            if move not in legal_moves:
                act_values[0][move] = np.NINF

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            X_train.append(state.reshape(16,))
            y_train.append(target_f.reshape(4,))

        self.model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=1, verbose=0)
        self.epsilon_func()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


    ##################
    ## EPSILON FUNC ##
    ##################

    def set_epsilon(self, epsilon_func_str):
        self.epsilon_func = self.epsilon_map[epsilon_func_str]
        self.epsilon_func()

    def decaying_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def fixed_value_epsilon(self):
        self.epsilon = self.epsilon_fixed
