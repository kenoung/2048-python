import random

import numpy as np

from DQNAgent import DQNAgent


class DDQNAgent(DQNAgent):
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

            target_f = self.model.predict(state)
            target_f[0][action] = target
            X_train.append(state.reshape(16,))
            y_train.append(target_f.reshape(4,))

        self.model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
