import random

import numpy as np

from DQNAgent import DQNAgent

class DDQNAgent(DQNAgent):
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for state, action, reward, next_state, done, legal_moves in minibatch:
            target = reward
            if not done:
                next_q = self.model.predict(next_state)[0]
                if self.filter_invalid:
                    for move in range(self.action_size):
                        if move not in legal_moves:
                            next_q[move] = np.NINF

                target += self.gamma * self.target_model.predict(next_state)[0][np.argmax(next_q)]

            target_f = self.model.predict(state)
            target_f[0][action] = target
            X_train.append(state.reshape(16,))
            y_train.append(target_f.reshape(4,))

        loss = self.model.evaluate(np.array(X_train), np.array(y_train), batch_size=batch_size, verbose=0)
        self.model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=1, verbose=0)
        self.epsilon_func()

        return loss