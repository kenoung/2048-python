import random

import numpy as np

from DQNAgent import DQNAgent
from SumTree import SumTree

class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.maxlen = capacity

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( data )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.maxlen

class DDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.01):
        super(DDQNAgent, self).__init__(state_size, action_size, gamma=0.95, lr=0.000001)
        self.memory = Memory(10000)

    def remember(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

        target_f = self.target_model.predict(state)
        predicted = target_f[0][action]

        self.memory.add(abs(target - predicted), (state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
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
