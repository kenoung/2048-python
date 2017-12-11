import random
from collections import deque

import numpy as np

from DQNAgent import DQNAgent
from SumTree import SumTree

class NormalMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=1000)

    def sample(self, n):
        return random.sample(self.memory, n)

class PrioritizedMemory:
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
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.01, filter_invalid=False, xp_replay=False):
        super(DDQNAgent, self).__init__(state_size, action_size, gamma=gamma, lr=lr, filter_invalid=filter_invalid)
        self.xp_replay = xp_replay
        if xp_replay:
            self.memory = PrioritizedMemory(10000)
        else:
            self.memory = NormalMemory(10000)

    def remember(self, state, action, reward, next_state, done, available_moves=[]):
        if self.xp_replay:
            target = reward
            if not done:
                target += self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

            target_f = self.target_model.predict(state)
            predicted = target_f[0][action]

            self.memory.add(abs(target - predicted), (state, action, reward, next_state, done, available_moves))
        else:
            super(DDQNAgent, self).remember(state, action, reward, next_state, done, available_moves)

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
