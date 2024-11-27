import numpy as np


class Environment:
    def __init__(self, CTR):
        self.CTR = CTR
        self.n_states, self.n_actions = self.CTR.shape
        self.current_state = None

    def observe(self):
        self.current_state = np.random.randint(self.n_states)
        return self.current_state

    def step(self, action):
        p = self.CTR[self.current_state, action]

        if np.random.rand() < p:
            return 1

        return 0