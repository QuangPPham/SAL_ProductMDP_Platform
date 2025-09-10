import numpy as np

class QLearning:
    def __init__(self, env, learning_rate: float = 0.5, timesteps: int = 100, exploration: float = 0.1, iteration: int = 1000000):
        self.gamma = env.gamma
        self.alpha = learning_rate
        self.epsilon = exploration
        self.T = timesteps
        try:
            self.S = env.SX
        except AttributeError:
            self.S = env.S
        self.A = env.A
        self.mdp_sim = env.sample
        self.iter = iteration

        self.s = np.random.randint(self.S)
        self.t = 0

        self.V = np.zeros(self.S)
        self.Q = np.zeros((self.A, self.S))
        self.policy = np.zeros(self.S, dtype=np.int32)

        self.truncated = False

    def simulate(self):
        s = self.s

        # epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            a = np.random.randint(0, self.A)
        else:
            a = np.argmax(self.Q[:, s])

        try:
            _, _, s, r, truncated = self.mdp_sim(sx=s, a=a)
        except ValueError:
            s, r, truncated = self.mdp_sim(s=s, a=a)

        return s, a, r, truncated

    def step(self):
        # Restart "episode"
        if self.t == self.T or self.truncated:
            self.s = np.random.randint(self.S)
            self.t = 0
            self.truncated = False
        new_s, a, r, self.truncated = self.simulate()
        self.t += 1
        q_max = np.max(self.Q[:, new_s])
        self.Q[a, self.s] += self.alpha * (r + self.gamma*q_max - self.Q[a, self.s])
        self.s = new_s
    
    def update_value(self):
        for s in range(self.S):
            greedy_Q = np.max(self.Q[:, s])
            self.policy[s] = np.argmax(self.Q[:, s])
            self.V[s] = ((1-self.epsilon)*greedy_Q + (self.epsilon/self.A)*(np.sum(self.Q[:, s])))

    def run(self):
        for i in range(self.iter):
            self.step()
        self.update_value()
        return self.V, self.policy