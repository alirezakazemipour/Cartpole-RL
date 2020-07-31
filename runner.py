import gym


class Worker:
    def __init__(self, n, env_name):
        self.n = n
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self._state = None
        self.ep_r = 0
        self.running_reward = 0
        self.reset()

    def __str__(self):
        return str(self.n)

    @property
    def state(self):
        return self._state

    def render(self):
        self.env.render()

    def reset(self):
        self._state = self.env.reset()
        self.running_reward = 0.99 * self.running_reward + 0.01 * self.ep_r
        self.ep_r = 0

    def step(self, action):
        next_state, r, d, _ = self.env.step(action)
        self._state = next_state
        self.ep_r += r
        if d:
            self.reset()
        return dict({"next_state": next_state, "reward": r, "done": d})
