import gym


class Worker:
    def __init__(self, id, env_name):
        self.id = id
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self._state = None
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        self._state = self.env.reset()

    def step(self, conn):
        while True:
            conn.send(self._state)
            action = conn.recv()
            next_state, r, d, _ = self.env.step(action)
            # self.render()
            self._state = next_state
            if d:
                self.reset()
            conn.send((next_state, r, d, _))
