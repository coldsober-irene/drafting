import numpy as np

class RLAgent:
    def __init__(self, env, num_episodes, alpha=0.1, gamma=0.99):
        self.action_space = [1, 2, 3, 4]
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        
    def get_q_value(self, state, action):
        return self.q_table.setdefault(state, {}).setdefault(action, 0)

    def act(self, state):
        """Epsilon-greedy action selection."""
        epsilon = 0.1
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_space)
        q_values = {action: self.get_q_value(state, action) for action in self.action_space}
        return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q_value = max([self.get_q_value(next_state, a) for a in self.action_space])
        current_q_value = self.get_q_value(state, action)
        self.q_table[state][action] = current_q_value + self.alpha * (reward + self.gamma * max_next_q_value - current_q_value)
        
    def learn(self):
        rewards = []

        for _ in range(self.num_episodes):
            cumulative_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                cumulative_reward += reward
            rewards.append(cumulative_reward)

        return rewards

    def display_v_table(self):
        return {state: max(actions.values()) for state, actions in self.q_table.items()}

    def display_policy(self):
        return {state: max(actions, key=actions.get) for state, actions in self.q_table.items()}

    def visualize_v_table(self):
        v_table = self.display_v_table()
        shape = (self.env.world_height, self.env.world_width)
        
        v_matrix = np.full(shape, -1000)
        for state, value in v_table.items():
            agent_row, agent_col = state[:2]
            v_matrix[agent_row, agent_col] = max(v_matrix[agent_row, agent_col], value)

        for i in range(shape[0]):
            row_values = [f"{v_matrix[i, j]:.2f}" if v_matrix[i, j] != -1000 else "0.00" for j in range(shape[1])]
            print(' | '.join(row_values))

