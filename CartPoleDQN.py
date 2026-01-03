import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

env = gym.make("CartPole-v1")
t = 0
replay = []
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
device = "cuda" if torch.cuda.is_available() else "cpu"
main = MLP(5, [64, 32], 1).to(device)#can adjust hidden sizes
target = MLP(5, [64, 32], 1).to(device)
target.load_state_dict(main.state_dict())
optimizer = torch.optim.Adam(main.parameters())
for j in range(100):# can adjust episodes
    episode_over = False
    old_observation, info = env.reset(seed=42)
    score = 0
    while not episode_over:
        o = [0, 1]
        e = np.random.rand()
        a = []
        for i in o:
            a.append(main(torch.tensor(np.append(old_observation, i), dtype=torch.float32).to(device)).item())
        if e < 0.1: #can adjust epsilon
            action = env.action_space.sample()
        else:
            if(a[0] > a[1]):
                action = 0
            else:
                action = 1
        new_observation, reward, terminated, truncated, info = env.step(action)
        replay.append((old_observation, action, reward, new_observation, terminated))
        score += reward
        if (len(replay) > 1000): #can adjust replay buffer size
            replay.pop(0)
        batch = np.random.choice(len(replay), min(32, len(replay)), replace=False)
        for i in batch:
            obs, act, rew, new_obs, term = replay[i]
            obs_tensor = torch.tensor(np.append(obs, act), dtype=torch.float32).to(device)
            if term:
                target_q = rew
            else:
                new_a = []
                for k in o:
                    new_a.append(target(torch.tensor(np.append(new_obs, k), dtype=torch.float32).to(device)).item())
                if(new_a[0] > new_a[1]):
                    max_next_q = new_a[0]
                else:
                    max_next_q = new_a[1]
                target_q = rew + 0.99 * max_next_q #can adjust discount factor
            predicted_q = main(obs_tensor)
            loss = nn.MSELoss()(predicted_q, torch.tensor([target_q], dtype=torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        old_observation = new_observation
        t += 1
        if(t % 100 == 0): #can adjust target network update frequency
            target.load_state_dict(main.state_dict())
        episode_over = terminated or truncated
    print(score)

env.close()
