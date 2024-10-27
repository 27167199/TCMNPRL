import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.gamma = torch.tensor(0.99)#0.99

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


def choose_action(state, model):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        probabilities = model(state_tensor).numpy()
    action = np.random.choice(len(probabilities.ravel()), p=probabilities.ravel())

    return action


def train(model, optimizer, episodes, herbs, evaluate_fitness, herb_score_dict, herb_pair_from_data):
    for episode in range(episodes):
        state = np.zeros(len(herbs))
        selected_herbs = []
        rewards = []
        log_probs = []

        for step in range(10):  # 确保最多选择10种草药
            action = choose_action(state.reshape(1, -1), model)
            if state[action] == 0:  # 避免重复选择同一种草药
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                action_probs = model(state_tensor)
                action_taken = torch.tensor([action], dtype=torch.long)
                log_prob = torch.log(action_probs.gather(1, action_taken.unsqueeze(0)))
                log_probs.append(log_prob)

                selected_herbs.append(herbs[action])
                state[action] = 1  # 更新状态
                reward = evaluate_fitness(selected_herbs, herb_score_dict, herb_pair_from_data)
                rewards.append(reward)

        # 计算每个时间步的累计回报 Gt
        G_list = []
        G_t = 0
        for reward in reversed(rewards):
            G_t = model.gamma * G_t + reward
            G_list.insert(0, G_t)  # 反向插入
        G_tensor = torch.tensor(G_list, dtype=torch.float32)

        # 更新模型
        optimizer.zero_grad()
        loss = 0
        for g, log_prob in zip(G_tensor, log_probs):
            loss -= log_prob * g
        loss.backward()
        optimizer.step()

def generate_prescriptions(model, herbs, num_prescriptions):
    prescriptions = []
    for _ in range(num_prescriptions):
        state = np.zeros(len(herbs))
        selected_herbs = []
        for step in range(10):  # 确保最多选择10种草药
            action = choose_action(state.reshape(1, -1), model)
            if state[action] == 0:  # 避免重复选择同一种草药
                selected_herbs.append(herbs[action])
                state[action] = 1  # 更新状态

        prescriptions.append(selected_herbs)
    return prescriptions
