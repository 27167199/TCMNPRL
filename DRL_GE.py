import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as rd


def innit_formula_seed(herb_score_dict, row_num, col_num_list):
    formula_herb_list = []
    herb_score_dict = sorted(herb_score_dict.items(), key=lambda x: x[1], reverse=False)

    for i in range(row_num):
        formula_herb_list_seed = []
        while len(formula_herb_list_seed) < col_num_list[i]:
            benchmark_num = 0
            r = rd.random()
            for (k, v) in herb_score_dict:
                if r < benchmark_num + float(v) and k not in formula_herb_list_seed:
                    formula_herb_list_seed.append(k)
                    break
                benchmark_num += float(v)
        formula_herb_list_seed = sorted(formula_herb_list_seed)
        formula_herb_list.append(formula_herb_list_seed)
    return formula_herb_list

def generate_formula_list(rows_num):#生成1-15的方剂列表
    rows_num_list = []
    for i in range(rows_num):
        num = rd.randint(5, 15)#从5味药到15味药
        rows_num_list.append(num)
    return rows_num_list

class DRLHerbRecommendation:
    def __init__(self, herb_data, herb_scores):
        self.herb_data = herb_data
        self.herb_scores = herb_scores
        self.population = self.init_population()
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def init_population(self):
        # 初始化种群使用 innit_formula_seed 方法
        row_num = 100  # 种群大小为100
        col_num_list = generate_formula_list(row_num)
        #col_num_list = [rd.randint(5, 15) for _ in range(row_num)]  # 每个个体包含随机数量的草药
        herb_score_dict = {herb: score for herb, score in zip(self.herb_data, self.herb_scores)}

        initial_population = innit_formula_seed(herb_score_dict, row_num, col_num_list)
        return initial_population

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(len(self.herb_data), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.herb_data)),
            nn.Softmax(dim=1)
        )
        return model

    def get_state(self):
        states = np.array([self.individual_to_state(ind) for ind in self.population])
        return torch.tensor(states, dtype=torch.float32)

    def individual_to_state(self, individual):
        state = np.zeros(len(self.herb_data))
        for herb in individual:
            state[self.herb_data.index(herb)] = 1
        return state

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            probabilities = self.model(state_tensor).numpy()
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def fitness(self, individual):
        return calculate_fitness(individual)

    def train_model(self, states, actions, rewards):
        self.optimizer.zero_grad()
        outputs = self.model(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        loss = self.loss_fn(outputs, actions_tensor)
        loss.backward()
        self.optimizer.step()

    def evolve(self, generations):
        for _ in range(generations):
            states = self.get_state()
            actions = [self.choose_action(state) for state in states]
            offspring = self.apply_actions(actions)
            rewards = [self.fitness(individual) for individual in offspring]
            self.train_model(states, actions, rewards)
            self.population = self.elitism(offspring)

    def apply_actions(self, actions):
        offspring = []
        for i, action in enumerate(actions):
            new_individual = self.mutate_individual(self.population[i], action)
            offspring.append(new_individual)
        return offspring

    def mutate_individual(self, individual, action):
        new_individual = individual.copy()
        new_herb = self.herb_data[action]
        replace_index = np.random.randint(0, len(individual))
        new_individual[replace_index] = new_herb
        return new_individual

    def elitism(self, offspring):
        combined = self.population + offspring
        combined.sort(key=self.fitness, reverse=True)
        return combined[:len(self.population)]


# 使用示例
def load_herb_data():
    # 假设这是加载草药数据的函数
    return ['herb1', 'herb2', 'herb3', ..., 'herbN']


def calculate_fitness(individual):
    # 假设这是计算适应度的函数
    return random.uniform(0, 1)

'''
herb_data = load_herb_data()  # 加载草药数据
herb_scores = [random.random() for _ in herb_data]  # 假设这是草药的分数

drl_herb_recommendation = DRLHerbRecommendation(herb_data, herb_scores)
drl_herb_recommendation.evolve(100)  # 运行100代
best_formula = drl_herb_recommendation.population[0]  # 获取最优方剂
'''