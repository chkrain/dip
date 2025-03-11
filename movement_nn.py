# movement_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from map import TILE_SIZE

class MovementNN(nn.Module):
    def __init__(self):
        super(MovementNN, self).__init__()
        self.fc1 = nn.Linear(7, 64)  
        self.fc2 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Forward pass через слои
        x = self.fc2(x)
        x = self.softmax(x)  # Преобразование в вероятности
        return x

class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def add_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

def choose_new_position(creature, creatures, density_map, radius=10):
    # Получаем допустимые координаты (с минимальной плотностью)
    valid_positions = []
    for x in range(creature.x - 1, creature.x + 2):  # Проверка на соседние клетки
        for y in range(creature.y - 1, creature.y + 2):
            if 0 <= x < density_map.shape[1] and 0 <= y < density_map.shape[0]:
                # Проверяем плотность на позиции (x, y)
                if get_density_at_position(x, y, creatures, radius) == 0:  # Плотность на клетке 0
                    valid_positions.append((x, y))
    if valid_positions:
        return valid_positions[np.random.randint(len(valid_positions))]
    else:
        return creature.x, creature.y  # Если нет свободных позиций, остаемся на месте

def get_density_at_position(x, y, creatures, radius=10):
    density = 0
    for other_creature in creatures:
        if np.linalg.norm(np.array([x, y]) - np.array([other_creature.x, other_creature.y])) < radius:
            density += 1
    return density


def get_state(self):
    tools = [int(self.tools["axe"]), int(self.tools["pickaxe"]), int(self.tools["shovel"])]
    evolution_stage = {"homo_habilis": 0, "homo_erectus": 1, "homo_sapiens": 2}[self.evolution_stage]
    state = np.array([self.x, self.y, self.speed, self.health, self.intelligence, evolution_stage, sum(tools)])
    return state

# Функция для выбора действия на основе состояния
def choose_action(state, model, creature, density_map, epsilon=0.5):
    # ε-жадный выбор: случайное действие с вероятностью epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    
    # Иначе — действие от модели
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action_probs = model(state_tensor).numpy().flatten()
    return np.argmax(action_probs)


def get_reward(creature, creatures, max_density=5, radius=10, house_location=None):
    reward = 0
    
    # Штраф за неподвижность
    if hasattr(creature, 'last_pos'):
        if creature.last_pos == (creature.x, creature.y):
            reward -= 2
    creature.last_pos = (creature.x, creature.y)
     
    # Бонус за движение к центру карты
    center_x = creature.screen.get_width() // (2 * TILE_SIZE)
    center_y = creature.screen.get_height() // (2 * TILE_SIZE)
    distance_to_center = abs(creature.x - center_x) + abs(creature.y - center_y)
    reward -= distance_to_center * 0.05  # Штрафуем отдаление от центра
     
    # Бонус за новые клетки
    if not hasattr(creature, 'visited'):
        creature.visited = set()
    if (creature.x, creature.y) not in creature.visited:
        reward += 1
        creature.visited.add((creature.x, creature.y))
    
    if creature.has_house:
        reward += 10
    
    # Штраф за перенаселение (плотность на позиции)
    density = get_density(creature, creatures, radius)
    if density > max_density:
        reward -= density * 2  # Дополнительный штраф за превышение плотности
    
    # Бонус за возвращение к дому, если таковой был построен
    if house_location:
        distance_to_house = np.linalg.norm(np.array([creature.x, creature.y]) - np.array(house_location))
        reward -= distance_to_house * 0.1  # Чем дальше от дома, тем меньше награда
    
    # Усиленный штраф за движение к краю
    edge_penalty = 0
    max_x = creature.screen.get_width() // TILE_SIZE - 1
    max_y = creature.screen.get_height() // TILE_SIZE - 1
    if creature.x < 5 or creature.x > max_x - 5:
        edge_penalty += 2
    if creature.y < 5 or creature.y > max_y - 5:
        edge_penalty += 2
    reward -= edge_penalty
    
    return reward


def get_density(creature, creatures, radius=10):
    # Считаем количество существ в радиусе 10 клеток от текущей позиции
    density = 0
    for other_creature in creatures:
        if np.linalg.norm(np.array([creature.x, creature.y]) - np.array([other_creature.x, other_creature.y])) < radius:
            density += 1
    return density

# Функция обучения
def train_model(model, optimizer, buffer, gamma=0.98):
    if len(buffer.states) == 0:
        return
    
    # Преобразование списка массивов в один массив NumPy
    states_np = np.array(buffer.states)  # Объединяем все состояния в один массив
    states = torch.tensor(states_np, dtype=torch.float32)
    
    actions = torch.tensor(buffer.actions, dtype=torch.int64)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
    
    # Очистка буфера
    buffer.states, buffer.actions, buffer.rewards = [], [], []
    
    # Расчёт дисконтированных наград
    discounted_rewards = []
    running_reward = 0
    for r in reversed(rewards):
        running_reward = r + gamma * running_reward
        discounted_rewards.insert(0, running_reward)
    
    # Нормализация наград
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
    
    # Расчёт потерь
    action_probs = model(states)
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    loss = -torch.mean(log_probs.squeeze() * discounted_rewards)
    
    # Обновление модели
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()