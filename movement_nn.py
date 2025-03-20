# movement_nn.py
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from map import TILE_SIZE

class MovementNN(nn.Module):
    def __init__(self):
        super(MovementNN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[Ошибка] NaN или Inf во входных данных:", x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)

        x = torch.nan_to_num(torch.relu(self.fc1(x)), nan=0.0, posinf=1e5, neginf=-1e5)
        x = torch.nan_to_num(torch.relu(self.fc2(x)), nan=0.0, posinf=1e5, neginf=-1e5)
        logits = torch.nan_to_num(self.fc3(x), nan=0.0, posinf=1e5, neginf=-1e5)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[Ошибка] NaN или Inf перед Softmax:", logits)
            logits = torch.zeros_like(logits)  # Предотвращаем проблему

        return self.softmax(logits)

class ExperienceBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
    
    def add_experience(self, state, action, reward):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

# В movement_nn.py можно улучшить архитектуру:
class EnhancedMovementNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.net(x)

# Улучшенный выбор новой позиции
def choose_new_position(creature, creatures, density_map, city_center, village_centers, radius=10):
    probabilities = {"city": 0.7, "village": 0.2, "wild": 0.1}
    
    # Генерируем цель в зависимости от распределения
    rand_val = np.random.rand()
    if rand_val < probabilities["city"]:
        target = city_center
        target_type = "город"
    elif rand_val < probabilities["city"] + probabilities["village"]:
        target = village_centers[np.random.randint(len(village_centers))]
        target_type = "деревня"
    else:
        target = (np.random.randint(density_map.shape[1]), np.random.randint(density_map.shape[0]))
        target_type = "дикая местность"
    
    print(f"[Лог] Существо {creature} выбрало цель: {target} ({target_type})")

    # Двигаемся в сторону цели
    best_position = (creature.x, creature.y)
    best_distance = np.linalg.norm(np.array(best_position) - np.array(target))

    for x in range(creature.x - 1, creature.x + 2):
        for y in range(creature.y - 1, creature.y + 2):
            if 0 <= x < density_map.shape[1] and 0 <= y < density_map.shape[0]:
                distance = np.linalg.norm(np.array([x, y]) - np.array(target))
                if distance < best_distance:
                    best_position = (x, y)
                    best_distance = distance

    # print(f"[Лог] Существо {creature} двигается к {best_position}, расстояние до цели: {best_distance:.2f}")
    
    return best_position


def get_state(self):
    from creature import Creature
    # Получаем значения для инструментов
    tools = [int(self.tools.get("топор", 0)), 
             int(self.tools.get("кирка", 0)), 
             int(self.tools.get("лопата", 0))]
    evolution_stage = {"Австралопитеки": 0, "Человек умелый": 1, "Гейдельбергский человек": 2, "Древние цивилизации": 3, "Средневековье": 4, "Новое время": 5}[self.evolution_stage]
    
    density = Creature.get_density_at_position(self.x, self.y, Creature.creatures, radius=10)  # Добавили плотность
    center_x = self.screen.get_width() // (2 * TILE_SIZE)
    center_y = self.screen.get_height() // (2 * TILE_SIZE)
    distance_to_center = abs(self.x - center_x) + abs(self.y - center_y)  # Добавили дистанцию до центра
    
    state = np.array([
        self.x, self.y, self.speed, self.health, self.intelligence, 
        evolution_stage, sum(tools), density, distance_to_center  # Теперь 9 признаков
    ])
    print(f"Размерность состояния: {state.shape}")
    
    return state


# Функция для выбора действия на основе состояния
def choose_action(state, model, creature, density_map, epsilon=0.5):
    if np.random.rand() < epsilon:
        action = np.random.randint(4)
        # print(f"[Лог] Существо {creature} выбрало случайное действие: {action}")
        return action
    
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
        print("[Ошибка] Найдены NaN или Inf в state_tensor:", state_tensor)
        return np.random.randint(4)  # Выбираем случайное действие вместо ошибки

    if state_tensor.shape[1] != 9:
        print(f"[Ошибка] Ожидаемая форма состояния: (1, 9), получено: {state_tensor.shape}")
        raise ValueError("State vector size mismatch. Expected 9 features.")

    with torch.no_grad():
        action_probs = model(state_tensor).numpy().flatten()
    
    if np.isnan(action_probs).any() or np.isinf(action_probs).any():
        print("[Ошибка] NaN или Inf в action_probs:", action_probs)
        return np.random.randint(4)  # Подстраховка

    chosen_action = np.argmax(action_probs)
    # print(f"[Лог] Существо {creature} выбрало действие {chosen_action} с вероятностями {action_probs}")
    
    return chosen_action


REWARD_CONFIG = {
    "move_penalty": -2,
    "center_bonus": 5,
    "new_tile_bonus": 1,
    "house_bonus": 10,
    "density_penalty": -2,
    "edge_penalty": -5,
    "near_house_penalty": 5
}

def get_reward(creature, creatures, city_center, village_centers, max_density=5, radius=10, house_location=None, config=REWARD_CONFIG):
    try:
        
        if not pygame.get_init():
            return 0
        from main import houses
        reward = 0
        
        # Штраф за неподвижность
        if hasattr(creature, 'last_pos'):
            if creature.last_pos == (creature.x, creature.y):
                reward -= 10 
        creature.last_pos = (creature.x, creature.y)

        
        if not hasattr(pygame, "get_init") or not pygame.get_init():
            return 0
        
        # Бонус за движение к центру карты
        center_x = creature.screen.get_width() // (2 * TILE_SIZE)
        center_y = creature.screen.get_height() // (2 * TILE_SIZE)
        distance_to_center = abs(creature.x - center_x) + abs(creature.y - center_y)
        reward -= distance_to_center * 0.15  # Штрафуем отдаление от центра
        
        # Бонус за новые клетки
        if not hasattr(creature, 'visited'):
            creature.visited = set()
        if (creature.x, creature.y) not in creature.visited:
            reward += 1
            creature.visited.add((creature.x, creature.y))
        
        if creature.has_house:
            reward += config["house_bonus"]
        
        # Штраф за перенаселение (плотность на позиции)
        density = get_density(creature, creatures, radius)
        if density > 3:
            reward -= 5 * (density - 3)  # Дополнительный штраф за превышение плотности
        
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
        
            # Бонус за нахождение в городе
        if np.linalg.norm(np.array([creature.x, creature.y]) - np.array(city_center)) < 10:
            reward += 5
        
        # Бонус за нахождение в деревне
        for village in village_centers:
            if np.linalg.norm(np.array([creature.x, creature.y]) - np.array(village)) < 5:
                reward += 3
                break
        
        # Штраф за край карты
        max_x = creature.screen.get_width() // TILE_SIZE - 1
        max_y = creature.screen.get_height() // TILE_SIZE - 1
        if creature.x < 5 or creature.x > max_x - 5 or creature.y < 5 or creature.y > max_y - 5:
            reward -= 15
            
        # Штраф за нахождение рядом с чужим домом
        for house in houses:
            if house != house_location:  # Игнорируем свой дом
                distance_to_house = np.linalg.norm(np.array([creature.x, creature.y]) - np.array(house))
                if distance_to_house <= 10:  # Если дом в радиусе 3 клеток
                    reward -= config["near_house_penalty"]  # Применяем штраф
                    break  # Прерываем цикл, чтобы не проверять остальные дома
        
        return reward
    except pygame.error:
        return 0
    except Exception as e:
        print(f"Ошибка в get_reward: {str(e)}")
        return 0


def get_density(creature, creatures, radius=10):
    # Считаем количество существ в радиусе 10 клеток от текущей позиции
    density = 0
    for other_creature in creatures:
        if np.linalg.norm(np.array([creature.x, creature.y]) - np.array([other_creature.x, other_creature.y])) < radius:
            density += 1
    return density

# Функция обучения
def train_model(model, optimizer, buffer, gamma=0.98):
    if len(buffer.states) < 32:
        return

    valid_indices = [
        i for i, s in enumerate(buffer.states)
        if not np.isnan(s).any() and not np.isinf(s).any()
    ]
    
    states = torch.tensor([buffer.states[i] for i in valid_indices], dtype=torch.float32)
    actions = torch.tensor([buffer.actions[i] for i in valid_indices], dtype=torch.int64)
    rewards = torch.tensor([buffer.rewards[i] for i in valid_indices], dtype=torch.float32)

    # Очистка буфера
    buffer.states, buffer.actions, buffer.rewards = [], [], []

    # Нормализация наград
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1e5, neginf=-1e5)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    optimizer.zero_grad()
    action_probs = model(states)

    if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
        print("[Ошибка] NaN или Inf в action_probs во время обучения:", action_probs)
        return

    # Исправленный расчет `log_probs`
    log_probs = torch.log(torch.clamp(action_probs.gather(1, actions.unsqueeze(1)), min=1e-9))
    loss = -torch.mean(log_probs.squeeze() * rewards)

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("[Ошибка] Найден NaN или Inf в loss!")
        return

    loss.backward()
    
    for param in model.parameters():
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print("[Ошибка] Найден NaN в градиентах!")
            return

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    
    
def update_and_train(creature, model, optimizer, buffer, world_map, creatures, world, city_center, village_centers):
    # Двигаем существо
    creature.move(world_map, creatures, model, world)
    
    # Получаем награду
    reward = get_reward(creature, creatures, city_center, village_centers)
    
    # Добавляем в буфер
    buffer.add_experience(creature.get_state(creatures), creature.action, reward)
    
    # Тренируем модель (здесь вызываем после всех действий с существом)
    train_model(model, optimizer, buffer)