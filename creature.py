# creature.py
import pygame
import random
from settings import TILE_SIZE, CREATURE_SIZE, CREATURE_SPEED
from movement_nn import choose_action
import numpy as np
import torch
from evolution_nn import EvolutionNN
import math
from map import load_character_sprites

MAX_CREATURES = 50

class Creature:
    def __init__(self, x, y, gender, race, screen, tile_size, speed=0.01, health=100, intelligence=1.0):
        self.evolution_model = EvolutionNN()
        self.evolution_model.load_state_dict(torch.load("evolution_model.pth"))
        self.evolution_model.eval()
        self.TILE_SIZE = tile_size
        self.x = x
        self.y = y
        self.gender = gender
        self.race = race
        self.age = random.randint(1, 100)
        self.color = (255, 0, 0)
        self.screen = screen
        self.speed = speed
        self.health = health  
        self.intelligence = intelligence  # Новый параметр интеллекта
        self.houses_built = 0
        self.evolution_stage = "Австралопитеки"  # Стартовая стадия
        self.evolution_progress = 0
        self.tools = {"топор": False, "кирка": False, "лопата": False}
        self.explored_tiles = set()  # Исследованные территории
        self.total_resources = 0     # Суммарно добыто ресурсов
        self.tools_created = 0      # Создано инструментов
        self.evolution_data = None  # Данные для нейросети
        self.sprite = pygame.image.load(f"textures/человек.png")  # Файлы должны быть в "textures/"
        self.sprite = pygame.transform.scale(self.sprite, (tile_size, tile_size))
        
        # Инвентарь для ресурсов
        self.inventory = {"wood": 0, "stone": 0, "food": 0}
        self.has_house = False  # Есть ли у существа дом
        self.gather_speed = 0.1 + (self.intelligence * 0.5)  # Пример: увеличиваем скорость добычи с интеллектом
        
    def evolve(self):
        # Переход на следующий этап эволюции
        evolution_stages = [
            "Австралопитеки", "Человек умелый", "Гейдельбергский человек",
            "Древние цивилизации", "Средневековье", "Новое время"
        ]
        
        # Находим индекс текущей стадии и увеличиваем его
        current_index = evolution_stages.index(self.evolution_stage)
        if current_index < len(evolution_stages) - 1:
            self.evolution_stage = evolution_stages[current_index + 1]
            print(f"Существо {self} эволюционировало в {self.evolution_stage}")
            return self.evolution_stage
        else:
            print(f"Существо {self} уже достигло максимальной стадии эволюции!")
            return
    
    def gather_resources(self, world_map, creatures):
        biome = world_map[self.x, self.y]
        multipliers = {
            "wood": 1.5 if self.tools["топор"] else 1.0,
            "stone": 1.8 if self.tools["кирка"] else 1.0,
            "food": 1.3 if self.tools["лопата"] else 1.0
        }

        collected = 0.0  # Инициализируем переменную для подсчета собранных ресурсов

        if biome < -0.1:
            return  # Выход, если это вода
        elif biome < 0.0:
            food = 0.01 * multipliers["food"]
            self.inventory["food"] += food
            collected += food
        elif biome < 0.2:
            food = 0.02 * multipliers["food"]
            wood = 0.01 * multipliers["wood"]
            stone = 0.009 * multipliers["stone"]
            self.inventory["food"] += food
            self.inventory["wood"] += wood
            self.inventory["stone"] += stone
            collected += food + wood + stone
        elif biome < 0.4:
            wood = 0.03 * multipliers["wood"]
            food = 0.02 * multipliers["food"]
            self.inventory["wood"] += wood
            self.inventory["food"] += food
            collected += wood + food
        else:
            stone = 0.02 * multipliers["stone"]
            self.inventory["stone"] += stone
            collected += stone

        # Обновляем общее количество собранных ресурсов
        self.total_resources += collected
        self.explored_tiles.add((self.x, self.y))

    def craft_tool(self, tool_type):
        requirements = {
            "топор": {"wood": 3, "stone": 1},
            "кирка": {"wood": 2, "stone": 3},
            "лопата": {"wood": 1, "stone": 2}
        }

        # Проверяем, достаточно ли ресурсов
        if all(self.inventory[k] >= v for k, v in requirements[tool_type].items()):
            # Уменьшаем ресурсы
            for item, amount in requirements[tool_type].items():
                self.inventory[item] -= amount
            # Активируем инструмент
            self.tools[tool_type] = True
            self.tools_created += 1  # Увеличиваем счетчик созданных инструментов
            print(f"Создан {tool_type}!")
        else:
            print(f"Недостаточно ресурсов для создания {tool_type}!")

    def check_evolution(self):
        
        if self.evolution_stage == "Новое время":
            print(f"Существо {self} уже достигло максимальной стадии эволюции!")
            return
        
        evolution_requirements = {
            "Австралопитеки": {"wood": 20, "stone": 10},
            "Человек умелый": {"houses": 1, "food": 50},
            "Гейдельбергский человек": {"tools": 2},
            "Древние цивилизации": {"tools": 2},
            "Средневековье": {"tools": 2},
            "Новое время": {"tools": 2}
        }
        
        current_stage = list(evolution_requirements.keys()).index(self.evolution_stage)
        if current_stage == len(evolution_requirements)-1: return

        req = evolution_requirements[self.evolution_stage]
        satisfied = True
        
        # Проверка ресурсных требований
        if "wood" in req and self.inventory["wood"] < req["wood"]:
            satisfied = False
        if "stone" in req and self.inventory["stone"] < req["stone"]:
            satisfied = False
        if "houses" in req and self.houses_built < req["houses"]:
            satisfied = False
        if "tools" in req and sum(self.tools.values()) < req["tools"]:
            satisfied = False

        # Проверка интеллекта через нейросеть
        if satisfied:
            input_data = torch.tensor([
                self.total_resources / 500,  # Нормализация
                sum(self.tools.values()),
                self.houses_built,
                len(self.explored_tiles) / 50,
                self.intelligence / 10,
                self.age / 100
            ], dtype=torch.float32)

            with torch.no_grad():
                evolution_chance = self.evolution_model(input_data).item()
            
            if evolution_chance > 0.3:  # Пороговое значение
                self.evolve()
    
    
    def get_state(self, creatures):
        tools = [int(self.tools["топор"]), int(self.tools["кирка"]), int(self.tools["лопата"])]
        evolution_stage = {"Австралопитеки": 0, "Человек умелый": 1, "Гейдельбергский человек": 2, "Древние цивилизации": 3, "Средневековье": 4, "Новое время": 5}[self.evolution_stage]
        
        # Добавим дополнительные признаки
        density = self.get_density(creatures, radius=10)  # Плотность
        center_x = self.screen.get_width() // (2 * self.TILE_SIZE)
        center_y = self.screen.get_height() // (2 * self.TILE_SIZE)
        distance_to_center = abs(self.x - center_x) + abs(self.y - center_y)  # Дистанция до центра
        
        # Возвращаем 9 признаков
        return np.array([
            self.x,
            self.y,
            self.speed,
            self.health,
            self.intelligence,
            evolution_stage,
            sum(tools),  # Сумма инструментов
            density,  # Плотность
            distance_to_center  # Дистанция до центра
        ])
        
        state = np.nan_to_num(state, nan=0.0, posinf=100.0, neginf=-100.0)
        return state.astype(np.float32)


    def get_density(self, creatures, radius=10):
        neighbors = self.get_neighbors(creatures, radius)
        return sum(neighbors)

    def get_neighbors(self, creatures, radius=10):
        neighbors = []
        for other in creatures:
            # Проверяем, что существо не является собой
            if other != self:
                # Вычисляем евклидово расстояние
                distance = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
                # Если существо в пределах радиуса, добавляем его
                if distance <= radius:
                    neighbors.append(1)
                else:
                    neighbors.append(0)
        return neighbors

    def _is_occupied(self, x, y, creatures):
        for creature in creatures:
            if creature.x == x and creature.y == y:
                return True  
        return False  

    def move(self, world_map, creatures, model, world):
        # Проверка границ перед обращением к world_map
        max_x = world_map.shape[0] - 1
        max_y = world_map.shape[1] - 1
        
        # Ограничение координат существа
        self.x = min(max(0, self.x), max_x)
        self.y = min(max(0, self.y), max_y)
        
        # безопасная проверка на воду
        if world_map[self.x, self.y] < -0.1:
            self.health = 0
            return

        density_map = self.get_density_map(creatures)
        state = self.get_state(creatures)
        action = choose_action(state, model, self, density_map)
        new_x, new_y = self.get_new_position(action, world_map)
        
        # Проверяем, не является ли новая клетка водой
        if world_map[new_x, new_y] < -0.1:
            self.health = 0  # Существо умирает, если попало в воду
            return

        self.x, self.y = new_x, new_y
        self.gather_resources(world_map, creatures)

        
    def perform_action(self, action):
        max_x = (self.screen.get_width() // self.TILE_SIZE) - 1
        max_y = (self.screen.get_height() // self.TILE_SIZE) - 1
        
        if action == 0:  # Вверх
            self.y = max(0, self.y - 1)
        elif action == 1:  # Вниз
            self.y = min(max_y, self.y + 1)
        elif action == 2:  # Влево
            self.x = max(0, self.x - 1)
        elif action == 3:  # Вправо
            self.x = min(max_x, self.x + 1)

    def get_new_position(self, action, world_map):
        max_x = (self.screen.get_width() // self.TILE_SIZE) - 1
        max_y = (self.screen.get_height() // self.TILE_SIZE) - 1
        
        new_x, new_y = self.x, self.y

        if action == 0:  # Вверх
            new_y = max(0, new_y - 1)
        elif action == 1:  # Вниз
            new_y = min(max_y, new_y + 1)
        elif action == 2:  # Влево
            new_x = max(0, new_x - 1)
        elif action == 3:  # Вправо
            new_x = min(max_x, new_x + 1)

        # Гарантируем, что координаты не выйдут за пределы
        new_x = min(max(0, new_x), max_x)
        new_y = min(max(0, new_y), max_y)
        
        # Проверка на воду
        if world_map[new_x, new_y] < -0.1:
            return self.x, self.y  # Не двигаемся
        
        return new_x, new_y

    def build_house(self, houses):
        required_wood = 5
        required_stone = 1
        max_houses_per_creature = 3
        min_distance = 10  # Минимальное расстояние между домами

        # Проверка лимита домов у существа
        if self.houses_built >= max_houses_per_creature:
            return

        # Проверка расстояния до других домов
        for (hx, hy) in houses:
            distance = np.sqrt((self.x - hx)**2 + (self.y - hy)**2)
            if distance < min_distance:
                return

        # Проверка ресурсов
        if self.inventory["wood"] >= required_wood and self.inventory["stone"] >= required_stone:
            houses.append((self.x, self.y))
            self.houses_built += 1
            self.inventory["wood"] -= required_wood
            self.inventory["stone"] -= required_stone
            self.has_house = True
            print(f"Дом построен на ({self.x}, {self.y})! Ресурсы: wood={self.inventory['wood']}, stone={self.inventory['stone']}")
            
    def get_density_map(self, creatures):
        width_in_tiles = self.screen.get_width() // self.TILE_SIZE
        height_in_tiles = self.screen.get_height() // self.TILE_SIZE
        density_map = np.zeros((height_in_tiles, width_in_tiles)) 

        for creature in creatures:
            x, y = creature.x, creature.y
            if 0 <= x < width_in_tiles and 0 <= y < height_in_tiles:
                density_map[y, x] += 1
        return density_map
    
    def draw(self):
        if self.health <= 0:  # Если существо мертво, не рисуем его
            return
        self.screen.blit(self.sprite, (self.x * self.TILE_SIZE, self.y * self.TILE_SIZE))



    def check_death(self):
        if self.age <= 30:
            death_prob = 0.00001 
        elif self.age <= 70:
            death_prob = 0.000001  
        else:
            death_prob = 0.0000001  

        return random.random() < death_prob

    def reproduce(self, creatures):
        from main import params
        MAX_CREATURES = params["max_creatures"]
        if len(creatures) >= MAX_CREATURES:
            return  # Ограничение на количество существ

        if self.gender == 'female':
            for other in creatures:
                if other.gender == 'male' and abs(self.x - other.x) < 2 and abs(self.y - other.y) < 2:
                    if random.random() < 0.01:
                        new_x = self.x + random.choice([-1, 0, 1])
                        new_y = self.y + random.choice([-1, 0, 1])
                        max_x = (self.screen.get_width() // self.TILE_SIZE) - 1
                        max_y = (self.screen.get_height() // self.TILE_SIZE) - 1
                        new_x = max(0, min(max_x, new_x))
                        new_y = max(0, min(max_y, new_y))
                        
                        if not any(c.x == new_x and c.y == new_y for c in creatures):
                            new_creature = Creature(new_x, new_y, random.choice(['male', 'female']), self.race, self.screen, self.TILE_SIZE)
                            creatures.append(new_creature)