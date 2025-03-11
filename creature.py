# creature.py
import pygame
import random
from settings import TILE_SIZE, CREATURE_SIZE, CREATURE_SPEED
from movement_nn import choose_action
import numpy as np

MAX_CREATURES = 50  # Лимит на количество существ

class Creature:
    def __init__(self, x, y, gender, race, screen, speed=0.01, health=100, intelligence=1.0):
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
        
        # Инвентарь для ресурсов
        self.inventory = {"wood": 0, "stone": 0, "food": 0}
        self.has_house = False  # Есть ли у существа дом
        self.gather_speed = 0.1 + (self.intelligence * 0.5)  # Пример: увеличиваем скорость добычи с интеллектом
    
    def gather_resources(self, world_map, creatures):
        biome = world_map[self.x, self.y]
        multipliers = {
            "wood": 1.5 if self.tools["топор"] else 1.0,
            "stone": 1.8 if self.tools["кирка"] else 1.0,
            "food": 1.3 if self.tools["лопата"] else 1.0
        }
        
        if biome < -0.1: return
        elif biome < 0.0:
            self.inventory["food"] += 0.01 * multipliers["food"]
        elif biome < 0.2:
            self.inventory["food"] += 0.02 * multipliers["food"]
            self.inventory["wood"] += 0.01 * multipliers["wood"]
            self.inventory["stone"] += 0.009 * multipliers["stone"]
        elif biome < 0.4:
            self.inventory["wood"] += 0.03 * multipliers["wood"]
            self.inventory["food"] += 0.02 * multipliers["food"]
        else:
            self.inventory["stone"] += 0.02 * multipliers["stone"]

    def craft_tool(self, tool_type):
        requirements = {
            "топор": {"wood": 3, "stone": 1},
            "кирка": {"wood": 2, "stone": 3},
            "лопата": {"wood": 1, "stone": 2}
        }
        
        if all(self.inventory[k] >= v for k, v in requirements[tool_type].items()):
            for item, amount in requirements[tool_type].items():
                self.inventory[item] -= amount
            self.tools[tool_type] = True
            print(f"Создан {tool_type}!")

    def check_evolution(self):
        evolution_requirements = {
            "Австралопитеки": {"wood": 20, "stone": 10},
            "Человек умелый": {"houses": 1, "food": 50},
            "Гейдельбергский человек": {"tools": 2, "intelligence": 5.0},
            "Древние цивилизации": {"tools": 2, "intelligence": 15.0},
            "Средневековье": {"tools": 2, "intelligence": 150.0},
            "Новое время": {"tools": 2, "intelligence": 1500.0},
        }
        
        current_stage = list(evolution_requirements.keys()).index(self.evolution_stage)
        if current_stage == len(evolution_requirements)-1: return

        req = evolution_requirements[self.evolution_stage]
        satisfied = True
        
        if "wood" in req and self.inventory["wood"] < req["wood"]: satisfied = False
        if "stone" in req and self.inventory["stone"] < req["stone"]: satisfied = False
        if "houses" in req and self.houses_built < req["houses"]: satisfied = False
        if "tools" in req and sum(self.tools.values()) < req["tools"]: satisfied = False
        if "intelligence" in req and self.intelligence < req["intelligence"]: satisfied = False

        if satisfied:
            self.evolution_stage = list(evolution_requirements.keys())[current_stage + 1]
            self.intelligence += 1.0
            self.speed *= 1.2
            print(f"Эволюция! Теперь существо {self.evolution_stage}")
    
    def get_state(self):
        tools = [int(self.tools["топор"]), int(self.tools["кирка"]), int(self.tools["лопата"])]
        evolution_stage = {"Австралопитеки": 0, "Человек умелый": 1, "Гейдельбергский человек": 2, "Древние цивилизации" : 3, "Средневековье": 4, "Новое время": 5}[self.evolution_stage]
        return np.array([
            self.x,
            self.y,
            self.speed,
            self.health,
            self.intelligence,
            evolution_stage,
            sum(tools)  # Сумма инструментов
        ])

    def get_density(self, creatures):
        neighbors = self.get_neighbors(creatures)
        return sum(neighbors)

    def get_neighbors(self, creatures):
        neighbors = []
        for other in creatures:
            if abs(self.x - other.x) <= 1 and abs(self.y - other.y) <= 1:
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
        
        # Теперь безопасная проверка на воду
        if world_map[self.x, self.y] < -0.1:
            self.health = 0
            return

        density_map = self.get_density_map(creatures)
        state = self.get_state()
        action = choose_action(state, model, self, density_map)
        new_x, new_y = self.get_new_position(action)
        
        # Проверяем, не является ли новая клетка водой
        if world[new_x, new_y] < -0.1:
            self.health = 0  # Существо умирает, если попало в воду
            return

        self.x, self.y = new_x, new_y
        self.gather_resources(world_map, creatures)
        
    def perform_action(self, action):
        max_x = (self.screen.get_width() // TILE_SIZE) - 1
        max_y = (self.screen.get_height() // TILE_SIZE) - 1
        
        if action == 0:  # Вверх
            self.y = max(0, self.y - 1)
        elif action == 1:  # Вниз
            self.y = min(max_y, self.y + 1)
        elif action == 2:  # Влево
            self.x = max(0, self.x - 1)
        elif action == 3:  # Вправо
            self.x = min(max_x, self.x + 1)

    def get_new_position(self, action):
        max_x = (self.screen.get_width() // TILE_SIZE) - 1
        max_y = (self.screen.get_height() // TILE_SIZE) - 1
        
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
        width_in_tiles = self.screen.get_width() // TILE_SIZE
        height_in_tiles = self.screen.get_height() // TILE_SIZE
        density_map = np.zeros((height_in_tiles, width_in_tiles)) 

        for creature in creatures:
            x, y = creature.x, creature.y
            if 0 <= x < width_in_tiles and 0 <= y < height_in_tiles:
                density_map[y, x] += 1
        return density_map
    
    def draw(self):
        pygame.draw.circle(self.screen, self.color, (self.x * TILE_SIZE + TILE_SIZE // 2, self.y * TILE_SIZE + TILE_SIZE // 2), CREATURE_SIZE)

    def check_death(self):
        if self.age <= 30:
            death_prob = 0.00001 
        elif self.age <= 70:
            death_prob = 0.000001  
        else:
            death_prob = 0.0000001  

        return random.random() < death_prob

    def reproduce(self, creatures):
        if len(creatures) >= MAX_CREATURES:
            return  # Ограничение на количество существ

        if self.gender == 'female':
            for other in creatures:
                if other.gender == 'male' and abs(self.x - other.x) < 2 and abs(self.y - other.y) < 2:
                    if random.random() < 0.01:
                        new_x = self.x + random.choice([-1, 0, 1])
                        new_y = self.y + random.choice([-1, 0, 1])
                        new_x = max(0, min(TILE_SIZE - 1, new_x))
                        new_y = max(0, min(TILE_SIZE - 1, new_y))
                        
                        if not any(c.x == new_x and c.y == new_y for c in creatures):
                            new_creature = Creature(new_x, new_y, random.choice(['male', 'female']), self.race, self.screen)
                            creatures.append(new_creature)