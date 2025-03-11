# main.py
import pygame
import random
from map import generate_map, draw_map
from creature import Creature
from movement_nn import MovementNN, choose_action, get_reward, ExperienceBuffer, train_model
from settings import WIDTH, HEIGHT, TILE_SIZE
import torch
import torch.optim as optim

# Константы для интерфейса
DETAILS_WIDTH = 400
DETAILS_HEIGHT = 600
INFO_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
SCROLL_SPEED = 20

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Генерация мира с существами")
stats_font = pygame.font.Font(None, 24)
details_font = pygame.font.Font(None, 18)

# Создание нейросети и оптимизатора
model = MovementNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Генерация мира
world_map = generate_map(WIDTH, HEIGHT, TILE_SIZE)
creatures = []
houses = []
experience_buffer = ExperienceBuffer()
show_details = False
scroll_offset = 0
max_scroll = 0

def draw_detailed_stats(screen, creatures, font, offset):
    """Отрисовка детальной статистики с прокруткой"""
    details_surface = pygame.Surface((DETAILS_WIDTH, DETAILS_HEIGHT))
    details_surface.fill(INFO_COLOR)
    
    title = font.render("Детальная статистика:", True, TEXT_COLOR)
    details_surface.blit(title, (10, 10))
    
    y_offset = 40 - offset
    for i, creature in enumerate(creatures):
        if y_offset > DETAILS_HEIGHT:
            break
        
        if y_offset < -50:
            y_offset += 90
            continue
            
        text = font.render(f"Существо {i+1}:", True, TEXT_COLOR)
        details_surface.blit(text, (10, y_offset))
        
        info = [
            f"Развитие: {creature.evolution_stage}",
            f"Инструменты: Топор={creature.tools['топор']}, Кирка={creature.tools['кирка']}, Лопата={creature.tools['лопата']}", 
            f"Ресурсы: Дерево={creature.inventory['wood']:.1f}, Камень={creature.inventory['stone']:.1f}, Еда={creature.inventory['food']:.1f}",
            f"Дом: {creature.houses_built}",
            f"Интеллкт: {creature.intelligence}",
            f"Возраст: {creature.age}",
            f"Пол: {creature.gender}"
        ]
        
        for line in info:
            y_offset += 20
            text_line = font.render(line, True, TEXT_COLOR)
            details_surface.blit(text_line, (20, y_offset))
        
        y_offset += 30

    if len(creatures) > 20:
        scroll_height = DETAILS_HEIGHT * (DETAILS_HEIGHT / (len(creatures)*90))
        scroll_y = (scroll_offset / max_scroll) * (DETAILS_HEIGHT - scroll_height)
        pygame.draw.rect(details_surface, (100, 100, 100), 
                        (DETAILS_WIDTH-15, scroll_y, 10, scroll_height))

    screen.blit(details_surface, (WIDTH - DETAILS_WIDTH - 10, 10))

def show_menu():
    """Меню выбора параметров"""
    race_options = ["человек", "эльф", "орк"]
    selected_race_index = 0  # Исправлено имя переменной
    num_creatures = 10
    
    while True:
        screen.fill((50, 50, 50))
        text = stats_font.render(f"Выбранная раса: {race_options[selected_race_index]}", True, (255, 255, 255))
        screen.blit(text, (50, 50))
        text = stats_font.render(f"Количество существ: {num_creatures}", True, (255, 255, 255))
        screen.blit(text, (50, 80))
        text = stats_font.render("Нажмите ENTER для старта", True, (255, 255, 255))
        screen.blit(text, (50, 110))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected_race_index = (selected_race_index - 1) % len(race_options)
                elif event.key == pygame.K_RIGHT:
                    selected_race_index = (selected_race_index + 1) % len(race_options)
                elif event.key == pygame.K_UP:
                    num_creatures += 1
                elif event.key == pygame.K_DOWN:
                    num_creatures = max(1, num_creatures - 1)
                elif event.key == pygame.K_RETURN:
                    return race_options[selected_race_index], num_creatures  # Возвращаем выбранное значение

# Инициализация игры
selected_race, num_creatures = show_menu()  # Получаем выбранную расу

# Создание существ
max_x = (WIDTH // TILE_SIZE) - 1
max_y = (HEIGHT // TILE_SIZE) - 1
for _ in range(num_creatures):
    gender = random.choice(['male', 'female'])
    x, y = random.randint(0, max_x), random.randint(0, max_y)
    creatures.append(Creature(x, y, gender, selected_race, screen))  # Используем selected_race

# Игровой цикл
running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Спавн только по левой кнопке мыши (button == 1)
            if event.button == 1:  
                mx, my = pygame.mouse.get_pos()
                x, y = mx // TILE_SIZE, my // TILE_SIZE
                creatures.append(Creature(x, y, random.choice(['male', 'female']), selected_race, screen))
            
            # Прокрутка колесиком (кнопки 4 и 5) без спавна
            if show_details and event.button in (4, 5):
                scroll_offset = max(0, scroll_offset + (-SCROLL_SPEED if event.button == 4 else SCROLL_SPEED))
                scroll_offset = min(scroll_offset, max_scroll)
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:  # Единая обработка клавиши
                show_details = not show_details
                scroll_offset = 0
                
            if show_details:
                if event.key == pygame.K_UP:
                    scroll_offset = max(0, scroll_offset - SCROLL_SPEED)
                elif event.key == pygame.K_DOWN:
                    scroll_offset = min(max_scroll, scroll_offset + SCROLL_SPEED)

    max_scroll = max(0, len(creatures)*90 - DETAILS_HEIGHT + 50)
    scroll_offset = min(scroll_offset, max_scroll)
    
    screen.fill((0, 0, 0))
    draw_map(world_map, screen, TILE_SIZE)

    for creature in creatures[:]:
        max_x = (creature.screen.get_width() // TILE_SIZE) - 1
        max_y = (creature.screen.get_height() // TILE_SIZE) - 1
        creature.x = min(max(0, creature.x), max_x)
        creature.y = min(max(0, creature.y), max_y)
        
        state = creature.get_state()
        density_map = creature.get_density_map(creatures)
        action = choose_action(state, model, creature, density_map)
        creature.check_evolution()
        creature.move(world_map, creatures, model, world_map)
        creature.draw()
        creature.reproduce(creatures)
        
        if creature.check_death():
            creatures.remove(creature)
            print("Существо умерло!")
            
        creature.build_house(houses)
        reward = get_reward(creature, creatures)
        experience_buffer.add_experience(state, action, reward)

    for house in houses:
        house_x, house_y = house
        pygame.draw.rect(screen, (139, 69, 19), 
                        (house_x * TILE_SIZE, house_y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    if len(experience_buffer.states) >= 32:
        train_model(model, optimizer, experience_buffer)

    stats = {
        "Всего существ": len(creatures),
        "Мужчины": sum(1 for c in creatures if c.gender == 'male'),
        "Женщины": sum(1 for c in creatures if c.gender == 'female'),
        "Дома": len(houses),
        "Дерево": int(sum(c.inventory["wood"] for c in creatures)),
        "Камень": int(sum(c.inventory["stone"] for c in creatures)),
        "Еда": int(sum(c.inventory["food"] for c in creatures))
    }
    
    y_offset = 10
    for key, value in stats.items():
        text = stats_font.render(f"{key}: {value}", True, (255, 255, 255))
        screen.blit(text, (10, y_offset))
        y_offset += 20

    if show_details and len(creatures) > 0:
        draw_detailed_stats(screen, creatures, details_font, scroll_offset)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()