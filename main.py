# main.py
import sys
import pygame
import random
from map import generate_map, draw_map, load_textures, load_character_sprites
from creature import Creature
from movement_nn import MovementNN, choose_action, get_reward, ExperienceBuffer, train_model, update_and_train
from settings import WIDTH, HEIGHT, TILE_SIZE
import torch
import torch.optim as optim
from evolution_nn import EvolutionNN

# Константы для интерфейса
DETAILS_WIDTH = 400
DETAILS_HEIGHT = 900
INFO_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
SCROLL_SPEED = 20

game_speed = 30  
paused = False  

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Генерация мира с существами")
stats_font = pygame.font.Font(None, 24)
details_font = pygame.font.Font(None, 18)

textures = load_textures()
sprites = load_character_sprites()

# Создание нейросети и оптимизатора
model = MovementNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Инициализация модели эволюции
evolution_model = EvolutionNN()
evolution_model.load_state_dict(torch.load('evolution_model.pth'))  # Загрузка обученной модели

zoom_level = 1.0

# Генерация мира
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
    
    # Корректный расчет смещения
    y_offset = 40 - offset
    for i, creature in enumerate(creatures):
        if y_offset > DETAILS_HEIGHT:
            break
        if y_offset < -90: 
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
    params = {
        'tile_size': 15,
        'max_creatures': 50,
        'scale': 80,
        'octaves': 6,
        'persistence': 0.1,
        'lacunarity': 2.0,
        'num_creatures': 10
    }
    selected_param = 0
    param_names = list(params.keys())
    race_options = ["человек", "эльф", "орк"]
    selected_race_index = 0
    
    while True:
        screen.fill((50, 50, 50))
        y = 50
        
        # Отрисовка параметров
        for i, (name, value) in enumerate(params.items()):
            color = (255, 255, 0) if i == selected_param else (255, 255, 255)
            text = stats_font.render(
                f"{name}: {value:.1f}" if isinstance(value, float) else f"{name}: {value}", 
                True, 
                color
            )
            screen.blit(text, (50, y))
            y += 30
        
        # Отрисовка выбора расы
        text = stats_font.render(f"Раса: {race_options[selected_race_index]}", True, (255, 255, 255))
        screen.blit(text, (50, y + 20))
        text = stats_font.render("ENTER - начать", True, (255, 255, 255))  # Исправлено здесь
        screen.blit(text, (50, y + 50))  # И здесь
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # В main.py в блоке обработки события MOUSEWHEEL:
            if event.type == pygame.MOUSEWHEEL:
                # Получаем позицию мыши в мировых координатах
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_x_before = (mouse_x + camera_offset.x) / zoom_level
                world_y_before = (mouse_y + camera_offset.y) / zoom_level

                # Изменяем масштаб
                old_zoom = zoom_level
                if event.y > 0:
                    zoom_level = min(2.0, zoom_level * 1.1)
                else:
                    # Рассчитываем минимальный допустимый зум
                    min_zoom_x = screen.get_width() / (WIDTH * TILE_SIZE)
                    min_zoom_y = screen.get_height() / (HEIGHT * TILE_SIZE)
                    min_zoom = max(min_zoom_x, min_zoom_y)
                    zoom_level = max(min_zoom, zoom_level * 0.9)

                # Корректируем смещение камеры для сохранения позиции под курсором
                world_x_after = (mouse_x + camera_offset.x) / zoom_level
                world_y_after = (mouse_y + camera_offset.y) / zoom_level
                camera_offset.x += (world_x_before - world_x_after) * zoom_level
                camera_offset.y += (world_y_before - world_y_after) * zoom_level

                # Ограничиваем смещение камеры
                max_offset_x = WIDTH * TILE_SIZE * zoom_level - screen.get_width()
                max_offset_y = HEIGHT * TILE_SIZE * zoom_level - screen.get_height()
                camera_offset.x = max(0, min(camera_offset.x, max_offset_x))
                camera_offset.y = max(0, min(camera_offset.y, max_offset_y))
                
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_param = (selected_param - 1) % len(params)
                elif event.key == pygame.K_DOWN:
                    selected_param = (selected_param + 1) % len(params)
                elif event.key == pygame.K_LEFT:
                    current_param = param_names[selected_param]
                    if current_param in ['persistence', 'lacunarity']:
                        # Для параметров с плавающей точкой
                        params[current_param] = round(max(0.1, params[current_param] - 0.1), 1)
                    elif current_param == 'tile_size':
                        # Размер тайла: шаг 1, минимум 5
                        params[current_param] = max(5, params[current_param] - 1)
                    elif current_param == 'max_creatures':
                        # Лимит существ: шаг 5, минимум 10
                        params[current_param] = max(10, params[current_param] - 5)
                    else:
                        # Для остальных целочисленных параметров
                        params[current_param] = max(1, params[current_param] - 1)

                elif event.key == pygame.K_RIGHT:
                    current_param = param_names[selected_param]
                    if current_param in ['persistence', 'lacunarity']:
                        # Для параметров с плавающей точкой
                        params[current_param] = round(min(5.0, params[current_param] + 0.1), 1)
                    elif current_param == 'tile_size':
                        # Размер тайла: шаг 1, максимум 20
                        params[current_param] = min(50, params[current_param] + 1)
                    elif current_param == 'max_creatures':
                        # Лимит существ: шаг 5, максимум 200
                        params[current_param] = min(200, params[current_param] + 5)
                    else:
                        # Для остальных целочисленных параметров
                        params[current_param] += 1

                elif event.key == pygame.K_TAB:
                    selected_race_index = (selected_race_index + 1) % len(race_options)
                    
                elif event.key == pygame.K_RETURN:
                    pygame.event.clear(pygame.KEYDOWN)
                    pygame.time.delay(50)
                    return race_options[selected_race_index], params

# Генерация мира продолжение
selected_race, params = show_menu()
TILE_SIZE = params['tile_size']          # Обновляем глобальный TILE_SIZE
MAX_CREATURES = params['max_creatures']  # Обновляем лимит существ
world_map = generate_map(WIDTH, HEIGHT, TILE_SIZE, params)
num_creatures = params['num_creatures']

# Создание существ
max_x = (WIDTH // TILE_SIZE) - 1
max_y = (HEIGHT // TILE_SIZE) - 1
for _ in range(num_creatures):
    gender = random.choice(['male', 'female'])
    x, y = random.randint(0, max_x), random.randint(0, max_y)
    creatures.append(Creature(x, y, gender, selected_race, screen, TILE_SIZE))

city_center = (WIDTH // (2 * TILE_SIZE), HEIGHT // (2 * TILE_SIZE))  # Центр карты
village_centers = [
    (WIDTH // (4 * TILE_SIZE), HEIGHT // (4 * TILE_SIZE)),  # Первая деревня
    (3 * WIDTH // (4 * TILE_SIZE), 3 * HEIGHT // (4 * TILE_SIZE))  # Вторая деревня
]


# Игровой цикл
running = True
clock = pygame.time.Clock()
camera_offset = pygame.Vector2(0, 0)
zoom_level = 1.0  # Стартовый масштаб
old_zoom = zoom_level  # Для расчёта смещения камеры
scroll_direction = 0  # Чтобы избежать ошибки, если скролла не было

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            player_x, player_y = 10, 10  # Координаты игрока
   
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Спавн только по левой кнопке мыши (button == 1)
                if event.button == 1:  
                    mx, my = pygame.mouse.get_pos()
                    x, y = mx // TILE_SIZE, my // TILE_SIZE
                    creatures.append(Creature(x, y, random.choice(['male', 'female']), selected_race, screen, TILE_SIZE))
                
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
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:  # Увеличить скорость
                    game_speed = min(game_speed + 100, 60000)  # Максимум 60 FPS
                    print(f"Скорость увеличена: {game_speed} FPS")
                elif event.key == pygame.K_DOWN:  # Уменьшить скорость
                    game_speed = max(game_speed - 5, 1)  # Минимум 1 FPS
                    print(f"Скорость уменьшена: {game_speed} FPS")
                elif event.key == pygame.K_SPACE or event.key == pygame.K_p:  # Пауза
                    paused = not paused
                    print("Пауза" if paused else "Игра продолжается")

        # Если игра на паузе, не обновляем логику, но продолжаем отрисовку
        if paused:
            pygame.display.flip()
            clock.tick(game_speed)
            continue  # Пропускаем этот цикл и ждем следующего события

        max_scroll = max(0, len(creatures)*90 - DETAILS_HEIGHT + 50)
        scroll_offset = min(scroll_offset, max_scroll)
        
        screen.fill((0, 0, 0))
        scaled_tile_size = int(TILE_SIZE * zoom_level)
        draw_map(world_map, screen, scaled_tile_size, textures, camera_offset)

        for creature in creatures[:]:
            # Обновление позиции существа с учетом экрана
            max_x = (creature.screen.get_width() // creature.TILE_SIZE) - 1
            max_y = (creature.screen.get_height() // creature.TILE_SIZE) - 1
            creature.x = min(max(0, creature.x), max_x)
            creature.y = min(max(0, creature.y), max_y)
            
            # Состояние и карта плотности
            state = creature.get_state(creatures)
            density_map = creature.get_density_map(creatures)
            
            # Выбираем действие для существа
            action = choose_action(state, model, creature, density_map)
            
            # Обновляем действия существа
            creature.action = action
            
            # Выполняем движение, рисование, размножение и проверку на смерть
            creature.check_evolution()
            creature.move(world_map, creatures, model, world_map)
            creature.draw(scaled_tile_size, camera_offset)
            creature.reproduce(creatures)
            
            if creature.check_death():
                creatures.remove(creature)
                print("Существо умерло!")
                
            # Строим дома для существа
            creature.build_house(houses)
            
            # Вызываем обновление и тренировку для каждого существа
            update_and_train(creature, model, optimizer, experience_buffer, world_map, creatures, world_map, city_center, village_centers)


        for house in houses:
            house_x, house_y = house
            screen_x = house_x * scaled_tile_size - camera_offset.x
            screen_y = house_y * scaled_tile_size - camera_offset.y
            if (-scaled_tile_size < screen_x < WIDTH and 
                -scaled_tile_size < screen_y < HEIGHT):
                pygame.draw.rect(screen, (139,69,19), 
                    (screen_x, screen_y, scaled_tile_size, scaled_tile_size))
        
        if len(experience_buffer.states) >= 32:
            train_model(model, optimizer, experience_buffer)

        stats = {
            "Всего существ": len(creatures),
            "Мужчины": sum(1 for c in creatures if c.gender == 'male'),
            "Женщины": sum(1 for c in creatures if c.gender == 'female'),
            "Дома": len(houses),
            "Дерево": int(sum(c.inventory["wood"] for c in creatures)),
            "Камень": int(sum(c.inventory["stone"] for c in creatures)),
            "Еда": int(sum(c.inventory["food"] for c in creatures)),
            "Текущая скорость": f"{game_speed} FPS"
        }
        
        y_offset = 10
        for key, value in stats.items():
            text = stats_font.render(f"{key}: {value}", True, (255, 255, 255))
            screen.blit(text, (10, y_offset))
            y_offset += 20

        if show_details and len(creatures) > 0:
            draw_detailed_stats(screen, creatures, details_font, scroll_offset)

        pygame.display.flip()
        clock.tick(game_speed)

except Exception as e:
    print(f"Critical error: {str(e)}")

finally:
    pygame.quit()
    sys.exit()