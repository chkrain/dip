# map.py
import numpy as np
import noise
import pygame
from settings import WIDTH, HEIGHT, TILE_SIZE

# Генерация карты с помощью шума Перлина
def generate_map(WIDTH, HEIGHT, TILE_SIZE):
    width_in_tiles = WIDTH // TILE_SIZE  # Количество тайлов по ширине
    height_in_tiles = HEIGHT // TILE_SIZE  # Количество тайлов по высоте
    world = np.zeros((width_in_tiles, height_in_tiles))
    
    # Масштабы шума для большего разнообразия
    scale = 100  # Масштаб шума
    octave_scale = 6  # Количество октав
    persistence = 0.5  # Параметр, контролирующий резкость
    lacunarity = 2.0  # Параметр, контролирующий частоту изменений

    for x in range(width_in_tiles):
        for y in range(height_in_tiles):
            noise_value = noise.pnoise2(x / scale, y / scale, octaves=octave_scale, persistence=persistence, lacunarity=lacunarity)
            world[x, y] = noise_value
    return world

# Определение биома по значению шума
def get_biome(value):
    if value < -0.1:
        return "water"
    elif value < 0.0:
        return "sand"
    elif value < 0.2:
        return "grass"
    elif value < 0.4:
        return "forest"
    else:
        return "mountain"


# Отрисовка карты
def draw_map(world, screen, TILE_SIZE):
    width_in_tiles = len(world)  # Количество тайлов по ширине
    height_in_tiles = len(world[0])  # Количество тайлов по высоте
    
    # Рисуем каждый тайл
    for x in range(width_in_tiles):
        for y in range(height_in_tiles):
            biome = get_biome(world[x, y])
            color = {
                "water": (0, 0, 255),  # Синий для воды
                "sand": (237, 201, 175),  # Песок
                "grass": (34, 139, 34),  # Трава
                "forest": (0, 100, 0),  # Лес
                "mountain": (139, 137, 137),  # Горы
            }[biome]
            # Отрисовываем прямоугольник для каждого тайла
            pygame.draw.rect(screen, color, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

WIDTH, HEIGHT = 1000, 1000
world = generate_map(WIDTH, HEIGHT, TILE_SIZE)
