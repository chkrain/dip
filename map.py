# map.py
import numpy as np
import noise
import pygame
from settings import WIDTH, HEIGHT, TILE_SIZE

# Генерация карты с помощью шума Перлина
def generate_map(WIDTH, HEIGHT, TILE_SIZE, params):
    width_in_tiles = WIDTH // TILE_SIZE  # Количество тайлов по ширине
    height_in_tiles = HEIGHT // TILE_SIZE  # Количество тайлов по высоте
    world = np.zeros((width_in_tiles, height_in_tiles))
    
    # Масштабы шума для большего разнообразия
    scale = 80  # Масштаб шума
    octave_scale = 6  # Количество октав
    persistence = 0.1  # Параметр, контролирующий резкость
    lacunarity = 2.0  # Параметр, контролирующий частоту изменений

    for x in range(width_in_tiles):
        for y in range(height_in_tiles):
            noise_value = noise.pnoise2(
                x/params['scale'], 
                y/params['scale'],
                octaves=params['octaves'],
                persistence=params['persistence'],
                lacunarity=params['lacunarity']
            )
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

def load_textures():
    return {
        "water": pygame.image.load("textures/вода.jpg"),
        "sand": pygame.image.load("textures/песок.jpg"),
        "grass": pygame.image.load("textures/трава.jpg"),
        "forest": pygame.image.load("textures/трава.jpg"),
        "mountain": pygame.image.load("textures/камень.jpg"),
    }
    
def load_character_sprites():
    return {
        "player": pygame.image.load("textures/человек.png"),
        # "enemy": pygame.image.load("textures/enemy.jpg")
    }

# Отрисовка карты
def draw_map(world, screen, TILE_SIZE, textures, camera_offset):
    width_in_tiles = len(world)
    height_in_tiles = len(world[0])

    for x in range(width_in_tiles):
        for y in range(height_in_tiles):
            biome = get_biome(world[x, y])
            texture = pygame.transform.scale(textures[biome], (TILE_SIZE, TILE_SIZE))
            screen.blit(texture, (x * TILE_SIZE - camera_offset.x, y * TILE_SIZE - camera_offset.y))



if __name__ == "__main__":
    params = {
        'scale': 80,
        'octaves': 6,
        'persistence': 0.1,
        'lacunarity': 2.0
    }
    world = generate_map(WIDTH, HEIGHT, TILE_SIZE, params)
