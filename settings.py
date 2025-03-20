# settings.py
import pygame
pygame.init()
info = pygame.display.Info()
WIDTH = info.current_w
HEIGHT = info.current_h

TILE_SIZE = 20
CREATURE_SIZE = TILE_SIZE // 4
CREATURE_SPEED = 1
