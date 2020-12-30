import math
import random

import pygame as pg
from pygame.locals import *
import numpy as np
import numba


def draw(screen, cell, width, height, wline, colors):
    screen.fill(colors['background'])

    # Draw lines
    x, y = cell, cell
    for i in range(width//cell - 1):
        pg.draw.line(
            screen, colors['line'],
            (x, 0), (x, height), wline
        )
        x += cell + wline

    for i in range(height//cell - 1):
        pg.draw.line(
            screen, colors['line'],
            (0, y), (width, y), wline
        )
        y += cell + wline

    calc = lambda cord: cord*(cell + wline)

    # Draw map
    for y, i in enumerate(map):
        for x, j in enumerate(i):
            if j == 1:
                pg.draw.rect(
                    screen, colors['cell'],
                    pg.Rect(calc(x), calc(y), cell, cell)
                )

    pg.display.update()

def update(simulate, checking_zone, sbrules):
    global map

    if simulate:
        map = check_cells(map, checking_zone, sbrules)

@numba.njit(fastmath=True)
def check_cells(map, checking_zone, sbrules):
    new_map = np.zeros(map.shape)

    for y, i in enumerate(map):
        for x, j in enumerate(i):
            current_cell = np.array([x, y])
            neighborhoods = 0

            for a in checking_zone:
                checking_cell = current_cell + a

                if (not (checking_cell[0] >= len(i) or checking_cell[0] < 0) and
                        not (checking_cell[1] >= len(map) or checking_cell[1] < 0)):
                    if map[checking_cell[1]][checking_cell[0]] == 1:
                        neighborhoods += 1

            if map[current_cell[1]][current_cell[0]] == 1:
                if neighborhoods in sbrules[0]:
                    new_map[current_cell[1]][current_cell[0]] = 1
            else:
                if neighborhoods in sbrules[1]:
                    new_map[current_cell[1]][current_cell[0]] = 1

    return new_map

@numba.njit(fastmath=True)
def filling_function(x, y):
    """Function for filling the map."""
    if random.random() < .1:
        return 1
    else:
        return 0

@numba.njit(fastmath=True)
def generate_map(cell, width, height):
    map = np.zeros((height//cell, width//cell))

    for y, i in enumerate(map):
        for x, j in enumerate(i):
            if filling_function(x, y) == 1:
                map[y][x] = 1

    return map

pg.init()

# Color palette
colors = {
    'background': pg.Color('#2F4858'),
    'line': pg.Color('#33658A'),
    'cell': pg.Color('#86BBD8')
}

# Display settings
height, width = 720, 1280
FPS = 15

# Game settings
run = True
simulate = True
cell = 15
wline = 1
# Rules. First tuple for survive, second tuple for born
sbrules = (
    (2, 3), (3,)
)
# Check this neighborhoods
checking_zone = np.array([
    (-1, -1), (0, -1), (1, -1), (-1, 0),
    (1, 0), (-1, 1), (0, 1), (1, 1)
])

map = generate_map(cell, width, height)
print('map:\n', map)

screen = pg.display.set_mode((width, height))
pg.display.set_caption('I pay respect to John Horton Conway')
clock = pg.time.Clock()

while run:
    clock.tick(FPS)

    for event in pg.event.get():
        if event.type == QUIT:
            quit()

        if event.type == KEYDOWN:
            # Regenerate map
            if event.key == K_r:
                map = generate_map(cell, width, height)
            elif event.key == K_d:
                simulate = not simulate
            elif event.key == K_c:
                map = np.zeros(map.shape)

        if event.type == MOUSEBUTTONDOWN and not simulate:
            if event.button == 1:
                pos = event.pos
                x = math.floor(pos[0]/(cell + wline))
                y = math.floor(pos[1]/(cell + wline))
                map[y][x] = not map[y][x]

    print(f'fps: {clock.get_fps():.2f}')
    draw(screen, cell, width, height, wline, colors)
    update(simulate, checking_zone, sbrules)
