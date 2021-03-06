import math
import time
import random

import pygame as pg
from pygame.locals import *
import numpy as np
import numba


class Timer:
    def __init__(self, interval):
        self.interval = interval
        self.last_time = time.time()

    def check(self):
        current_time = time.time()

        if current_time - self.last_time >= self.interval:
            self.last_time = current_time
            return True
        else:
            return False


class MouseControl:
    def get_mouse_position_on_map(self, pos=None):
        if pos is None:
            pos = pg.mouse.get_pos()

            x = math.floor(pos[0]/(cell + wline))
            y = math.floor(pos[1]/(cell + wline))

            return (x, y)
        else:
            x = math.floor(pos[0]/(cell + wline))
            y = math.floor(pos[1]/(cell + wline))

            return (x, y)

    def remember_pos(self):
        self.remembered_pos = np.array(pg.mouse.get_pos())

    def get_remembered_pos(self):
        return self.get_mouse_position_on_map(self.remembered_pos)

    def get_mouse_offset(self):
        self.new_pos = np.array(pg.mouse.get_pos())

        x, y = self.get_mouse_position_on_map(
            self.remembered_pos - self.new_pos
        )

        return (x, y)


def draw():
    global screen, cell, width, height, wline, colors, cursor, mouse_control

    calc = lambda cord: cord*(cell + wline)

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

    # Draw map
    for y, i in enumerate(map):
        for x, j in enumerate(i):
            if j == 1:
                pg.draw.rect(
                    screen, colors['cell'],
                    pg.Rect(calc(x), calc(y), cell, cell)
                )

    # Draw cursor
    if not simulate:
        x, y = mouse_control.get_mouse_position_on_map()
        screen.blit(cursor, (calc(x), calc(y)))

    pg.display.update()

def update():
    global map, simulate, checking_zone, sbrules, check_cells_timer

    if simulate and check_cells_timer.check():
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
    'cell': pg.Color('#86BBD8'),
    'cursor': pg.Color('green')
}

# Display settings
height, width = 720, 1280
FPS = 60

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

# Interval in seconds
check_cells_timer = Timer(.2)

map = generate_map(cell, width, height)
print('map:\n', map)

# Cursor
cursor = pg.Surface((cell, cell))
cursor.fill(colors['cursor'])
cursor.set_alpha(128)
# Hide cursor
pg.mouse.set_visible(False)

mouse_control = MouseControl()

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
            elif event.key == K_e:
                simulate = not simulate
            elif event.key == K_c:
                map = np.zeros(map.shape)

    if not simulate:
        mouse_buttons = pg.mouse.get_pressed()
        # Create cell
        if mouse_buttons[0]:
            x, y = mouse_control.get_mouse_position_on_map()
            map[y][x] = 1
        # Delete cell
        elif mouse_buttons[2]:
            x, y = mouse_control.get_mouse_position_on_map()
            map[y][x] = 0


    # print(f'fps: {clock.get_fps():.2f}')
    draw()
    update()
