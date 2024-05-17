import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import sys

HEX_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)
]


class Cell:
    def __init__(self):
        self.walls = [True] * 6  # Each cell has 6 walls initially
        self.visited = False


def initialize_grid(rows, cols):
    return [[Cell() for _ in range(cols)] for _ in range(rows)]


def is_valid_move(x, y, grid, all_visited_sets, current_set_index):
    rows, cols = len(grid), len(grid[0])
    if 0 <= x < rows and 0 <= y < cols and not grid[x][y].visited:
        for i, visited_set in enumerate(all_visited_sets):
            if i != current_set_index and (x, y) in visited_set:
                return False
        return True
    return False


def generate_maze(grid, start_x, start_y, all_visited_sets, current_set_index):
    stack = [(start_x, start_y)]
    grid[start_x][start_y].visited = True

    while stack:
        x, y = stack[-1]
        neighbors = []

        for i, (dx, dy) in enumerate(HEX_DIRECTIONS):
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, grid, all_visited_sets, current_set_index):
                neighbors.append((nx, ny, i))

        if neighbors:
            nx, ny, direction = random.choice(neighbors)
            grid[nx][ny].visited = True
            stack.append((nx, ny))

            # Remove wall between current cell and chosen neighbor
            grid[x][y].walls[direction] = False
            grid[nx][ny].walls[(direction + 3) % 6] = False  # Opposite wall

            # Add the cell to the current visited set
            all_visited_sets[current_set_index].add((nx, ny))
        else:
            stack.pop()

        yield


def get_random_border_point(rows, cols, exclude_points=[]):
    sides = ["top", "bottom", "left", "right"]
    point = None
    while not point or point in exclude_points:
        side = random.choice(sides)
        if side == "top":
            point = (0, random.randint(0, cols - 1))
        elif side == "bottom":
            point = (rows - 1, random.randint(0, cols - 1))
        elif side == "left":
            point = (random.randint(0, rows - 1), 0)
        elif side == "right":
            point = (random.randint(0, rows - 1), cols - 1)
    return point


def create_intertwined_mazes(rows, cols, num_mazes, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = initialize_grid(rows, cols)

    # Define random starting points on the border of the maze
    starts = []
    for _ in range(num_mazes):
        start = get_random_border_point(rows, cols, exclude_points=starts)
        starts.append(start)

    # Set of visited cells for each maze to avoid overlap
    all_visited_sets = [set([start]) for start in starts]

    # Initialize maze generators
    mazes = [generate_maze(grid, start[0], start[1], all_visited_sets, i) for i, start in enumerate(starts)]

    # Alternate steps between the mazes
    while mazes:
        for i, maze in enumerate(mazes[:]):
            try:
                next(maze)
            except StopIteration:
                mazes.remove(maze)

    # Define random exit points within the visited cells of each maze
    exits = []
    for i, visited_set in enumerate(all_visited_sets):
        exit = random.choice(list(visited_set))
        exits.append(exit)

    return grid, starts, exits, all_visited_sets


def hsl_to_rgb(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = (r + m)
    g = (g + m)
    b = (b + m)
    return (r, g, b)


def get_complementary_colors(seed, num_colors):
    random.seed(seed)
    h = random.randint(0, 360)
    s, l = 0.7, 0.5
    colors = []
    for i in range(num_colors):
        hue = (h + (i * (360 // num_colors))) % 360
        colors.append(hsl_to_rgb(hue, s, l))
    return colors


def draw_hex(ax, x_center, y_center, size, color='black'):
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = x_center + size * np.cos(angles)
    y_hex = x_center + size * np.sin(angles)
    ax.plot(x_hex, y_hex, color=color)


def plot_maze(grid, starts, exits, all_visited_sets, colors):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            x_center = col * 1.5 * size
            y_center = row * np.sqrt(3) * size + (col % 2) * (np.sqrt(3) / 2) * size
            cell = grid[row][col]
            for direction, wall in enumerate(cell.walls):
                if wall:
                    x0, y0 = x_center, y_center
                    x1, y1 = x_center + size * np.cos(np.pi / 3 * direction), y_center + size * np.sin(
                        np.pi / 3 * direction)
                    color = 'black'
                    for i, visited_set in enumerate(all_visited_sets):
                        if (row, col) in visited_set:
                            color = colors[i]
                            break
                    ax.plot([x0, x1], [y0, y1], color=color)

    # Mark entrances and exits
    def mark_point(ax, x, y, label, color):
        x_center = y * 1.5 * size
        y_center = x * np.sqrt(3) * size + (y % 2) * (np.sqrt(3) / 2) * size
        ax.text(x_center, y_center, label, color=color, ha='center', va='center', fontsize=12, fontweight='bold')

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, (start, exit) in enumerate(zip(starts, exits)):
        mark_point(ax, start[0], start[1], labels[i], colors[i])
        mark_point(ax, exit[0], exit[1], labels[i], colors[i])

    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate intertwined mazes.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows in the grid")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns in the grid")
    parser.add_argument("--num_mazes", type=int, default=3, help="Number of intertwined mazes")
    parser.add_argument("--seed", type=int, default=random.randint(0, sys.maxsize), help="Random seed for maze generation")
    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    num_mazes = args.num_mazes
    seed = args.seed
    colors = get_complementary_colors(seed, num_mazes)
    grid, starts, exits, all_visited_sets = create_intertwined_mazes(rows, cols, num_mazes, seed)
    plot_maze(grid, starts, exits, all_visited_sets, colors)


if __name__ == "__main__":
    main()
