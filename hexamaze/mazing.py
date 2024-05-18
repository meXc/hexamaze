import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# Hexagonal directions
HEX_DIRECTIONS = [
    (+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)
]

# Colors for sides
SIDE_COLORS = ['Blue', 'Green', 'Orange', 'Purple', 'Pink', 'Yellow']

# Spacing between hexagons
SPACING = 1


class Cell:
    def __init__(self):
        self.walls = [True] * 6  # Each cell has 6 walls initially
        self.visited = False
        self.set = None


def initialize_hexagon_grid(size):
    grid = {}
    for q in range(-size, size + 1):
        r1 = max(-size, -q - size)
        r2 = min(size, -q + size)
        for r in range(r1, r2 + 1):
            grid[(q, r)] = Cell()
    return grid


def is_valid_move(q, r, grid, current_set_index):
    if (q, r) in grid and not grid[(q, r)].visited:
        return True
    return False


def generate_maze(grid, start_q, start_r, current_set_index):
    stack = [(start_q, start_r)]
    grid[(start_q, start_r)].visited = True
    grid[(start_q, start_r)].set = current_set_index

    while stack:
        q, r = stack[-1]
        neighbors = []
        added = False

        for i, (dq, dr) in enumerate(HEX_DIRECTIONS):
            nq, nr = q + dq, r + dr
            if is_valid_move(nq, nr, grid, current_set_index):
                neighbors.append((nq, nr, i))

        if neighbors:
            nq, nr, direction = random.choice(neighbors)

            # Proceed with valid neighbor and update the walls and visited set
            grid[(nq, nr)].visited = True
            stack.append((nq, nr))
            grid[(q, r)].walls[direction] = False
            grid[(nq, nr)].walls[(direction + 3) % 6] = False  # Opposite wall
            grid[(nq, nr)].set = current_set_index
            added = True
        else:
            stack.pop()
        if added:
            yield


def get_random_border_point(grid, exclude_points=None, current_set=None):
    if not exclude_points:
        exclude_points = []
    border_points = [point for point in grid if
                     len([True for dq, dr in HEX_DIRECTIONS if (point[0] + dq, point[1] + dr) not in grid]) > 0]
    point = None
    wrong_set = True
    while not point or point in exclude_points or wrong_set:
        point = random.choice(border_points)
        if current_set is None:
            wrong_set = False
        else:
            wrong_set = (grid[(point[0], point[1])].set != current_set)
    return point


def get_random_point_in_set(grid, exclude_points=None, current_set=None):
    if not exclude_points:
        exclude_points = []
    points_in_set = [point for point in grid if grid[point].set == current_set and point not in exclude_points]
    return random.choice(points_in_set) if points_in_set else None


def create_intertwined_mazes(size, num_mazes, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = initialize_hexagon_grid(size)

    # Define random starting points on the border of the maze
    starts = []
    for _ in range(num_mazes):
        start = get_random_border_point(grid, exclude_points=starts)
        starts.append(start)

    # Initialize maze generators
    mazes = [generate_maze(grid, start[0], start[1], i) for i, start in enumerate(starts)]

    # Alternate steps between the mazes
    while mazes:
        for i, maze in enumerate(mazes[:]):
            try:
                next(maze)
            except StopIteration:
                mazes.remove(maze)

    # Define random exit points within the sets, excluding start points
    exits = []
    for i in range(num_mazes):
        exit_point = get_random_point_in_set(grid, exclude_points=[starts[i]], current_set=i)
        exits.append(exit_point)

    return grid, starts, exits


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


def draw_hex(ax, q, r, x_center, y_center, size, color='black', fill_color=None):
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = x_center + size * np.cos(angles) * (1/SPACING)
    y_hex = y_center + size * np.sin(angles) * (1/SPACING)
    if SPACING > 1:
        ax.plot(x_hex, y_hex, color="lightgray")
    if fill_color:
        ax.fill(x_hex, y_hex, color=fill_color)
    ax.text(x_center, y_center, f'{q},{r}', color='gray', ha='center', va='center', fontsize=5)


def plot_maze(grid, starts, exits, colors, solutions):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for (q, r), cell in grid.items():
        x_center = size * 3 / 2 * q * SPACING
        y_center = - size * np.sqrt(3) * (r + q / 2) * SPACING
        if grid[(q, r)].set is not None:
            draw_hex(ax, q, r, x_center, y_center, size, color='black', fill_color=colors[grid[(q, r)].set])
        else:
            draw_hex(ax, q, r, x_center, y_center, size, color='black')
        for direction, wall in enumerate(cell.walls):
            if wall:
                x0, y0 = x_center + size * np.cos(np.pi / 3 * ((direction - 1) % 6)), y_center + size * np.sin(
                    np.pi / 3 * ((direction - 1) % 6))
                x1, y1 = x_center + size * np.cos(np.pi / 3 * ((direction - 0) % 6)), y_center + size * np.sin(
                    np.pi / 3 * ((direction - 0) % 6))
                if SPACING > 1:
                    ax.plot([x0, x1], [y0, y1], color=SIDE_COLORS[direction])
                else:
                    ax.plot([x0, x1], [y0, y1], color='black')

    # Mark entrances and exits
    def mark_point(ax, q, r, label, color):
        x_center = size * 3 / 2 * q * SPACING
        y_center = - size * np.sqrt(3) * (r + q / 2) * SPACING
        ax.text(x_center, y_center, label, color='gray', ha='center', va='center', fontsize=5, fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, (start, exit) in enumerate(zip(starts, exits)):
        mark_point(ax, start[0], start[1], labels[i], colors[i])
        mark_point(ax, exit[0], exit[1], labels[i], colors[i])

    # Draw solution paths
    for i, solution in enumerate(solutions):
        for (q, r), (nq, nr) in zip(solution, solution[1:]):
            x0 = size * 3 / 2 * q * SPACING
            y0 = - size * np.sqrt(3) * (r + q / 2) * SPACING
            x1 = size * 3 / 2 * nq * SPACING
            y1 = - size * np.sqrt(3) * (nr + nq / 2) * SPACING
            ax.plot([x0, x1], [y0, y1], color="grey", linestyle='--')

    plt.axis('off')
    plt.show()


def find_solution(grid, start, grid_exit):
    stack = [(start, [start])]
    visited = {start}

    while stack:
        (q, r), path = stack.pop()
        if (q, r) == grid_exit:
            return path

        for dq, dr in HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in grid and not grid[(nq, nr)].walls[(HEX_DIRECTIONS.index((dq, dr)) + 3) % 6] and (
                    nq, nr) not in visited:
                visited.add((nq, nr))
                stack.append(((nq, nr), path + [(nq, nr)]))

    return []


def main():
    parser = argparse.ArgumentParser(description="Generate intertwined mazes.")
    parser.add_argument("--size", type=int, default=10, help="Size of the hexagonal grid")
    parser.add_argument("--num_mazes", type=int, default=3, help="Number of intertwined mazes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for maze generation")
    args = parser.parse_args()

    size = args.size
    num_mazes = args.num_mazes
    seed = args.seed

    if seed is None:
        seed = random.randrange(0, 2 ** 32)
        print(f'{seed=}')

    colors = get_complementary_colors(seed, num_mazes)
    grid, starts, exits = create_intertwined_mazes(size, num_mazes, seed)
    solutions = [find_solution(grid, start, exit) for start, exit in zip(starts, exits)]
    plot_maze(grid, starts, exits, colors, solutions)


if __name__ == "__main__":
    main()
