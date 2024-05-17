import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

HEX_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)
]


class Cell:
    def __init__(self):
        self.walls = [True] * 6  # Each cell has 6 walls initially
        self.visited = False


def initialize_hexagon_grid(size):
    grid = {}
    for q in range(-size, size + 1):
        r1 = max(-size, -q - size)
        r2 = min(size, -q + size)
        for r in range(r1, r2 + 1):
            grid[(q, r)] = Cell()
    return grid


def is_valid_move(q, r, grid, all_visited_sets, current_set_index):
    if (q, r) in grid and not grid[(q, r)].visited:
        for i, visited_set in enumerate(all_visited_sets):
            if i != current_set_index and (q, r) in visited_set:
                return False
        return True
    return False


def generate_maze(grid, start_q, start_r, all_visited_sets, current_set_index):
    stack = [(start_q, start_r)]
    grid[(start_q, start_r)].visited = True

    while stack:
        q, r = stack[-1]
        neighbors = []

        for i, (dq, dr) in enumerate(HEX_DIRECTIONS):
            nq, nr = q + dq, r + dr
            if is_valid_move(nq, nr, grid, all_visited_sets, current_set_index):
                neighbors.append((nq, nr, i))

        if neighbors:
            nq, nr, direction = random.choice(neighbors)
            grid[(nq, nr)].visited = True
            stack.append((nq, nr))

            # Remove wall between current cell and chosen neighbor
            grid[(q, r)].walls[direction] = False
            grid[(nq, nr)].walls[(direction + 3) % 6] = False  # Opposite wall

            # Add the cell to the current visited set
            all_visited_sets[current_set_index].add((nq, nr))
        else:
            stack.pop()

        yield


def get_random_border_point(grid, exclude_points=[]):
    border_points = [point for point in grid if
                     len([True for dq, dr in HEX_DIRECTIONS if (point[0] + dq, point[1] + dr) not in grid]) > 0]
    point = None
    while not point or point in exclude_points:
        point = random.choice(border_points)
    return point


def create_intertwined_mazes(size, num_mazes, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = initialize_hexagon_grid(size)

    # Define random starting points on the border of the maze
    starts = []
    for _ in range(num_mazes):
        start = get_random_border_point(grid, exclude_points=starts)
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

    # Define random exit points within the visited sets
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
    y_hex = y_center + size * np.sin(angles)
    ax.plot(x_hex, y_hex, color=color)


def plot_maze(grid, starts, exits, all_visited_sets, colors):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for (q, r), cell in grid.items():
        x_center = size * 3 / 2 * q
        y_center = size * np.sqrt(3) * (r + q / 2)
        for direction, wall in enumerate(cell.walls):
            if wall:
                x0, y0 = x_center, y_center
                x1, y1 = x_center + size * np.cos(np.pi / 3 * direction), y_center + size * np.sin(
                    np.pi / 3 * direction)
                color = 'black'
                for i, visited_set in enumerate(all_visited_sets):
                    if (q, r) in visited_set:
                        color = colors[i]
                        break
                ax.plot([x0, x1], [y0, y1], color=color)

    # Mark entrances and exits
    def mark_point(ax, q, r, label, color):
        x_center = size * 3 / 2 * q
        y_center = size * np.sqrt(3) * (r + q / 2)
        ax.text(x_center, y_center, label, color=color, ha='center', va='center', fontsize=12, fontweight='bold')

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, (start, exit) in enumerate(zip(starts, exits)):
        mark_point(ax, start[0], start[1], labels[i], colors[i])
        mark_point(ax, exit[0], exit[1], labels[i], colors[i])

    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate intertwined mazes.")
    parser.add_argument("--size", type=int, default=5, help="Size of the hexagonal grid")
    parser.add_argument("--num_mazes", type=int, default=3, help="Number of intertwined mazes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for maze generation")
    args = parser.parse_args()

    size = args.size
    num_mazes = args.num_mazes
    seed = args.seed

    colors = get_complementary_colors(seed, num_mazes)
    grid, starts, exits, all_visited_sets = create_intertwined_mazes(size, num_mazes, seed)
    plot_maze(grid, starts, exits, all_visited_sets, colors)


if __name__ == "__main__":
    main()
