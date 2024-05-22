from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections.abc import Generator
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


@dataclass
class Cell:
    """
    Represents a single hexagonal cell in a maze.
    
    Attributes:
        walls (List[bool]): A list of six boolean values indicating the presence of walls.
        visited (bool): Indicates whether the cell has been visited.
        set (Optional[str]): An optional identifier to group cells.
    """
    walls: List[bool] = field(default_factory=lambda: [True] * 6)
    visited: bool = False
    set: Optional[str] = None


def initialize_hexagon_grid(size: int) -> Dict[tuple, Cell]:
    """
    Create a hexagonal grid of cells.

    Args:
    - size: An integer defining the radius of the hexagon grid.

    Returns:
    - A dictionary where each key is a tuple representing the coordinates (q, r) of a cell in the hexagonal grid,
      and each value is an instance of the Cell class.
    """
    grid = {}
    for q in range(-size, size + 1):
        r1 = max(-size, -q - size)
        r2 = min(size, -q + size)
        for r in range(r1, r2 + 1):
            grid[(q, r)] = Cell()
    return grid


def is_valid_move(q: int, r: int, grid: Dict[tuple, Cell]) -> bool:
    """
    Check if a move to a specified cell in a hexagonal grid is valid based on the cell's presence in the grid and its visited status.

    Args:
        q (int): The q-coordinate of the cell in the hexagonal grid.
        r (int): The r-coordinate of the cell in the hexagonal grid.
        grid (Dict[tuple, Cell]): A dictionary representing the hexagonal grid where keys are coordinate tuples (q, r) and values are Cell instances.

    Returns:
        bool: True if the move to the cell is valid, False otherwise.
    """
    return (q, r) in grid and not grid[(q, r)].visited


def generate_maze(grid: Dict[Tuple[int, int], Cell], start_q: int, start_r: int, current_set_index: int) -> Generator[None, None, None]:
    """
    Generate a maze on a hexagonal grid using depth-first search algorithm.

    Args:
    - grid: A dictionary representing the hexagonal grid.
    - start_q: The q-coordinate of the starting cell.
    - start_r: The r-coordinate of the starting cell.
    - current_set_index: An identifier for the current path or set in the maze.

    Yields:
    - None
    """
    stack = [(start_q, start_r)]
    grid[(start_q, start_r)].visited = True
    grid[(start_q, start_r)].set = current_set_index

    while stack:
        q, r = stack[-1]
        neighbors = []
        added = False

        for i, (dq, dr) in enumerate(HEX_DIRECTIONS):
            nq, nr = q + dq, r + dr
            if is_valid_move(nq, nr, grid):
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


def hsl_to_rgb(hue: float, saturation: float, lightness: float) -> Tuple[float, float, float]:
    """
    Convert a color from HSL (Hue, Saturation, Lightness) format to RGB (Red, Green, Blue) format.

    Args:
        hue (float): The hue component of the color (0-360).
        saturation (float): The saturation of the color (0-1).
        lightness (float): The lightness of the color (0-1).

    Returns:
        Tuple[float, float, float]: RGB components of the color scaled between 0 and 1.
    """
    c = (1 - abs(2 * lightness - 1)) * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness - c / 2

    hue_sector = int(hue // 60) % 6
    rgb_order = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    red, green, blue = rgb_order[hue_sector]

    return red + m, green + m, blue + m


def get_complementary_colors(seed: int, num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generates a list of RGB color values that are evenly spaced around the color wheel.

    Args:
        seed (int): An integer to initialize the random number generator for reproducibility.
        num_colors (int): The number of complementary colors to generate.

    Returns:
        List[Tuple[int, int, int]]: A list of RGB color tuples, each representing a complementary color.
    """
    random.seed(seed)
    hue_start = random.randint(0, 360)
    saturation, lightness = .70, .50
    colors = []
    
    for i in range(num_colors):
        hue = (hue_start + (i * (360 // num_colors))) % 360
        colors.append(hsl_to_rgb(hue, saturation, lightness))
    
    return colors


def draw_hex(ax, q, r, x_center, y_center, size, fill_color=None, debug=False):
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = x_center + size * np.cos(angles) * (1/SPACING)
    y_hex = y_center + size * np.sin(angles) * (1/SPACING)
    if SPACING > 1:
        ax.plot(x_hex, y_hex, color='lightgray') # noqa -> Color Name is spelled that way
    if fill_color:
        ax.fill(x_hex, y_hex, color=fill_color)
    if debug:
        ax.text(x_center, y_center, f'{q},{r}', color='gray', ha='center', va='center', fontsize=5)


def plot_maze(grid, starts, exits, colors, solutions=None, debug=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for (q, r), cell in grid.items():
        x_center = size * 3 / 2 * q * SPACING
        y_center = - size * np.sqrt(3) * (r + q / 2) * SPACING
        if grid[(q, r)].set is not None:
            draw_hex(ax, q, r, x_center, y_center, size, fill_color=colors[grid[(q, r)].set],debug=debug)
        else:
            draw_hex(ax, q, r, x_center, y_center, size,debug=debug)
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
    def mark_point(point_ax, point_q, point_r, label, color):
        point_x_center = size * 3 / 2 * point_q * SPACING
        point_y_center = - size * np.sqrt(3) * (point_r + point_q / 2) * SPACING
        factor = .8
        if debug:
            point_ax.text(point_x_center, point_y_center, label, color='gray', ha='center', va='center', fontsize=7,
                      fontweight='bold',
                      bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))
        else:
            point_ax.arrow(point_x_center - ((size / 2) * factor), point_y_center - ((size / 2) * factor), size * factor, size * factor, color='grey')
            point_ax.arrow(point_x_center + ((size / 2) * factor), point_y_center - ((size / 2) * factor), -size * factor, size * factor, color='grey')

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # noqa -> Not a word
    for i, (grid_start, grid_exit) in enumerate(zip(starts, exits)):
        mark_point(ax, grid_start[0], grid_start[1], labels[i], colors[i])
        mark_point(ax, grid_exit[0], grid_exit[1], labels[i], colors[i])

    # Draw solution paths
    if solutions:
        for i, solution in enumerate(solutions):
            for (q, r), (nq, nr) in zip(solution, solution[1:]):
                x0 = size * 3 / 2 * q * SPACING
                y0 = - size * np.sqrt(3) * (r + q / 2) * SPACING
                x1 = size * 3 / 2 * nq * SPACING
                y1 = - size * np.sqrt(3) * (nr + nq / 2) * SPACING
                ax.plot([x0, x1], [y0, y1], color="grey", linestyle='--')

    plt.axis('off')
    plt.show()


def find_solution(grid, grid_start, grid_exit):
    stack = [(grid_start, [grid_start])]
    visited = {grid_start}

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
    parser.add_argument("--debug", action='store_true', help="Adds some debug output")
    args = parser.parse_args()

    size = args.size
    num_mazes = args.num_mazes
    seed = args.seed
    debug = args.debug

    solutions = None

    if seed is None:
        seed = random.randrange(0, 2 ** 32)
        print(f'{seed=}')

    colors = get_complementary_colors(seed, num_mazes)
    grid, starts, exits = create_intertwined_mazes(size, num_mazes, seed)
    if debug:
        solutions = [find_solution(grid, grid_start, grid_exit) for grid_start, grid_exit in zip(starts, exits)]
    plot_maze(grid, starts, exits, colors, solutions, debug)


if __name__ == "__main__":
    main()
