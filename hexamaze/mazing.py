from dataclasses import dataclass, field
from typing import List, Dict, Optional, NamedTuple
from collections.abc import Generator
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# Spacing between hexagons
SPACING = 1


@dataclass
class Cell:
    """
    Represents a single hexagonal cell in a maze.
    
    Attributes:
        walls (List[bool]): A list of six boolean values indicating the presence of walls.
        set (Optional[str]): An optional identifier to group cells.
    """
    walls: List[bool] = field(default_factory=lambda: [True] * 6)
    set: Optional[str] = None


class HexCoordinates(NamedTuple):
    q: int
    r: int


class ScreenCoordinates(NamedTuple):
    x: int
    y: int


class RGBColor(NamedTuple):
    red: int
    green: int
    blue: int


@dataclass
class Grid:
    starts: Dict[int, HexCoordinates] = field(default_factory=dict)
    exits: Dict[int, HexCoordinates] = field(default_factory=dict)
    cells: Dict[HexCoordinates, Cell] = field(default_factory=dict)

    def __getitem__(self, index):
        return self.cells[index]

    def __setitem__(self, index, value):
        self.cells[index] = value

    def __iter__(self):
        return iter(self.cells)

    def items(self):
        return self.cells.items()


# Hexagonal directions
HEX_DIRECTIONS = [
    HexCoordinates(+1, 0), HexCoordinates(+1, -1), HexCoordinates(0, -1), HexCoordinates(-1, 0), HexCoordinates(-1, +1),
    HexCoordinates(0, +1)
]

# Colors for sides
SIDE_COLORS = ['Blue', 'Green', 'Orange', 'Purple', 'Pink', 'Yellow']

"""
Create a set of intertwined mazes on a hexagonal grid.

Args:
- size: An integer defining the radius of the hexagon grid.
- num_mazes: An integer specifying the number of mazes to generate.
- seed: An optional integer seed for random number generation.

Returns:
- A tuple containing:
    - A dictionary representing the hexagonal grid with maze cells.
    - A list of starting points for each maze.
    - A list of exit points for each maze.
"""


def create_intertwined_mazes(size: int, num_mazes: int, seed: int = None):
    if seed is not None:
        random.seed(seed)

    grid = initialize_hexagon_grid(size)

    # Define random starting points on the border of the maze

    for i in range(num_mazes):
        start_cell = get_random_border_point(grid, exclude_points=grid.starts)
        grid.starts[i] = start_cell

    # Initialize maze generators
    mazes = [generate_maze(grid, i) for i in range(num_mazes)]

    # Alternate steps between the mazes
    while mazes:
        for i, maze in enumerate(mazes[:]):
            try:
                next(maze)
            except StopIteration:
                mazes.remove(maze)

    return grid


def initialize_hexagon_grid(size: int) -> Grid:
    """
    Create a hexagonal grid of cells.

    Args:
    - size: An integer defining the radius of the hexagon grid.

    Returns:
    - A dictionary where each key is a tuple representing the coordinates (q, r) of a cell in the hexagonal grid,
      and each value is an instance of the Cell class.
    """
    grid = Grid()
    for q in range(-size, size + 1):
        r1 = max(-size, -q - size)
        r2 = min(size, -q + size)
        for r in range(r1, r2 + 1):
            grid[HexCoordinates(q, r)] = Cell()
    return grid


def generate_maze(grid: Grid, current_set_index: int) -> Generator[None, None, None]:
    """
    Generate a maze on a hexagonal grid using depth-first search algorithm.

    Args:
    - grid: A dictionary representing the hexagonal grid.
    - start_cell: The coordinates of the starting cell.
    - current_set_index: An identifier for the current path or set in the maze.

    Yields:
    - None
    """
    start_cell = grid.starts[current_set_index]
    stack = [(start_cell, 0)]
    grid[start_cell].set = current_set_index
    stack_max = (start_cell, 0)

    while stack:
        stack_index = random.randrange(len(stack))
        hex_coordinates, depth = stack[stack_index]
        neighbors = []
        added = False

        for i, hex_direction in enumerate(HEX_DIRECTIONS):
            coordinates_neighbour = HexCoordinates(
                hex_coordinates.q + hex_direction.q, hex_coordinates.r + hex_direction.r)
            if is_valid_move(coordinates_neighbour, grid):
                neighbors.append((coordinates_neighbour, i))

        if neighbors:
            coordinates_neighbour, direction = random.choice(neighbors)
            if not grid[coordinates_neighbour].set is None:
                continue

            # Proceed with valid neighbor and update the walls and visited set
            stack.append((coordinates_neighbour, depth + 1))
            grid[hex_coordinates].walls[direction] = False
            grid[coordinates_neighbour].walls[(direction + 3) % 6] = False  # Opposite wall
            grid[coordinates_neighbour].set = current_set_index
            added = True
        else:
            if stack_max and depth > stack_max[1]:
                stack_max = (hex_coordinates, depth)
                grid.exits[current_set_index] = hex_coordinates
            stack.pop(stack_index)
        if added:
            yield


def is_valid_move(hex_coordinates: HexCoordinates, grid: Grid) -> bool:
    """
    Check if a move to a specified cell in a hexagonal grid is valid
    based on the cell's presence in the grid and its visited status.

    Args:
        hex_coordinates: The coordinate of the cell in the hexagonal grid.
        grid : A hexagonal grid where keys are coordinates tuples (q, r) and values are Cell instances.

    Returns:
        bool: True if the move to the cell is valid, False otherwise.
    """
    return hex_coordinates in grid and (grid[hex_coordinates].set is None)


def get_random_border_point(grid, exclude_points=None):
    if not exclude_points:
        exclude_points = []
    border_points = [point for point in grid if
                     len([True for hex_direction in HEX_DIRECTIONS if
                          HexCoordinates(point.q + hex_direction.q, point.r + hex_direction.r) not in grid]) > 0]
    point = None

    while not point or point in exclude_points:
        point = random.choice(border_points)
    return point


def get_random_point_in_set(grid, exclude_points=None, current_set=None):
    if not exclude_points:
        exclude_points = []
    points_in_set = [point for point in grid if grid[point].set == current_set and point not in exclude_points]
    return random.choice(points_in_set) if points_in_set else None


def get_complementary_colors(seed: int, num_colors: int) -> List[RGBColor]:
    """
    Generates a list of RGB color values that are evenly spaced around the color wheel.

    Args:
        seed (int): An integer to initialize the random number generator for reproducibility.
        num_colors (int): The number of complementary colors to generate.

    Returns:
        List[RGBColor]: A list of RGB color tuples, each representing a complementary color.
    """
    random.seed(seed)
    hue_start = random.randint(0, 360)
    saturation, lightness = .70, .50
    colors = []

    for i in range(num_colors):
        hue = (hue_start + (i * (360 // num_colors))) % 360
        colors.append(hsl_to_rgb(hue, saturation, lightness))

    return colors


def hsl_to_rgb(hue: float, saturation: float, lightness: float) -> RGBColor:
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

    return RGBColor(red + m, green + m, blue + m)


def plot_maze(grid, colors, solutions=None, debug=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for hex_coordinates, cell in grid.items():
        screen_coordinates: ScreenCoordinates = ScreenCoordinates(
            size * 3 / 2 * hex_coordinates.q * SPACING,
            - size * np.sqrt(3) * (hex_coordinates.r + hex_coordinates.q / 2) * SPACING)
        if grid[hex_coordinates].set is not None:
            draw_hex(ax, hex_coordinates, screen_coordinates, size,
                     fill_color=colors[grid[hex_coordinates].set], debug=debug)
        else:
            draw_hex(ax, hex_coordinates, screen_coordinates, size, debug=debug)
        for direction, wall in enumerate(cell.walls):
            if wall:
                x0, y0 = screen_coordinates.x + size * np.cos(
                    np.pi / 3 * ((direction - 1) % 6)), screen_coordinates.y + size * np.sin(
                    np.pi / 3 * ((direction - 1) % 6))
                x1, y1 = screen_coordinates.x + size * np.cos(
                    np.pi / 3 * ((direction - 0) % 6)), screen_coordinates.y + size * np.sin(
                    np.pi / 3 * ((direction - 0) % 6))
                if SPACING > 1:
                    ax.plot([x0, x1], [y0, y1], color=SIDE_COLORS[direction])
                else:
                    ax.plot([x0, x1], [y0, y1], color='black')

    # Mark entrances and exits
    def mark_point(axis, point, label, color):
        point_center = ScreenCoordinates(size * 3 / 2 * point.q * SPACING,
                                         - size * np.sqrt(3) * (point.r + point.q / 2) * SPACING)
        factor = .8
        if debug:
            axis.text(*point_center, label, color='gray', ha='center', va='center', fontsize=7,
                      fontweight='bold',
                      bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))
        else:
            axis.arrow(point_center.x - ((size / 2) * factor), point_center.y - ((size / 2) * factor), size * factor,
                       size * factor, color='grey')
            axis.arrow(point_center.x + ((size / 2) * factor), point_center.y - ((size / 2) * factor), -size * factor,
                       size * factor, color='grey')

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # noqa -> Not a word
    for i, grid_start in grid.starts.items():
        mark_point(ax, grid_start, f'{labels[i]}->', colors[i])

    for i, grid_exit in grid.exits.items():
        mark_point(ax, grid_exit, f'->{labels[i]}', colors[i])

    # Draw solution paths
    if solutions:
        for i, solution in enumerate(solutions):
            for hex_coordinates, hex_next in zip(solution, solution[1:]):
                x0 = size * 3 / 2 * hex_coordinates.q * SPACING
                y0 = - size * np.sqrt(3) * (hex_coordinates.r + hex_coordinates.q / 2) * SPACING
                x1 = size * 3 / 2 * hex_next.q * SPACING
                y1 = - size * np.sqrt(3) * (hex_next.r + hex_next.q / 2) * SPACING
                ax.plot([x0, x1], [y0, y1], color="grey", linestyle='--')

    plt.axis('off')
    plt.show()


def draw_hex(axis, coordinates: HexCoordinates, screen_coordinates: ScreenCoordinates,
             size, fill_color=None, debug=False):
    """
    Draw a hexagon on a matplotlib axis.

    Parameters:
    - ax: The matplotlib axis on which the hexagon will be drawn.
    - coordinates: The axial coordinates of the hexagon, used for debugging text.
    - screen_coordinates: The x and y coordinates for the center of the hexagon.
    - size: The radius of the hexagon.
    - fill_color: Optional. The color used to fill the hexagon.
    - debug: Optional. If True, displays the axial coordinates on the hexagon.
    """
    angles = np.linspace(0, 2 * np.pi, 7)
    coord_hex = ScreenCoordinates(screen_coordinates.x + size * np.cos(angles) * (1 / SPACING),
                                  screen_coordinates.y + size * np.sin(angles) * (1 / SPACING))
    if SPACING > 1:
        axis.plot(*coord_hex, color='lightgray')  # noqa -> Color Name is spelled that way
    if fill_color:
        axis.fill(*coord_hex, color=fill_color)
    if debug:
        axis.text(*screen_coordinates, f'{coordinates.q},{coordinates.r}',
                  color='gray', ha='center', va='center', fontsize=5)


def find_solution(grid: Grid, grid_start: HexCoordinates, grid_exit: HexCoordinates) -> List[HexCoordinates]:
    """
    Find a path from grid_start to grid_exit in a hexagonal grid maze.

    Args:
        grid: A dictionary representing the hexagonal grid.
        grid_start: Starting point coordinates.
        grid_exit: Exit point coordinates.

    Returns:
        List of Coordinates representing the path from grid_start to grid_exit. Empty list if no path is found.
    """
    stack = [(grid_start, [grid_start])]
    traversed = {grid_start}

    while stack:
        hex_coordinates, path = stack.pop()
        if hex_coordinates == grid_exit:
            return path

        for hex_direction in HEX_DIRECTIONS:
            hex_neighbour = HexCoordinates(hex_coordinates.q + hex_direction.q, hex_coordinates.r + hex_direction.r)
            if (hex_neighbour in grid and not grid[hex_neighbour].walls[(HEX_DIRECTIONS.index(hex_direction) + 3) % 6]
                    and hex_neighbour not in traversed):
                traversed.add(hex_neighbour)
                stack.append((hex_neighbour, path + [hex_neighbour]))

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
    debug = args.debug  # or True

    solutions = None

    if seed is None:
        seed = random.randrange(0, 2 ** 32)
        print(f'{seed=}')

    colors = get_complementary_colors(seed, num_mazes)
    grid = create_intertwined_mazes(size, num_mazes, seed)
    if debug:
        solutions = [find_solution(grid, grid.starts[i], grid.exits[i]) for i in range(num_mazes)]
    plot_maze(grid, colors, solutions, debug)


if __name__ == "__main__":
    main()
