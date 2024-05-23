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

# Hexagonal directions
HEX_DIRECTIONS = [
    HexCoordinates(+1, 0), HexCoordinates(+1, -1), HexCoordinates(0, -1), HexCoordinates(-1, 0), HexCoordinates(-1, +1), HexCoordinates(0, +1)
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
def create_intertwined_mazes(size: int, num_mazes:int, seed: int=None):
    if seed is not None:
        random.seed(seed)

    grid = initialize_hexagon_grid(size)

    # Define random starting points on the border of the maze
    starts = []
    for _ in range(num_mazes):
        start_cell = get_random_border_point(grid, exclude_points=starts)
        starts.append(start_cell)

    # Initialize maze generators
    mazes = [generate_maze(grid, start_cell, i) for i, start_cell in enumerate(starts)]

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


def initialize_hexagon_grid(size: int) -> Dict[HexCoordinates, Cell]:
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
            grid[HexCoordinates(q, r)] = Cell()
    return grid


def generate_maze(grid: Dict[HexCoordinates, Cell], start_cell: HexCoordinates, current_set_index: int) -> Generator[None, None, None]:
    """
    Generate a maze on a hexagonal grid using depth-first search algorithm.

    Args:
    - grid: A dictionary representing the hexagonal grid.
    - start_cell: The coordinates of the starting cell.
    - current_set_index: An identifier for the current path or set in the maze.

    Yields:
    - None
    """
    stack = [start_cell]
    grid[start_cell].set = current_set_index

    while stack:
        hex_coords = stack[-1]
        neighbors = []
        added = False

        for i, hex_direction in enumerate(HEX_DIRECTIONS):
            coords_neighbour = HexCoordinates(hex_coords.q + hex_direction.q, hex_coords.r + hex_direction.r)
            if is_valid_move(coords_neighbour, grid, current_set_index):
                neighbors.append((coords_neighbour, i))

        if neighbors:
            coords_neighbour, direction = random.choice(neighbors)
            # Proceed with valid neighbor and update the walls and visited set
            stack.append(coords_neighbour)
            grid[hex_coords].walls[direction] = False
            grid[coords_neighbour].walls[(direction + 3) % 6] = False  # Opposite wall
            grid[coords_neighbour].set = current_set_index
            added = True
        else:
            stack.pop()
        if added:
            yield


def is_valid_move(hex_coords: HexCoordinates, grid: Dict[HexCoordinates, Cell], set: int) -> bool:
    """
    Check if a move to a specified cell in a hexagonal grid is valid based on the cell's presence in the grid and its visited status.

    Args:
        q (int): The q-coordinate of the cell in the hexagonal grid.
        r (int): The r-coordinate of the cell in the hexagonal grid.
        grid (Dict[Coordinates, Cell]): A dictionary representing the hexagonal grid where keys are coordinate tuples (q, r) and values are Cell instances.

    Returns:
        bool: True if the move to the cell is valid, False otherwise.
    """
    return hex_coords in grid and (grid[hex_coords].set is None)


def get_random_border_point(grid, exclude_points=None):
    if not exclude_points:
        exclude_points = []
    border_points = [point for point in grid if
                     len([True for hex_direction in HEX_DIRECTIONS if HexCoordinates(point.q + hex_direction.q, point.r + hex_direction.r) not in grid]) > 0]
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

    return red + m, green + m, blue + m


def plot_maze(grid, starts, exits, colors, solutions=None, debug=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    size = 1
    for hex_coords, cell in grid.items():
        screen_coords : ScreenCoordinates = ScreenCoordinates(size * 3 / 2 * hex_coords.q * SPACING, - size * np.sqrt(3) * (hex_coords.r + hex_coords.q / 2) * SPACING)
        if grid[hex_coords].set is not None:
            draw_hex(ax, hex_coords, screen_coords, size, fill_color=colors[grid[hex_coords].set],debug=debug)
        else:
            draw_hex(ax, hex_coords, screen_coords, size,debug=debug)
        for direction, wall in enumerate(cell.walls):
            if wall:
                x0, y0 = screen_coords.x + size * np.cos(np.pi / 3 * ((direction - 1) % 6)), screen_coords.y + size * np.sin(
                    np.pi / 3 * ((direction - 1) % 6))
                x1, y1 = screen_coords.x + size * np.cos(np.pi / 3 * ((direction - 0) % 6)), screen_coords.y + size * np.sin(
                    np.pi / 3 * ((direction - 0) % 6))
                if SPACING > 1:
                    ax.plot([x0, x1], [y0, y1], color=SIDE_COLORS[direction])
                else:
                    ax.plot([x0, x1], [y0, y1], color='black')

    # Mark entrances and exits
    def mark_point(axis, point, label, color):
        point_center = ScreenCoordinates(size * 3 / 2 * point.q * SPACING, - size * np.sqrt(3) * (point.r + point.q / 2) * SPACING)
        factor = .8
        if debug:
            axis.text(*point_center, label, color='gray', ha='center', va='center', fontsize=7,
                      fontweight='bold',
                      bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))
        else:
            axis.arrow(point_center.x - ((size / 2) * factor), point_center.y - ((size / 2) * factor), size * factor, size * factor, color='grey')
            axis.arrow(point_center.x + ((size / 2) * factor), point_center.y - ((size / 2) * factor), -size * factor, size * factor, color='grey')

    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # noqa -> Not a word
    for i, (grid_start, grid_exit) in enumerate(zip(starts, exits)):
        mark_point(ax, grid_start, f'{labels[i]}->', colors[i])
        mark_point(ax, grid_exit, f'->{labels[i]}', colors[i])

    # Draw solution paths
    if solutions:
        for i, solution in enumerate(solutions):
            for hex_coords, hex_next in zip(solution, solution[1:]):
                x0 = size * 3 / 2 * hex_coords.q * SPACING
                y0 = - size * np.sqrt(3) * (hex_coords.r + hex_coords.q / 2) * SPACING
                x1 = size * 3 / 2 * hex_next.q * SPACING
                y1 = - size * np.sqrt(3) * (hex_next.r + hex_next.q / 2) * SPACING
                ax.plot([x0, x1], [y0, y1], color="grey", linestyle='--')

    plt.axis('off')
    plt.show()


def draw_hex(axis, coords: HexCoordinates, screen_coords: ScreenCoordinates, size, fill_color=None, debug=False):
    """
    Draw a hexagon on a matplotlib axis.

    Parameters:
    - ax: The matplotlib axis on which the hexagon will be drawn.
    - coords: The axial coordinates of the hexagon, used for debugging text.
    - screen_coords: The x and y coordinates for the center of the hexagon.
    - size: The radius of the hexagon.
    - fill_color: Optional. The color used to fill the hexagon.
    - debug: Optional. If True, displays the axial coordinates on the hexagon.
    """
    angles = np.linspace(0, 2 * np.pi, 7)
    coord_hex = ScreenCoordinates(screen_coords.x + size * np.cos(angles) * (1/SPACING), screen_coords.y + size * np.sin(angles) * (1/SPACING))
    if SPACING > 1:
        axis.plot(*coord_hex, color='lightgray') # noqa -> Color Name is spelled that way
    if fill_color:
        axis.fill(*coord_hex, color=fill_color)
    if debug:
        axis.text(*screen_coords, f'{coords.q},{coords.r}', color='gray', ha='center', va='center', fontsize=5)


def find_solution(grid: Dict[HexCoordinates, Cell], grid_start: HexCoordinates, grid_exit: HexCoordinates) -> List[HexCoordinates]:
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
        hex_coords, path = stack.pop()
        if hex_coords == grid_exit:
            return path

        for hex_direction in HEX_DIRECTIONS:
            hex_neighbour = HexCoordinates( hex_coords.q + hex_direction.q, hex_coords.r + hex_direction.r)
            if hex_neighbour in grid and not grid[hex_neighbour].walls[(HEX_DIRECTIONS.index(hex_direction) + 3) % 6] and hex_neighbour not in traversed:
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
