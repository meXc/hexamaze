
from hexamaze.mazing import get_complementary_colors


class TestGetComplementaryColors:

    # Verify that the function returns a list of tuples
    def test_returns_list_of_tuples(self):
        result = get_complementary_colors(5, 42)
        assert isinstance(result, list), "Result should be a list"
        assert all(isinstance(color, tuple) for color in result), "All items in the list should be tuples"
        assert all(len(color) == 3 for color in result), "Each tuple should have three elements"

    # Test with 'num_colors' set to 1 to see if it returns a single color
    def test_single_color_return(self):
        result = get_complementary_colors(1, 123)
        assert len(result) == 1, "Should return exactly one color tuple"
        assert isinstance(result[0], tuple), "The single item should be a tuple"
        assert len(result[0]) == 3, "The tuple should contain three elements"
        assert isinstance(result[0][0], float), "Red value should be a float"
        assert isinstance(result[0][1], float), "Green value should be a float"
        assert isinstance(result[0][2], float), "Blue value should be a float"
        assert 0 <= result[0][0] <= 1, "Red value should be within 0-1 range"
        assert 0 <= result[0][1] <= 1, "Green value should be within 0-1 range"
        assert 0 <= result[0][2] <= 1, "Blue value should be within 0-1 range"

    # Ensure that no duplicate color tuples are generated when generating 5 colors
    def test_no_duplicate_colors(self):
        colors = get_complementary_colors(42, 5)
        assert len(colors) == len(set(colors)), "Duplicate color tuples found"
