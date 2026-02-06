"""
ArrayGen
1.3
~ Aarin J
"""

# imports ----------------------
import src.ArrayGen as q
import unittest
from typing import Tuple, Union
import re

# ----------------------------------------------------------------------------------------------------------------------
def get_num(expr: str, var: int = None) -> Union[Tuple[int, ...], str]:
    """
    Extracts specified number of integers from the given string expression.

    Parameters
    ----------
    expr: str
        The input string containing numbers.
    var: int
        Number of terms to extract.

    Returns
    -------
    tuple of ints
        Extracted integers from the string.
    """
    if expr == 'help':
        _get_num_help = """
        This is a function to extract all of the numbers in any string.
        ---------------------------------------------------------------
        Here is how it goes:
            _get_num('2i + 5j -6k')
            >> this will return 2, 5 and -6
        """
        return _get_num_help
    no = re.findall(r'[+-]?\s*\d+', expr)
    no = [int(num.replace(' ', '')) for num in no]

    if var is not None:
        if len(no) < var:
            raise ValueError(f'Not enough numbers found in the expression, required {var}!')
        return tuple(no[:var])
    return tuple(no)



class TestGetNum(unittest.TestCase):
    def test_basic_positive(self):
        self.assertEqual(get_num('2i + 5j -6k'), (2, 5, -6))

    def test_numbers_with_spaces(self):
        self.assertEqual(get_num('a = +4, b = -78'), (4, -78))

    def test_with_var_parameter(self):
        self.assertEqual(get_num('2i + 5j -6k', var=2), (2, 5))

    def test_var_too_large(self):
        with self.assertRaises(ValueError):
            get_num('42 is nice', var=3)

    def test_empty_string(self):
        self.assertEqual(get_num(''), ())

if __name__ == '__main__':
    unittest.main()


# ----------------------------------------------------------------------------------------------------------------------
