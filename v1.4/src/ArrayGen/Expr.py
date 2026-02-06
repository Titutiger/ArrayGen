import numpy as np
import re
from typing import Callable, Any, Union, Tuple, List

def expr(expression: Callable[..., float], loops: int = 10, negative: bool = True,
              var: int = 1, **kwargs: Any) -> Union[Tuple[List[float], List[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Evaluates a mathematical expression with one or two variables over a specified range.
    This function evaluates the given callable expression over a range of integer input values.
    It supports either a singular variable (1D) or two variables (2D mesh grid) evaluation.
    The results include error handling to ensure safe evaluation, setting invalid called to 1.0.

    Parameters
    ----------
    expression: Callable[..., float]
        A function representing the mathematical expression to evaluate.
        Should accept one or two float arguments depending on `var`.
    loops: int, optional
        Maximum absolute value for the range of input values (default is 10).
        For `var=1`, range is from -loops to loops if `negative` is True, else from 0 to loops.
    negative: bool, optional
        Whether to include negative numbers in the input ranges when `var=1` (default is True).
    var: int, optional
        Number of variables in the expression: 1 or 2 (default is 1).
    **kwargs: Any
        Additional keyword arguments, supports:
        - `x_range`: tuple (int, int) specifying custom range for x when `var=2`.
        - `y_range`: tuple (int, int) specifying custom range for y when `var=2`.

    Returns
    -------
    Union[Tuple[List[float], List[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        - If `var=1`: Tuple of (x_values, evaluated_results) as lists.
        - If `var=2`: Typle of meshgrid arrays (X, Y) and evaluated results Z (all numpy arrays).

    Raises
    ------
    ValueError
        If `var` is not 1 or 2.

    Examples
    --------
    Evaluate a 1D expression (e.g., square function) from -5 to 5:
    >>> import ArrayGen.ArrayGen as q
    >>> square = lambda x: x**2
    >>> x_vals, results = q.expr(square, loops=5, negative=True, var=1)
    >>> print(x_vals)
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    >>> print(results)
    [25, 16, 9, 4, 1, 1.0, 1, 4, 9, 16, 25]

    Evaluate a 2D expression (e.g., multiplication) over a custom range:
    >>> multiply = lambda x, y: x * y
    >>> X, Y, Z = q.expr(multiply, var=2, loops=2, x_range=(-2, 2), y_range=(0, 3))
    >>> print(Z)
    [[ 0.  0.  0.  0.  0.]
     [ 0.  1.  2.  3.  4.]
     [ 0.  2.  4.  6.  8.]
     [ 0.  3.  6.  9. 12.]]

    """
    if var == 1:
        def safe_eval(expr_func, xs):
            results = []
            for x in xs:
                try:
                    val = expr_func(x)
                except Exception as e:
                    print(f'Warning at x={x}: {e}; setting value=1')
                    val = 1.0
                results.append(val)
            return results

        if negative:
            x_vals = list(range(-loops, loops + 1))
        else:
            x_vals = list(range(loops + 1))
        results = safe_eval(expression, x_vals)
        if results:
            results[0] = 1.0
        return x_vals, results

    elif var == 2:
        # Optionally, accept custom ranges for x and y via kwargs
        x_range = kwargs.get("x_range", (-loops, loops))
        y_range = kwargs.get("y_range", (-loops, loops))
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_vals = np.arange(y_range[0], y_range[1] + 1)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                try:
                    Z[i, j] = expression(x, y)
                except Exception as e:
                    print(f"Warning at (x={x}, y={y}): {e}; setting val=1")
                    Z[i, j] = 1.0
        return X, Y, Z

    else:
        raise ValueError("Only var=1 or var=2 is supported.")
