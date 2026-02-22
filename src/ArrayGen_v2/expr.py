# expr.py
import numpy as np
import sympy as sp

from typing import Union, Tuple


def expr(
    expr: str,
    type_: str = '(x,y)',
    vars_: str = "x",
    loops: int = 10,
    negative: bool = True,
    x_range: Tuple[int, int] | None = None,
    y_range: Tuple[int, int] | None = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Evaluate a mathematical expression or equation over a finite domain.

    This function supports:

    1. Single-variable expressions:
       Example:
           expr = "3*x"
       If loops=3 and negative=False:
           Domain:   0, 1, 2, 3
           Co-domain: 0, 3, 6, 9

    2. Two-variable expressions:
       Example:
           expr = "2*x + 4*y"
       Returns a mesh grid evaluation.

    3. Implicit equations:
       Example:
           expr = "x^2 + y^2 = 25"
       The equation is internally converted to:
           x^2 + y^2 - 25 = 0
       The function returns Z = F(X, Y).
       To plot the curve, use contour at Z = 0.

    The function automatically:
    - Converts '^' to '**'
    - Converts equations into zero-form
    - Evaluates safely using NumPy
    - Replaces invalid (NaN / inf) values with 1.0

    Parameters
    ----------
    expr : str
        Mathematical expression or equation.
        Examples:
            "3*x"
            "2*x + 4*y"
            "x^2 + y^2 = 25"

    type_: str
        Checks for command execution.
        If `type_` is `r` then it calls the
        root function.
        If `type_` is `xy` then it calls the
        evaluate/solve function.

    vars_ : str, optional
        Variable names separated by space.
        Examples:
            "x"
            "x y"

    loops : int, optional
        Maximum magnitude of domain if ranges not explicitly provided.
        If negative=True:
            domain = [-loops, loops]
        Else:
            domain = [0, loops]

    negative : bool, optional
        Whether domain should include negative values.

    x_range : tuple[int, int], optional
        Explicit range for x (used in 2-variable mode).

    y_range : tuple[int, int], optional
        Explicit range for y (used in 2-variable mode).

    Returns
    -------
    For 1 variable:
        (x_values, y_values)

    For 2 variables or equations:
        (X, Y, Z)
        where:
            X, Y = meshgrid arrays
            Z = evaluated function values

    Raises
    ------
    ValueError
        If invalid expression is provided
        If more than 2 variables are specified

    Notes
    -----
    - Invalid numerical results (NaN, ±inf) are replaced with 1.0.
    - For implicit equations, use contour plotting:
          plt.contour(X, Y, Z, levels=[0])
    """

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------

    expr = expr.replace("^", "**")  # allow caret exponent syntax

    var_names = vars_.split()
    if not 1 <= len(var_names) <= 2:
        raise ValueError("Only 1 or 2 variables supported.")

    symbols = sp.symbols(var_names)

    if type_.strip() in ['r', 'root', 'roots', 'factor', 'factors']:
        if len(symbols) != 1:
            raise ValueError("Root solving supports only one variable.")

        if '=' not in expr:
            expr = expr + ' = 0'
        parts = expr.split('=')

        if len(parts) != 2:
            raise ValueError("Equation must contain exactly one '='.")

        lhs, rhs = expr.split('=')
        expr = sp.sympify(lhs) - sp.sympify(rhs)
        roots = sp.solve(expr, symbols)
        return roots

    elif type_.strip() in ['xy', '(x,y)', 'x,y','dc', 'cd']:

        # -----------------------------------------
        # Handle equation form
        # -----------------------------------------

        try:
            if "=" in expr:
                left, right = expr.split("=")
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)
                sym_expr = left_expr - right_expr  # convert to F(x,y)=0
            else:
                sym_expr = sp.sympify(expr)
        except Exception as exc:
            raise ValueError(f"Invalid expression: {exc}")

        func = sp.lambdify(symbols, sym_expr, modules="numpy")

        # -----------------------------------------
        # Single Variable Evaluation
        # -----------------------------------------

        if len(symbols) == 1:
            start = -loops if negative else 0
            x_vals = np.arange(start, loops + 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                y_vals = func(x_vals)

            y_vals = np.nan_to_num(y_vals, nan=1.0, posinf=1.0, neginf=1.0)

            return x_vals, y_vals

        # -----------------------------------------
        # Two Variable / Equation Evaluation
        # -----------------------------------------

        xr = x_range if x_range else (-loops, loops)
        yr = y_range if y_range else (-loops, loops)

        x_vals = np.arange(xr[0], xr[1] + 1)
        y_vals = np.arange(yr[0], yr[1] + 1)

        X, Y = np.meshgrid(x_vals, y_vals)

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = func(X, Y)

        Z = np.nan_to_num(Z, nan=1.0, posinf=1.0, neginf=1.0)

        return X, Y, Z
    return None