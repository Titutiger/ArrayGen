import numpy as np
import sympy as sp
from typing import Tuple

from src.ArrayGen_v2.utils import _build_range

def expr(
        expression: str,
        vars_: str = "x y",
        domain: Tuple[float, float] = (-10, 10),
        step: float = 1.0
):
    """
    Evaluates a mathematical expression.

    Parameters
    ----------
    expression: str
        Mathematical expression to be evaluated.
    vars_: str
        To identify variables.
    domain: tuple
        Start and end ranges for the values to be plugged into the function.
    step: float
         Increments, whether by +1 or +0.3.

    Returns
    -------
    tuple of np.ndarray
    """

    # normalize
    eq = expression.replace("^", "**")
    var_names = vars_.split()
    symbols = sp.symbols(var_names)

    if '=' in eq:
        lhs, rhs = eq.split('=')

        lhs = sp.sympify(lhs)
        rhs = sp.sympify(rhs)

        expr = lhs - rhs
    else:
        expr = sp.sympify(eq)

    func = sp.lambdify(symbols, expr, modules="numpy")

    x_vals = _build_range(domain, step)
    y_vals = _build_range(domain, step)

    X, Y = np.meshgrid(x_vals, y_vals)

    with np.errstate(divide="ignore", invalid='ignore'):
        Z = func(X, Y)

    Z = np.nan_to_num(Z, nan=np.nan)

    return X, Y, Z