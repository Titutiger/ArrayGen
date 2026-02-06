import numpy as np
import re
from typing import Callable, Any, Union, Tuple, List
import sympy as sp
import os
from pathlib import Path

def get_project_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name in {'src', 'tests'}:
            return parent.parent
    return Path.cwd()

def get_output_dir(fmt='gif'):
    root = get_project_root()
    outdir = root / 'vid' / fmt
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def get_unique_filename(outdir: Path, base: str = 'output', ext: str = 'gif') -> Path:
    i = 0
    while True:
        fname = f"{base}{i if i else ''}.{ext}"
        fpath = outdir / fname
        if not fpath.exists():
            return fpath
        i += 1

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


def expr(
    expr: str,
    loops: int = 10,
    negative: bool = True,
    vars_: str = 'x',
    **kwargs: Any
) -> Union[
     Tuple[np.ndarray, np.ndarray],
     Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    var_names = vars_.split()
    if not (1 <= len(var_names) <= 2):
        raise ValueError("Only one or two variables supported in 'vars', e.g., 'x' or 'x y'.")
    symbols = sp.symbols(var_names)
    try:
        expr_ = sp.sympify(expr)
    except Exception as exc:
        raise ValueError(f"Could not parse expression: {exc}")

    if len(symbols) == 1:
        func = sp.lambdify(symbols[0], expr_, modules=['numpy'])
        if negative:
            x_vals = np.arange(-loops, loops + 1)
        else:
            x_vals = np.arange(0, loops + 1)
        results = np.empty_like(x_vals, dtype=np.float64)
        for i, val in enumerate(x_vals):
            try:
                y = func(val)
                if np.isnan(y) or np.isinf(y):
                    y = 1.0
                results[i] = y
            except Exception:
                results[i] = 1.0
        return x_vals, results

    elif len(symbols) == 2:
        func = sp.lambdify(symbols, expr_, modules=['numpy'])
        x_range = kwargs.get("x_range", (-loops, loops))
        y_range = kwargs.get("y_range", (-loops, loops))
        if (not isinstance(x_range, (tuple, list)) or not isinstance(y_range, (tuple, list))
            or len(x_range) != 2 or len(y_range) != 2):
            raise ValueError("x_range and y_range must be tuples/lists with 2 integers each")
        x_vals = np.arange(x_range[0], x_range[1] + 1)
        y_vals = np.arange(y_range[0], y_range[1] + 1)
        X, Y = np.meshgrid(x_vals, y_vals)
        # Eval vectorized if possible, elementwise if fails
        try:
            Z = func(X, Y)
            mask = np.isnan(Z) | np.isinf(Z)
            Z[mask] = 1.0
        except Exception:
            nx, ny = X.shape
            Z = np.empty_like(X, dtype=np.float64)
            for i in range(nx):
                for j in range(ny):
                    try:
                        v = func(X[i, j], Y[i, j])
                        if np.isnan(v) or np.isinf(v):
                            v = 1.0
                    except Exception:
                        v = 1.0
                    Z[i, j] = v
        return X, Y, Z
    else:
        raise ValueError("Only 1 or 2 variables supported for evaluation.")

