# importing libs
import numpy as np
import re
from typing import Callable, Any, Union, Tuple, List
import sympy as sp
import os
from pathlib import Path

from numpy import poly1d


def get_project_root() -> Path:
    """
    Gets the root of the project using Pathlib's `Path`.

    Returns
    -------
        Path
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name in {'src', 'tests'}:
            return parent.parent
    return Path.cwd()

def get_output_dir(fmt='gif') -> Path:
    """
    Uses the project root to navigate to the output directory.
    Here, the `vid` folder.

    Parameters
    ----------
    fmt : str
        This is the file extension.
        Here, the default value is 'gif', therefore,
        it will get sent to .../vid/gif.
        If it were mp4 then -> .../vid/mp4.

    Returns
    -------
        Path
    """
    root = get_project_root()
    outdir = root / 'vid' / fmt
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def get_unique_filename(outdir: Path, base: str = 'output', ext: str = 'gif') -> Path:
    """
    Generates a unique filename by incrementing the value of `i` per non unique filename.
    Meaning, output.py -> output1.py -> output2.py ...

    Parameters
    ----------
    outdir : Path
        Path to the output directory.
    base : str
        This is the base filename.
    ext : str
        This is the file type.

    Returns
    -------
        Path
    """
    i = 0
    while True:
        fname = f"{base}{i if i else ''}.{ext}"
        fpath = outdir / fname
        if not fpath.exists():
            return fpath
        i += 1

def get_num(expr: str, var: int = None, type_: str = '') -> Union[Tuple[int, ...], str]:
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

    if type_ in ['vec', 'vector']:
        matches = re.findall(r'([+-]?\d*)\s*([ijkxyz])', expr.replace(' ', ''))
        axes = ['i', 'j', 'k']
        vals = {a: 0 for a in axes}
        for coef, axis in matches:
            val = int(coef) if coef not in ('', '+', '-') else (1 if coef in ('', '+') else -1)
            vals[axis] += val
        return tuple(vals[a] for a in axes)

    else:
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
    """
    Generates an array of domains and co-domains of the given expression.
    Say the expression is `3x`. Then:
    Domain: 0, 1, 2, 3
    Co-domain: 0, 3, 6, 9
    Here, start and end point can be changed via `negative` and `loops`.

    For expressions with 2 variables:
    `2x + 4y`
    Keep `vars_` as 'x y' for any variable.

    Or in other words:

    Evaluates a mathematical expression with one or two variables over a specified range.
    This function evaluates the given callable expression over a range of integer input values.
    It supports either a singular variable (1D) or two variables (2D mesh grid) evaluation.
    The results include error handling to ensure safe evaluation, setting invalid called to 1.0.


    Parameters
    ----------
    expr: str
        This is the mathematical expression.
    loops: int
        This is the number of iterations.
    negative: bool
        If true, then range = -x to x for x is the number of loops.
    vars_: str
        This is the variable name(s).

    Returns
    -------
        np.ndarray

    """
    var_names = vars_.split()
    if not (1 <= len(var_names) <= 2):
        raise ValueError("Only one or two variables supported in 'vars', e.g., 'x' or 'x y'.")
    symbols = sp.symbols(var_names)
    try:
        expr_ = sp.sympify(expr)
    except Exception as exc:
        # I really have not faced any kind of specific exceptions therefore the bare
        # except cause.
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
                # Again, the same thing. I have not faced any specific exceptions therefore
                # the bare cause.
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
            # Same thing...
            nx, ny = X.shape
            Z = np.empty_like(X, dtype=np.float64)
            for i in range(nx):
                for j in range(ny):
                    try:
                        v = func(X[i, j], Y[i, j])
                        if np.isnan(v) or np.isinf(v):
                            v = 1.0
                    except Exception:
                        # Yes, you guessed it! The same reason.
                        v = 1.0
                    Z[i, j] = v
        return X, Y, Z
    else:
        raise ValueError("Only 1 or 2 variables supported for evaluation.")


def get_function(y_values: list, degree: int = 1) -> poly1d:
    """
    Fits a polynomial function f(x) to given y_values.

    Parameters
    ----------
    y_values : list
        List of y-values (function outputs).
    degree : int, optional
        Degree of the polynomial to fit (default = 1).

    Returns
    -------
    poly1d
        A NumPy polynomial function that can be called like f(x).
    """
    x_values = np.arange(1, len(y_values) + 1)

    coeffs = np.polyfit(x_values, y_values, degree)
    p = np.poly1d(coeffs)

    # Create a readable string representation
    terms = []
    power = degree
    for coef in coeffs:
        if abs(coef) < 1e-10:
            power -= 1
            continue
        if power > 1:
            terms.append(f"{coef:.3f}x^{power}")
        elif power == 1:
            terms.append(f"{coef:.3f}x")
        else:
            terms.append(f"{coef:.3f}")
        power -= 1

    func_str = " + ".join(terms)
    print(f"f(x) = {func_str}")
    return p
