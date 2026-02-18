# importing libs
import numpy as np
import re
from typing import Callable, Any, Union, Tuple, List
import sympy as sp
import os
from pathlib import Path

from numpy import poly1d


def _get_project_root() -> Path:
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

def _get_output_dir(f_type: str = 'gif') -> Path:
    """
    Uses the project root to navigate to the output directory.
    Here, the `vid` folder.

    Parameters
    ----------
    f_type : str
        This is the file extension.
        Here, the default value is 'gif', therefore,
        it will get sent to .../vid/gif.
        If it were mp4 then -> .../vid/mp4.

    Returns
    -------
        Path
    """
    if f_type in ['gif', 'mp4']:
        root = _get_project_root()
        outdir = root / 'vid' / f_type
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    else:
        raise ValueError("Unsupported file type. Supported types are 'gif' and 'mp4'.")

def _get_unique_filename(outdir: Path, f_name: str = 'output', f_type: str = 'gif') -> Path:
    """
    Generates a unique filename by incrementing the value of `i` per non unique filename.
    Meaning, output.py -> output1.py -> output2.py ...

    Parameters
    ----------
    outdir : Path
        Path to the output directory.
    f_name : str
        This is the base filename.
    f_type : str
        This is the file type.

    Returns
    -------
        Path
    """
    i = 0
    while True:
        fname = f"{f_name}{i if i else ''}.{f_type}"
        fpath = outdir / fname
        if not fpath.exists():
            return fpath
        i += 1

#=============================================================

def norm(expr_: str) -> str:
    """
    Used in `ArrayGen/t.py` to normalize the expression.
    It replaces '^' with '**' and adds '*' between numbers and variables.

    Parameters:
    -----------
    expr_: str
        The input mathematical expression as a string.

    Returns:
    --------
    str
    """
    expr_ = expr_.replace('^', '**')
    expr_ = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_)
    return expr_

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

def _build_range(domain: Tuple[float, float], step: float):
    return np.arange(domain[0], domain[1] + step, step)

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
