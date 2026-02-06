from src.ArrayGen import get_num
import src.ArrayGen as q

def cross(expr1: str, expr2: str) -> str:
    """
    Returns the cross product (as a string in i, j, k form) of two 3D vectors given as strings.

    Parameters
    ----------
    expr1 : str
        First vector, e.g., '3i+6j-3k'
    expr2 : str
        Second vector, e.g., '4i-2j+5k'
    Returns
    -------
    str
        The cross product in 'ai+bj+ck' format.

    Raises
    ------
    ValueError
        If vectors are not 3D.
    """
    x1, y1, z1 = get_num(expr1, type_='vec')
    x2, y2, z2 = get_num(expr2, type_='vec')
    i = y1 * z2 - z1 * y2
    j = -(z1 * x2 - x1 * z2)
    k = x1 * y2 - y1 * x2
    parts = []
    for v, sym in zip([i, j, k], ['i', 'j', 'k']):
        if v == 0: continue
        s = f"{'+' if v > 0 and parts else ''}{v}{sym}" if v != 1 and v != -1 else (f"+{sym}" if v == 1 else f"-{sym}")
        parts.append(s)
    result = ''.join(parts)
    return result if result else '0'


if __name__ == '__main__':
    cross_val = cross('3i+6j-3k', '4i-2j+5k')
    print(f"Cross: {cross_val}")  # e.g., '-24i-27j-30k'

