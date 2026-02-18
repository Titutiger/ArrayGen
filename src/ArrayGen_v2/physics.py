import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, Tuple
from sympy import symbols, sin, expand, pretty, Eq, cos, tan, pi, simplify, pretty
from IPython.display import display, Math
import re
from .utils import *
import math

class Physics:
    @staticmethod
    def projectile_motion(
            u: float | str,
            theta_degrees: float = None,
            range_: Optional[float] = None,
            height_: Optional[float] = None,
            f_range: bool = False,
            f_height: bool = False,
            gravity: float = 9.8,
            graph: bool = True,
            grid: bool = True,
            graph_range_x: float = 50,
            graph_range_y: float = 50,
    ) -> Union[None, Tuple[float, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Plots the parabolic trajectory of a projectile in 2D using u as the initial velocity, theta in degrees,
        gravity as a general constant (can be changed) as 9.8, with graph and grid being on by default.

        Parameters
        ----------
        u: float
            This is the initial velocity of the projectile.
        theta_degrees: float
            This is the angle at which the projectile is aimed at from the surface.
        range_: Optional[float]
        height_: Optional[float]
            If this is given, you have the option to find the range or the height of the projectile.
        gravity: float
            This is a constant which affects the flight path of the projectile.
            This is set to 9.8 by default.
        graph: bool
        grid: bool
            These are there to display the graph and/or the grid; which are set to True by default.
        graph_range_x: float
        graph_range_y: float
            These are values that the projectile is unaffected by. These affect the graph as if to limit the x or y-axis
            range to variable values.
            These are set to 50 by default.
        f_range: bool
        f_height: bool
            These are set to False by default. They find the range and/or the height of the path of the projectile.


        Returns
        -------
            None if graph is True, else:
            Tuple containing computed range or height and arrays of x and y trajectory points.

        Raises
        ------
            None

        Examples
        --------
        >>> import ArrayGen.ArrayGen
        >>> ArrayGen.ArrayGen.Physics.projectile_motion(10.0, 45, graph_range_x=11, graph_range_y=14)
        *Graph displays*

        """

        if isinstance(u, str):
            if u in ['form', 'formula']:
                x_sym, u_sym, theta_sym, g_sym = symbols('x u theta g')
                R_expr = (u_sym ** 2 * sin(2 * theta_sym)) / g_sym
                h_expr = (u_sym ** 2 * sin(theta_sym) ** 2) / (2 * g_sym)
                y_expr = x_sym * tan(theta_sym) - (g_sym / (2 * u_sym ** 2 * cos(theta_sym) ** 2)) * x_sym ** 2

                # Show Range, Height, and Trajectory Equations in LaTeX
                display(Math(r"R = \frac{u^2 \sin 2\theta}{g}"))
                display(Math(r"h = \frac{u^2 \sin^2\theta}{2g}"))
                display(Math(r"y(x) = x\tan\theta - \frac{g}{2u^2 \cos^2\theta}x^2"))
            return None
        else:

            g = gravity
            theta = np.radians(theta_degrees)

            if f_range:
                R = (u ** 2 * np.sin(2 * theta)) / g
            elif f_height:
                R = (u ** 2 * (np.sin(theta) ** 2)) / (2 * g)
            else:
                R = (u ** 2 * np.sin(2 * theta)) / g

            x = np.linspace(0, R, 500)
            y = x * np.tan(theta) - (g / (2 * u ** 2 * np.cos(theta) ** 2)) * x ** 2

            if graph:
                plt.figure(figsize=(8, 5))
                plt.plot(x, y, label=f'u={u} m/s, θ={theta_degrees}°')
                plt.title('Projectile Trajectory')
                plt.xlabel('Horizontal distance (m)')
                plt.ylabel('Vertical distance (m)')
                plt.xlim(0, max(graph_range_x, x.max()))
                plt.ylim(0, max(graph_range_y, y.max()))
                plt.legend()
                plt.grid(grid)
                plt.show()
                return None

            if f_range or f_height:
                return R, x, y

            return x, y


    @staticmethod
    def dot(expr1: str, expr2: str) -> float:
        """
        Computes the dot product of two 3-dimentional vectors which are represented as strings.

        Each input string must contain exactly three integers (+ or -).
        The function extracts these integers and calculates their dot product.

        Parameters
        ----------
        expr1 : str
            A string containing exactly three integers representing the first vector.
        expr2 : str
            A string containing exactly three vectors representing the second vector.

        Returns
        -------
        float
            The dot product of the two vectors.

        Raises
        ------
        ValueError
            If either expression does not contain exactly three integers.

        Examples
        --------
        >>> import src.ArrayGen as q
        >>> print(q.Physics.dot('3c+5y-7s', '1a+2f-3d'))
        -8.0

        """
        try:
            x, y, z = get_num(expr1)
            a, b, c = get_num(expr2)
        except ValueError:
            raise ValueError(f'Error: The expression must have exactly 3 numbers!')

        return (x * a) + (y * b) + (z * c)

    @staticmethod
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
            s = f"{'+' if v > 0 and parts else ''}{v}{sym}" if v != 1 and v != -1 else (
                f"+{sym}" if v == 1 else f"-{sym}")
            parts.append(s)
        result = ''.join(parts)
        return result if result else '0'