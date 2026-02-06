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
    def attraction() -> None:
        """
        Creates a vector field where the vectors are attracted to the red dot (controlled by the mouse).

        Parameters
        ----------
            None

        Returns
        -------
            Interactive plot | static plot

        Raises
        ------
        None

        Example
        -------
        >>> import ArrayGen.ArrayGen
        >>> ArrayGen.ArrayGen.Physics.attraction()
        """
        # Vector field grid
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)

        # Initial static positions of vectors
        pos = np.stack([X, Y], axis=-1)

        def update_vectors(mag_pos, strength=5):
            # Vector from each point to magnet position
            vec_to_mag = mag_pos - pos
            dist = np.linalg.norm(vec_to_mag, axis=-1)

            # Avoid zero division by adding a small epsilon
            epsilon = 1e-9
            dist = np.where(dist < epsilon, epsilon, dist)

            # Calculate direction vectors towards magnet, scaled by strength/distance^2 (like inverse square law)
            direction = vec_to_mag / dist[..., np.newaxis]
            magnitude = strength / dist ** 2

            # Limit magnitude to prevent too large vectors near magnet
            magnitude = np.clip(magnitude, 0, 1)
            vectors = direction * magnitude[..., np.newaxis]

            return vectors[..., 0], vectors[..., 1]

        # Create plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

        # Draw initial vectors
        U, V = update_vectors(np.array([0, 0]))
        quiver = ax.quiver(X, Y, U, V)

        # Draw the magnet circle (initially at origin)
        magnet_circle = plt.Circle((0, 0), 0.3, color='red', fill=True)
        ax.add_patch(magnet_circle)

        # Update function for mouse movement
        def on_mouse_move(event):
            if event.inaxes != ax:
                return

            mag_pos = np.array([event.xdata, event.ydata])

            # Update vectors directions
            U, V = update_vectors(mag_pos)
            quiver.set_UVC(U, V)

            # Move the magnet circle to mouse position
            magnet_circle.center = mag_pos

            fig.canvas.draw_idle()

        # Connect mouse motion event
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

        ax.set_title('Interactive Magnet Vector Field')
        plt.grid(True)
        plt.show()

    @staticmethod
    def vector_magnitude(vector: str) -> float | None:
        """
        Gets the magnitude of a vector.

        Parameters
        ----------
        vector: str
            This is the vector to get the magnitude for.
        Returns
        -------
        float

        Raises
        ------
        ValueError
            The vector can only be of 2 or 3 dimensions.

        Examples --------------------------------------------------------------------------------------------- ---------
        --------
        ...
        """
        if vector in ['form', 'formula']:
            i, j, k = symbols('i j k')
            display(Math(r""))
        else:
            a = get_num(vector)
            if isinstance(a, str):
                raise ValueError(f"Invalid vector input: {vector!r}")
            if len(a) == 2:
                x, y = a
                return math.sqrt(x**2 + y**2)
            elif len(a) == 3:
                x, y, z = a
                return math.sqrt(x**2 + y**2 + z**2)
            else:
                raise ValueError('The vector can only be of 2 or 3 dimensions!')

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
    def cross(expr1: str, expr2: str, theta: float, unit_vector: int) -> float:
        """
        ...
        Parameters
        ----------
        expr1 : str
            A string containing exactly three integers representing the first vector.
        expr2 : str
            A string containing exactly three vectors representing the second vector.
        theta : float
            The angle between the two vectors.
        unit_vector : int
            The unit vector (1:i, 2:j, or 3:k) representing the direction of the resultant vector.
        
        Returns
        -------
        float
            The magnitude of the cross product of the two vectors.
        
        Raises
        ------
        None

        Examples
        --------
        ...
        """
        a = Physics.vector_magnitude(expr1)
        b = Physics.vector_magnitude(expr2)
        return a * b * math.sin(theta) * unit_vector