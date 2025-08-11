import math
from typing import Tuple, List, Callable, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray, dtype, float64


# Please credit me since I am still a student, thank you ;D
# ----------------------------------------------------------------------------------------------------------------------

class Expression:
    @staticmethod
    def eval_expr(expression: Callable[..., float], loops: int = 10, negative: bool = True,
                  var: int = 1, **kwargs: Any) -> Union[Tuple[List[float], List[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Evaluate an expression for 1 or 2 variables over a range.
        :param expression: The function to evaluate (1 or 2 arguments).
        :param loops: Range max (min is -loops if negative else 0)
        :param negative: Whether to include negatives in ranges.
        :param var: Number of variables: 1 or 2.
        :return: (x, y) for var=1, (X, Y, Z) mesh for var=2.
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

# ----------------------------------------------------------------------------------------------------------------------

class Physics:
    @staticmethod
    def parabolic_curve(initial_velocity: float, theta_degrees: float,
                        graph: bool = True, grid: bool = True) -> tuple[
                                                                      ndarray[tuple[Any, ...], dtype[float64]], ndarray[Any, dtype[Any]] | Any] | None:
        """
        Plots the parabolic trajectory of a projectile in 2D.
        :param initial_velocity: float = u
        :param theta_degrees: float = Q
        :param graph: Whether to plot the graph or not.
        :param grid: Whether to plot the grid or not
        :return: None
        """

        theta = np.radians(theta_degrees) # deg -> rad
        g = 9.8

        R = (initial_velocity**2 * np.sin(2 * theta)) / g # horizontal range
        x = np.linspace(0, R, 500) # R from 0 -> R
        y = x * np.tan(theta) - (g / (2 * initial_velocity **2 * np.cos(theta)**2)) * x**2 # vertical pos at each x

        if graph:
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, label=f'u={initial_velocity} m/s, theta={theta_degrees}')
            plt.title('Projectile trajectory')
            plt.xlabel('Horizontal distance (m)')
            plt.ylabel('Vertical distance (m)')
            plt.legend()

            if grid:
                plt.grid(True)
                plt.show()
            else:
                plt.grid(False)
                plt.show()
        else:
            return x, y

    @staticmethod
    def attraction() -> None:
        """
        Creates a vector field where the vectors are attracted to the red dot (controlled by the mouse).
        :return: Interactive plot | static plot
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

# ----------------------------------------------------------------------------------------------------------------------

class Maths:
    @staticmethod
    def complex_calc(real_part: float, imaginary_part: float, real_part2: float, imaginary_part2: float, addition: bool = False, subtraction: bool = False) -> \
    tuple[float, float] | None:
        if addition:
            real = real_part + real_part2
            imaginary = imaginary_part + imaginary_part2
            return real, imaginary
        elif subtraction:
            real = real_part - real_part2
            imaginary = imaginary_part - imaginary_part2
            return real, imaginary
        return None

    @staticmethod
    def hypo(a: float, b: float, angle_A: float, round_: bool = True, digits: Optional[int] = None) -> float:
        """
        Returns the hypotenuse of any triangle
        :param digits: Optional rounding to n digits
        :param round_: rounding?
        :param angle_A: Angle in degrees
        :param a: Length of side a
        :param b: Length of side b
        :return: c (hypotenuse) -> float
        """
        angle_A = math.radians(angle_A)
        c = math.sqrt(b*b + a*a - 2*b*a * math.cos(angle_A))
        if round_:
            if digits: return round(c, digits)
            else: return round(c, 2)
        else:
            return c

    @staticmethod
    def AP(a: float, d: float, n: int) -> tuple[list[Any], list[Any]]:
        lst = []
        for i in range(n):
            lst.append(i)
            lst.append(a + (i-1)*d)

        x = lst[0::2]
        y = lst[1::2]
        return x, y

    @staticmethod
    def plot(x: float, y: float) -> None:
        plt.scatter(x, y)
        plt.grid()
        plt.show()

    @staticmethod
    def complex_plot(expr: np.ndarray, theta: np.ndarray) -> None:
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        plt.scatter(theta, expr)
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------

class Graphing:

    @staticmethod
    def plot(
            x: Union[List[float], np.ndarray],
            y: Union[List[float], np.ndarray],
            z: Optional[np.ndarray] = None,
            centered: bool = False,
            static: bool = False,
            speed: int = 50,
    ) -> None:
        """
        Plot a graph of one or two variables (1D line or 2D surface).

        Parameters:
        - x, y: For 1D: lists or 1D arrays of coordinates.
                For 2D: meshgrid arrays.
        - z: For 2D surface function values (2D array), optional.
        - centered: if True, center axes at origin.
        - static: if True, show static plot; else animate.
        - speed: interval speed for animation in milliseconds.
        """
        if z is not None:
            Graphing._plot_2var(x, y, z, centered=centered, static=static, speed=speed)
        else:
            Graphing._plot_1var(x, y, centered=centered, static=static, speed=speed)

    @staticmethod
    def _plot_1var(x: List[float], y: List[float], centered: bool, static: bool, speed: int):
        fig, axis = plt.subplots()

        # Setup axes according to centered or not
        if centered:
            xlim = max(abs(min(x)), abs(max(x))) * 1.1
            ylim = max(abs(min(y)), abs(max(y))) * 1.1
            axis.set_xlim(-xlim, xlim)
            axis.set_ylim(-ylim, ylim)
            axis.spines['left'].set_position('zero')
            axis.spines['bottom'].set_position('zero')
            axis.spines['top'].set_color('none')
            axis.spines['right'].set_color('none')
            axis.xaxis.set_ticks_position('bottom')
            axis.yaxis.set_ticks_position('left')
            axis.grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
            axis.set_xlim(min(x), max(x))
            axis.set_ylim(min(y) - 0.1, max(y) + 0.1)
            axis.grid(True)

        if static:
            axis.plot(x, y, lw=2)
            axis.text(0.02, 0.95,
                      f"x = {x[-1]:.3f}\ny = {y[-1]:.3f}",
                      transform=axis.transAxes,
                      fontsize=12,
                      verticalalignment='top')
            plt.show()
        else:
            animated_plot, = axis.plot([], [], lw=2)
            value_text = axis.text(0.50, 0.95, '', transform=axis.transAxes, fontsize=12,
                                   verticalalignment='top')

            def update(frame):
                animated_plot.set_data(x[:frame], y[:frame])
                if 0 < frame <= len(x):
                    curr_x = x[frame - 1]
                    curr_y = y[frame - 1]
                    value_text.set_text(f"x = {curr_x:.3f}\ny = {curr_y:.3f}")
                return animated_plot, value_text

            anim = FuncAnimation(
                fig=fig,
                func=update,
                frames=len(x) + 1,
                interval=speed,
                blit=True
            )
            plt.show()

    @staticmethod
    def _plot_2var(
            X: np.ndarray,
            Y: np.ndarray,
            Z: np.ndarray,
            centered: bool,
            static: bool,
            speed: int
    ):
        from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')

        # Axis limits setup
        if centered:
            x_lim = max(abs(X.min()), abs(X.max())) * 1.1
            y_lim = max(abs(Y.min()), abs(Y.max())) * 1.1
            z_lim = max(abs(Z.min()), abs(Z.max())) * 1.1
            axis.set_xlim(-x_lim, x_lim)
            axis.set_ylim(-y_lim, y_lim)
            axis.set_zlim(-z_lim, z_lim)
        else:
            axis.set_xlim(X.min(), X.max())
            axis.set_ylim(Y.min(), Y.max())
            axis.set_zlim(Z.min(), Z.max())

        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')

        if static:
            axis.plot_surface(X, Y, Z, cmap='viridis')
            plt.show()
        else:
            # Start with an empty surface plot
            surface = [axis.plot_surface(X, Y, np.zeros_like(Z), cmap='viridis')]

            def update(frame):
                # Remove previous surface before drawing new
                # Remove current surface artist
                if surface[0] is not None:
                    surface[0].remove()
                # Plot surface up to current frame rows
                surface[0] = axis.plot_surface(
                    X[:frame, :], Y[:frame, :], Z[:frame, :], cmap='viridis', edgecolor='none'
                )
                return surface

            anim = FuncAnimation(
                fig=fig,
                func=update,
                frames=Z.shape[0] + 1,
                interval=speed,
                blit=False  # Blitting doesn't work well for 3d plots
            )
            plt.show()

# ----------------------------------------------------------------------------------------------------------------------
