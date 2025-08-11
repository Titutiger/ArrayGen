# Version 1.1

from typing import Tuple, List, Callable, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Please credit me since I am still a student, thank you ;D

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


class Graphing:

    @staticmethod
    def plot(
            x: Union[List[float], np.ndarray],
            y: Union[List[float], np.ndarray],
            centered: bool = False,
            static: bool = False,
            speed: int = 50,
            z: Optional[np.ndarray] = None,
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
