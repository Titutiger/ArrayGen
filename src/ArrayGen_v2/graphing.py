import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Union, List, Optional
from src.ArrayGen_v2 import utils


class Graphing:
    @staticmethod
    def simple_plot(x: float, y: float) -> None:
        """
        Plots a simple point via x and y on a normal plot.

        Parameters
        ----------
        x: float
            The x coordinate of the point.
        y: float
            The y coordinate of the point.


        Returns
        -------
        None -> Creates a plot

        Raises
        ------
        None

        Example
        -------
        ...

        """
        plt.scatter(x, y)
        plt.grid()
        #plt.c
        plt.show()

    @staticmethod
    def complex_plot(expr: np.ndarray, theta: np.ndarray, color: str = 'viridis') -> None:
        """
        Creates an Argand plot / complex plot via the expression and theta angle provided.

        Parameters
        ----------
        expr: np.ndarray
            This is the complex expression.
        theta: np.ndarray
            This is the angle of the complex expression.
        color: str
            This is purely for visual purposes, it changes the theme of the graph.

        Returns
        -------
        None -> Creates a plot.

        Raises
        ------
        None

        Examples
        --------
        ...

        """
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        colors = np.abs(expr)
        plt.scatter(theta, np.abs(expr), c=colors, cmap=color)
        plt.show()


    @staticmethod
    def mandelbrot(mandel_set,
                   xmin: float = -2.0, xmax: float = 1.0,
                   ymin: float = -1.5, ymax: float = 1.5, ):
        plt.imshow(mandel_set, cmap='hot', extent=[xmin, xmax, ymin, ymax])
        plt.colorbar(label='Iterations to diverge')
        plt.title('Mandelbrot Set')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.show()



    @staticmethod
    def plot(
            x: Union[List[float], np.ndarray],
            y: Union[List[float], np.ndarray],
            z: Optional[np.ndarray] = None,
            centered: bool = False,
            static: bool = False,
            speed: int = 50,
            color=None,  # string or array or None
            save: tuple = (False, '')
    ) -> None:
        if z is not None:
            Graphing._plot_2var(x, y, z, centered=centered, static=static, speed=speed, color=color, save=save)
        else:
            Graphing._plot_1var(x, y, centered=centered, static=static, speed=speed, color=color, save=save)

    @staticmethod
    def _plot_1var(x, y, centered, static, speed, color, save):
        fig, axis = plt.subplots()
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
            value_text = axis.text(0.50, 0.95, '',
                                   transform=axis.transAxes, fontsize=12,
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
            if save[0]:
                out_dir = utils.get_output_dir(save[1])
                ext = save[1]
                filepath = utils.get_unique_filename(out_dir, 'output', ext)
                if save[1] == 'gif':
                    anim.save(str(filepath), writer='pillow')
                elif save[1] == 'mp4':
                    anim.save(str(filepath), writer='ffmpeg')
                else:
                    print(f'Unknown format: {save[1]}')
                print('Animation saved!')
            else:
                plt.show()


    @staticmethod
    def _plot_2var(X, Y, Z, centered, static, speed, color, save):
        from mpl_toolkits.mplot3d import Axes3D
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


        # Core logic for color/cmap handling
        surface_kwargs = {}
        if color is None:
            # Use default colormap
            surface_kwargs['cmap'] = 'viridis'
        elif isinstance(color, str):
            if color in plt.colormaps():
                # Named colormap, e.g., 'plasma', 'inferno'
                surface_kwargs['cmap'] = color
            else:
                # Assume it's a color name like 'red', 'skyblue', etc.
                surface_kwargs['color'] = color
        elif isinstance(color, np.ndarray) or isinstance(color, list):
            # If color is a 2D array, use as facecolors
            surface_kwargs['facecolors'] = color
        else:
            surface_kwargs['cmap'] = 'viridis'

        if static:
            axis.plot_surface(X, Y, Z, **surface_kwargs)
            plt.show()
        else:
            surface = [axis.plot_surface(X, Y, np.zeros_like(Z), **surface_kwargs)]

            def update(frame):
                if surface[0] is not None:
                    surface[0].remove()
                surface[0] = axis.plot_surface(
                    X[:frame, :], Y[:frame, :], Z[:frame, :], **surface_kwargs
                )
                return surface

            anim = FuncAnimation(
                fig=fig,
                func=update,
                frames=Z.shape[0] + 1,
                interval=speed,
                blit=False
            )
            if save[0]:
                out_dir = utils.get_output_dir(save[1])
                ext = save[1]
                filepath = utils.get_unique_filename(out_dir, 'output', ext)
                if save[1] == 'gif':
                    anim.save(str(filepath), writer='pillow')
                elif save[1] == 'mp4':
                    anim.save(str(filepath), writer='ffmpeg')
                else:
                    print(f'Unknown file format: {save[1]}')
                print(f'Animation saved to {filepath}')
            else:
                plt.show()

if __name__ == '__main__':
    import src.ArrayGen as q

    x, y, z = q.expr('x**2 + 3*x*y + y**3', vars_="x y")
    Graphing.plot(x, y, z=z, static=False, color='twilight', save=(True, 'gif'))

