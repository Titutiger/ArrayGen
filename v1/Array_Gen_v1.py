import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from typing import Tuple, List, Callable, Any, Optional

# Please credit me since I am still a student, thank you ;D

class Expression:
    @staticmethod
    def evaluate_expression(expression: Callable[[float], float], loops: int = 10, negative: bool = True
                            ) -> Tuple[List[int], List[float]]:
        """
        Evaluates the user-provided expression for x values from -loops to loops inclusive if negative=True,
        else from 0 to loops. Handles common math functions (log10, log, sqrt, etc.).
        Catches ZeroDivisionError and ValueError during evaluation:
        - If any error occurs during negative range evaluation, switches to positive range only,
          with first result = 1.
        - In case of errors in positive range, sets the result at that x to 1 and continues.

        :param expression: A function that takes a float x and returns a float result.
        :param loops: The maximum absolute x value (loops >= 0).
        :param negative: Whether to evaluate from -loops to loops or only 0 to loops.
        :return: Tuple of (x values list, corresponding evaluated results)
        """

        def safe_eval(expr_func: Callable[[float], float], xs: List[int]) -> List[float]:
            results = []
            for x in xs:
                try:
                    val = expr_func(x)
                except (ZeroDivisionError, ValueError) as e:
                    print(f"Warning: error '{e}' at x={x}, setting value=1")
                    val = 1.0
                results.append(val)
            return results

        if negative:
            try:
                x_vals = list(range(-loops, loops + 1))
                results = safe_eval(expression, x_vals)

                # Check if any values were defaulted to 1 due to error in the negative part
                # If yes, cancel negative range and re-run for positive only (with first result=1)
                negative_errors = any(
                    (x < 0 and val == 1.0) for x, val in zip(x_vals, results)
                )
                if negative_errors:
                    print("Error(s) encountered in negative range. Re-evaluating from 0 to loops only.")
                    x_vals = list(range(loops + 1))
                    results = safe_eval(expression, x_vals)
                    # Force first value = 1
                    if results:
                        results[0] = 1.0

            except Exception as e:
                print(f"Unexpected error during negative evaluation: {e}")
                x_vals = list(range(loops + 1))
                results = safe_eval(expression, x_vals)
                if results:
                    results[0] = 1.0

        else:
            x_vals = list(range(loops + 1))
            results = safe_eval(expression, x_vals)
            if results:
                results[0] = 1.0  # Always set first to 1 as per request

        return x_vals, results


    @staticmethod
    def evaluate_expression_updated(expression: Callable[[float], float], loops: int = 10, negative: bool = True,
                                    step: float = 0.1) -> Tuple[List[float], List[float]]:
        """
        Evaluates the user-provided expression for float x values (e.g., radians) from -loops to loops (if negative=True), or 0 to loops.
        Returns x values and results as lists. Handles math errors gracefully.
        """

        def safe_eval(expr_func: Callable[[float], float], xs: List[float]) -> List[float]:
            results = []
            for x in xs:
                try:
                    val = expr_func(x)
                except (ZeroDivisionError, ValueError):
                    val = 1.0
                results.append(val)
            return results

        if negative:
            try:
                x_vals = [i * step for i in range(int(-loops / step), int(loops / step) + 1)]
                results = safe_eval(expression, x_vals)
                negative_errors = any((x < 0 and val == 1.0) for x, val in zip(x_vals, results))
                if negative_errors:
                    x_vals = [i * step for i in range(int(loops / step) + 1)]
                    results = safe_eval(expression, x_vals)
                    if results:
                        results[0] = 1.0
            except Exception:
                x_vals = [i * step for i in range(int(loops / step) + 1)]
                results = safe_eval(expression, x_vals)
                if results:
                    results[0] = 1.0
        else:
            x_vals = [i * step for i in range(int(loops / step) + 1)]
            results = safe_eval(expression, x_vals)
            if results:
                results[0] = 1.0

        return x_vals, results


class Trignometry:
    @staticmethod
    def sin_c(start_coordinate_x: float, end_coordinate_x: float, steps: float=0.1) -> tuple[list, list[float | Any]]:
        """
        Returns an array of [sin(x)/x]
        :param start_coordinate_x: P(x,y) only x
        :param end_coordinate_x: P(x,y) only x
        :param steps: For the difference in units // the resolution.
        :return: array
        :Example:
        >>> print(Trignometry.sin_c(0.0, 5.0))
        """
        start_x: float = start_coordinate_x
        end_x: float = end_coordinate_x

        x_vals: list = []
        current_x = start_x
        while current_x <= end_x:
            x_vals.append(current_x)
            current_x += steps

        result = [1.0 if x == 0 else math.sin(x)/x for x in x_vals] # 1.0 is the math limit
        return x_vals, result

    @staticmethod
    def cos_c(start_coordinate_x: float, end_coordinate_x: float, steps: float = 0.1) -> Tuple[List[float], List[float]]:
        """
        Returns arrays of x and cos(x)/x values over the specified interval.
        :param start_coordinate_x: Starting x value
        :param end_coordinate_x: Ending x value
        :param steps: Step size (resolution)
        :return: (x values, cos(x)/x values)
        :Example:
        >>> print(Trignometry.cos_c(0.0, 5.0))
        """
        x_vals: List[float] = []
        current_x = start_coordinate_x
        while current_x <= end_coordinate_x:
            x_vals.append(current_x)
            current_x += steps

        result = [1.0 if x == 0 else math.cos(x)/x for x in x_vals]  # Limit at x=0 can be defined as needed
        return x_vals, result


class AP:
    @staticmethod
    class Taylor_Series:
        @staticmethod
        def sin_taylor(start_coordinate_x: float, end_coordinate_x: float, steps: float = 0.1, total: int = 10) -> Tuple[
            List[float], List[float]]:
            """
            Returns x values and Taylor series approximations of sin(x) using 'total' terms.

            :param start_coordinate_x: Starting x-value for the evaluation range
            :param end_coordinate_x: Ending x-value for the evaluation range
            :param steps: Step size between x-values (resolution)
            :param total: Number of terms in the Taylor series expansion
            :return: Tuple containing (x values list, corresponding sin(x) approximation list)
            """

            def taylor_sin(x: float, n_terms: int) -> float:
                # Calculate sin(x) using Taylor series sum of 'n_terms' terms
                sin_approx = 0.0
                for n in range(n_terms):
                    sign = (-1) ** n
                    term = (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
                    sin_approx += sign * term
                return sin_approx

            x_vals = []
            y_vals = []
            current_x = start_coordinate_x

            while current_x <= end_coordinate_x:
                x_vals.append(current_x)
                y_vals.append(taylor_sin(current_x, total))
                current_x += steps

            return x_vals, y_vals

        @staticmethod
        def cos_taylor(start:float, end:float, steps:float=0.1, total:int = 10
                       ) -> Tuple[List[float], List[float]]:
            """

            """
            def taylor_cos(x:float, n_terms:int) -> float:
                cos_approx = 0.0
                for n in range(n_terms):
                    sign = (-1) ** n
                    term = (x ** (2 * n)) / math.factorial(2 * n)
                    cos_approx += sign * term
                return cos_approx

            x_vals = []
            y_vals = []
            current_x = start

            while current_x <= end:
                x_vals.append(current_x)
                y_vals.append(taylor_cos(current_x, total))
                current_x += steps

            return x_vals, y_vals

    pass


class Graphing:
    @staticmethod
    def animated_graph(x: list[float], y: list[float], speed:int = 2) -> None:
        fig, axis = plt.subplots()
        axis.set_xlim((min(x), max(x)))
        axis.set_ylim((min(y) - 0.1, max(y) + 0.1))



        animated_plot, = axis.plot([], [], lw=2)
        # Add a text artist for displaying coordinates and value
        value_text = axis.text(0.50, 0.95, '', transform=axis.transAxes, fontsize=12,

                               verticalalignment='top')
        axis.grid(True)
        def update_data(frame):
            animated_plot.set_data(x[:frame], y[:frame])
            if 0 < frame <= len(x):
                # Current coordinates and function value
                curr_x = x[frame - 1]
                curr_y = y[frame - 1]
                value_text.set_text(f"x = {curr_x:.3f}\ny = {curr_y:.3f}")
            return animated_plot, value_text

        animation = FuncAnimation(
            fig=fig,
            func=update_data,
            frames=len(x) + 1,
            interval=speed,
            blit=True
        )

        plt.show()

    @staticmethod
    def animated_graph_centered_origin(x: list[float], y: list[float], speed: int = 2) -> None:
        fig, axis = plt.subplots()

        # Set symmetric limits around zero to center origin
        xlim = max(abs(min(x)), abs(max(x))) * 1.1
        ylim = max(abs(min(y)), abs(max(y))) * 1.1
        axis.set_xlim(-xlim, xlim)
        axis.set_ylim(-ylim, ylim)

        # Move left spine (y-axis) to zero and bottom spine (x-axis) to zero
        axis.spines['left'].set_position('zero')
        axis.spines['bottom'].set_position('zero')

        # Hide the top and right spines
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')

        # Make ticks only appear on left and bottom
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        # Enable grid
        axis.grid(True)

        animated_plot, = axis.plot([], [], lw=2)

        # Text for dynamic coordinate display
        value_text = axis.text(0.50, 0.95, '', transform=axis.transAxes, fontsize=12,
                               verticalalignment='top')

        def update_data(frame):
            animated_plot.set_data(x[:frame], y[:frame])
            if 0 < frame <= len(x):
                curr_x = x[frame - 1]
                curr_y = y[frame - 1]
                value_text.set_text(f"x = {curr_x:.3f}\ny = {curr_y:.3f}")
            return animated_plot, value_text

        animation = FuncAnimation(
            fig=fig,
            func=update_data,
            frames=len(x) + 1,
            interval=speed,
            blit=True
        )

        plt.show()

    @staticmethod
    def static_graph(x: list[float], y: list[float]) -> None:
        fig, axis = plt.subplots()
        axis.set_xlim((min(x), max(x)))
        axis.set_ylim((min(y) - 0.1, max(y) + 0.1))

        axis.plot(x, y, lw=2)

        axis.grid(True)

        # Show the last coordinate and value as static text
        curr_x = x[-1]
        curr_y = y[-1]
        axis.text(0.50, 0.95,
                  f"x = {curr_x:.3f}\ny = {curr_y:.3f}",
                  transform=axis.transAxes,
                  fontsize=12,
                  verticalalignment='top')

        plt.show()

    @staticmethod
    def static_graph_centered_origin(x: list[float], y: list[float]) -> None:
        fig, axis = plt.subplots()
        axis.plot(x, y, lw=2)

        # Set spines (axes lines) so that the origin (0,0) is in the center
        axis.spines['left'].set_position('zero')  # y-axis at x=0
        axis.spines['bottom'].set_position('zero')  # x-axis at y=0

        # Hide the top and right spines
        axis.spines['top'].set_color('none')
        axis.spines['right'].set_color('none')

        # Move ticks to bottom and left (optional, for clean look)
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        # Enable grid lines behind the graph
        axis.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set axis limits symmetrically around zero to emphasize center if needed
        x_lim = max(abs(min(x)), abs(max(x))) * 1.1
        y_lim = max(abs(min(y)), abs(max(y))) * 1.1
        axis.set_xlim(-x_lim, x_lim)
        axis.set_ylim(-y_lim, y_lim)

        # Show coordinate annotation on top-left relative to axis
        curr_x = x[-1]
        curr_y = y[-1]
        axis.text(0.02, 0.95,
                  f"x = {curr_x:.3f}\ny = {curr_y:.3f}",
                  transform=axis.transAxes,
                  fontsize=12,
                  verticalalignment='top')

        plt.show()

class Spirals:
    @staticmethod
    def Archimedean_spiral():
        def archimedean_spiral_r(phi: float, a: float = 0.0, k: float = 1.0) -> float:
            """Radius for Archimedean spiral r = a + k * phi"""
            if phi < 0:
                phi = abs(phi)  # Angle is usually positive in polar, adjust if needed
            return a + k * phi
        angles, radii = Expression.evaluate_expression(archimedean_spiral_r, loops=100, negative=False)
        x_vals = [r * math.cos(angle) for r, angle in zip(radii, angles)]
        y_vals = [r * math.sin(angle) for r, angle in zip(radii, angles)]
        return x_vals, y_vals

    @staticmethod
    def Logarithmic_spiral():
        def logarithmic_spiral_r(phi: float, a: float = 1.0, b: float = 0.1) -> float:
            """Radius for logarithmic spiral r = a * exp(b * phi)"""
            if phi < 0:
                phi = abs(phi)
            return a * math.exp(b * phi)

        angles, radii = Expression.evaluate_expression(logarithmic_spiral_r, loops=100, negative=False)
        x_vals = [r * math.cos(angle) for r, angle in zip(radii, angles)]
        y_vals = [r * math.sin(angle) for r, angle in zip(radii, angles)]
        return x_vals, y_vals


class Logs:
    @staticmethod
    def compute_log(value: float, log10: bool = False, loge: bool = False) -> Optional[float]:
        """
        Computes the logarithm of the given value.

        :param value: The input number (must be > 0).
        :param log10: If True, compute log base 10.
        :param loge: If True, compute natural log (base e).
        :return: The logarithm of the value if a log flag is set, else None.
        :raises ValueError: If value <= 0 or if both/neither log flags are True.
        """

        if value <= 0:
            raise ValueError("Value must be greater than 0 to compute logarithm.")

        if (log10 and loge) or (not log10 and not loge):
            raise ValueError("Exactly one of log10 or loge must be True.")

        if log10:
            return math.log10(value)
        elif loge:
            return math.log(value)


