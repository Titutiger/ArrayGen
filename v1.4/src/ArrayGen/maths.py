import numpy as np
import math
import re
from typing import LiteralString
from .utils import *
from sympy import symbols, sin, cos, tan
from IPython.display import display, Math

class Maths:

    @staticmethod
    def factorial(num: int) -> int:
        """
        This gets the factorial of a number using recursion.

        Parameters
        ----------
        num: int
            This is the number of which you want to find the factorial of.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If num is negative or float.

        Examples
        --------
        ...

        """
        if num < 0 or isinstance(num, float):
            raise ValueError('Cannot get factorials of negative numbers and decimal numbers.')
        elif num == 0 or num == 1:
            return 1
        else:
            return num * Maths.factorial(num - 1)
    
    @staticmethod
    def permutations(out_of: int, select: int) -> float:
        """
        This gets the total permutations of a group.
        Using the formula: nCr, where `n` is the total size of the group and r is the number of selections.
        It means, from n, select r or from 10 select any 3.

        Parameters
        ----------
        out_of: int
            This is the size of the group.
        select: int
            This is the number of selections possible.


        Returns
        -------
        float

        Raises
        ------
        None

        Examples
        --------
        ...
        """
        res = Maths.factorial(out_of) / (Maths.factorial(select) * Maths.factorial(out_of - select))
        return res

    @staticmethod
    class Complex:
        @staticmethod
        def z_algebra(z1: str, z2: str, operation: str) -> \
                tuple[float | int | LiteralString | str, float | int] | tuple[float | int, float | int] | None:
            """
            Does simple algebra calculations for complex numbers via real and imaginary parts.

            Parameters
            ----------
            z1: str
                This is the complex number written as: a + bi
            z2: str
                This is the second complex number written as a+bi

            Returns
            -------
            float
                Returns the real and imaginary part of the complex number.
            None
                If the operation is none, then it will return None.

            Raises
            ------
            None

            Examples
            --------
            ...

            """
            x1, y1 = get_num(z1, var=2)
            x2, y2 = get_num(z2, var=2)
            if operation == '+':
                real = x1 + x2
                imaginary = y1 + y2
                return real, imaginary
            elif operation == '-':
                real = x1 - x2
                imaginary = y2 - y2
                return real, imaginary
            return None

        @staticmethod
        def z_forms(z: str, _round: str = '2', polar: bool = True, values: bool = False) -> str | tuple[float, float]:
            """
            <...>

            Parameters
            ----------

            Returns
            -------

            Raises
            ------

            Examples
            --------

            """
            if z == 'help':
                z_form_help = """
                This is a function to find the polar forms or values of a complex number 'z'.
                -----------------------------------------------------------------------------
                Here is how it goes:
                    z_forms('3 + 5i') # by default polar=True
                    This returns the polar form.
                """
                return z_form_help

            a, b = get_num(z.replace(' ', ''), 2)
            # 'a' is the real part ; 'b' is the imaginary part
            mag = math.sqrt(a ** 2 + b ** 2)
            theta = math.atan2(b, a)
            if values:
                return mag, theta
            elif polar:
                return f'z = {mag:.{_round}f}(cos{theta:.{_round}f} + i.sin{theta:.{_round}f})'
            else:
                return f'z = {mag}e ^ (i * {theta:.{_round}f})'

    @staticmethod
    def hypo(a: float, b: float = None, angle_A: float = None) -> float | None:
        """
        Gets the magnitude of the hypotenuse for any triangle.
        Using the formula: sqrt(a^2 + b^2 - 2ab * cos(theta))

        Parameters
        ----------
        a: float
            This can be any side 1.
        b: float
            This can be any side 2.
        angle_A: float
            This is the angle of 1.

        Returns
        -------
        float
            Returns the magnitude of the hypotenuse.

        Raises
        ------
        None

        Examples
        --------
        ...

        """

        if a in ['form', 'formula']:
            a, b, theta = symbols('a b theta')
            display(Math(r"Hypotenuse = \sqrt{a^2 + b^2 - 2ab cos \theta}"))
            return None
        else:
            match(a, b, angle_A):
                case(float(), float(), float()):
                    angle_A = math.radians(angle_A)
                    c = math.sqrt(b * b + a * a - 2 * b * a * math.cos(angle_A))
                    return c
                case _:
                    raise ValueError('Invalid input datatypes!')


    @staticmethod
    def pascal_triangle(n: int) -> list[list[int]] | None:
        """
        Generates a Pascal's triangle up to te nth layer (0-indexed).

        Parameters
        ----------
        n: int
            This is the number of layers (rows) to generate.

        Returns
        -------
        list of lists of int
            Pascal's triangle rows up to layer n.

        Raises
        ------
            None for now...

        Examples
        --------
        """
        if n == 'help':
            pascal_help = """
            This is a function for generating a pascal's triangle upto the nth term.
            ------------------------------------------------------------------------
                Here is how it goes:
                pascal_triangle(5)
                >> This generates a pascal's triangle upto the 5th term.
            """
        triangle = []
        for i in range(n):
            if i == 0:
                triangle.append([1])
            else:
                prev_row = triangle[-1]
                row = [1]
                for j in range(1, i):
                    row.append(prev_row[j - 1] + prev_row[j])
                row.append(1)
                triangle.append(row)
        return triangle


    @staticmethod
    class Progressions:
        @staticmethod
        def term(type_progression: str, p: str = None, get_term: int = None, **kwargs) -> float | None | str:
            """
            Gets the user specified term of a progression of type AP, GP, HP or AGP; and returns it in a float value.
            Gets the progression via a string format and then calls the `_get_num()` function to extract numbers from
            the string, and then stores them as `a`, `d` / `r` / `...` to be used later in `Tn`.

            Parameters
            ----------
            type_progression : str
                This is to specify whether the Progression is an AP, GP, HP or AGP.
            p: str
                This is the progression itself. The progression must have a minimum of 2 terms.
                Ex: '1, 3, 5, 7'
            get_term : int
                This is the user defined nth term. If this is 5, this function will get the 5th term.
            **kwargs : Any
                This will be added later on ############################################################################

            Returns
            -------
            float
                Returns the nth term as a float.

            Raises
            ------
            ValueError
                When the progression doesn't meet the minimum of 2 terms.
            TypeError
                When the type of progression does not match.

            Examples
            --------
            ...

            """
            if type_progression == 'help':
                p, get_term = None, None
                term_help = """
                    This is a function for getting a term of an accepted progression.
                    For now, this accepts APs, GPs, HPs, and AGPs.
                    ----------------------------------------------
                    Here is how it goes:
                        if progression is an AP:
                            term('ap', '2, 4, 6', 7)
                            >> This will get the 7th term of the ap 2, 4, 6.
                            NOTE: The minimum length of any progression should be 2 terms.
                                    """
                return term_help

            match type_progression.lower():
                case ('ap'):
                    match (type_progression, p, get_term):
                        case (str(), str(), int()):
                            try:
                                a, b = get_num(p, 2)
                                d = b - a
                                tn = a + (get_term - 1) * d
                                return tn

                            except ValueError:
                                raise ValueError('The Progression must have at least 2 terms!')

                        case _:
                            raise TypeError('Invalid input types!')

                case ('gp'):
                    match (type_progression, p, get_term):
                        case (str(), str(), int()):
                            try:
                                a, b = get_num(p, 2)
                                r = b / a
                                tn = a * (r ** (get_term - 1))
                                return tn
                            except ValueError:
                                raise ValueError('The Progression must have at least 2 terms!')
                        case _:
                            raise TypeError('Invalid input types!')

                case ('HP'):  ############################################################################################################################################################################################
                    return None

                case _:
                    raise ValueError('Input a valid progression!')

        @staticmethod
        def get_progression(type_progression: str, a: float | int = None, **kwargs) -> list[Any] | str:
            match type_progression.lower():
                case ('ap'):
                    d = kwargs.get('d')
                    n = kwargs.get('n')
                    if d is None or n is None:
                        raise ValueError('For AP, "d" and "n" must be provided!')

                    d = float(d)
                    try:
                        n = int(n)
                    except ValueError:
                        raise ValueError('n must be a positive integer!')
                    if n <= 0:
                        raise ValueError('n must be a positive integer!')

                    ap = [a + i * d for i in range(n)]
                    return ap

                case ('gp'):
                    r = kwargs.get('r')
                    n = kwargs.get('n')
                    if r is None or n is None:
                        raise ValueError('For GP, "r" and "n" must be provided!')

                    r = float(r)
                    try:
                        n = int(n)
                    except ValueError:
                        raise ValueError('n must be a positive integer!')

                    if n <= 0:
                        raise ValueError('n must be a positive integer!')

                    gp = [a * (r ** i) for i in range(n)]
                    return gp

                case ('hp'):
                    d = kwargs.get('d')
                    n = kwargs.get('n')
                    if d is None or n is None:
                        raise ValueError('For HP, "d" and "n" must be provided!')

                    d = float(d)
                    try:
                        n = int(n)
                    except ValueError:
                        raise ValueError('n must be a positive integer!')
                    if n <= 0:
                        raise ValueError('n must be a positive integer!')

                    hp = []
                    for i in range(n):
                        term = a + i * d
                        if term == 0:
                            raise ZeroDivisionError('Term in AP sequence became 0, hence, cannot take the reciprocal'
                                                    'for HP!')
                        hp.append(1 / term)
                    return hp

                case ('agp'):
                    d = kwargs.get('d')
                    r = kwargs.get('r')
                    n = kwargs.get('n')
                    if d is None or n is None or r is None:
                        raise ValueError('For AGP, "d", "r" and "n" must be provided!')

                    d = float(d)
                    r = float(r)
                    try:
                        n = int(n)
                    except ValueError:
                        raise ValueError('n must be a positive integer!')
                    if n <= 0:
                        raise ValueError('n must be a positive integer!')

                    agp = [(a + i * d) * (r ** i) for i in range(n)]
                    return agp

                case ('help'):
                    prog_help = """
                    This is a function for generating progressions!
                    Note that `a` is always necessary for any progression.
                    ----------------------------------------------------
                        Here is how it goes:
                    For an AP, do:
                    xxx.get_progression('ap', a: float)
                    - wherein the first parameter determines the type of progression (ap, gp, hp or agp)
                    - the second is a which is the first term of that progression.
                    next:
                    if ap -> a, d, n
                    if gp, a, r, n
                    if hp -> a, d, n
                    if agp, a, r, n
    
                    """
                    return prog_help

                case _:
                    raise ValueError('Input a valid progression!')


    @staticmethod
    def mandelbrot_set(
            custom_formula: Callable[[complex, complex], complex],
            width: int = 800, height: int = 800,
            max_iterations: int = 100,
            xmin: float = -2.0, xmax: float = 1.0,
            ymin: float = -1.5, ymax: float = 1.5
    ) -> np.ndarray:
        """
        Calculate Mandelbrot set escape iterations using a custom iteration formula.

        Parameters
        ----------
        custom_formula: Callable[[complex, complex], complex]
            Function f(z, c) defining the iteration step.
        width, height: int
            Pixels in the output grid.
        max_iterations: int
            Maximum iterations for escape.
        xmin, xmax, ymin, ymax: float
            Bounds for the complex plane.

        Returns
        -------
        2D numpy array of iteration counts for each point.

        Raises
        ------
        None

        Examples
        --------
        ...

        """

        def mandelbrot(c: complex, max_iter: int) -> int:
            z = 0 + 0j
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = custom_formula(z, c)
                n += 1
            return n

        real_axis = np.linspace(xmin, xmax, width)
        imag_axis = np.linspace(ymin, ymax, height)

        result_set = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                c = complex(real_axis[j], imag_axis[i])
                result_set[i, j] = mandelbrot(c, max_iterations)
        return result_set