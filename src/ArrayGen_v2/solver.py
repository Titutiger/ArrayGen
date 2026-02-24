# solver.py

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
)


class EquationSolver:
    '''
    Mini symbolic equation engine.

    Capabilities:
        • Parse algebraic expressions
        • Handle implicit multiplication (e.g., 2x, at)
        • Convert equations to zero-form
        • Detect unknown variables
        • Solve symbolically
        • Substitute known values
        • Return numeric results

    Example
    -------
        >>> engine = EquationSolver('v = u + a*t', ('v', 'u', 'a', 't'))
        >>> result = engine.solve(given={'v': 20, 'a': 2, 't': 5})
        >>> # returns 10.0

        OR

        >>> engine = EquationSolver(
        >>>     "P1 + 1/2*rho*v1^2 + rho*g*h1 = P2 + 1/2*rho*v2^2 + rho*g*h2",
        >>>     ('P1','rho','v1','g','h1','P2','v2','h2')
        >>> )
        >>>
        >>> engine.solve(
        >>>     given={
        >>>         'rho': 1000,
        >>>         'v1': 3,
        >>>         'g': 9.81,
        >>>         'h1': 2,
        >>>         'P2': 80000,
        >>>         'v2': 5,
        >>>         'h2': 1,
        >>>         'P1': 100000
        >>>     }
        >>> )

    '''

    def __init__(self, expr: str, symbols: tuple[str]):
        self.original_expr = expr.replace('^', '**')
        self.symbol_names = symbols
        self.symbols = sp.symbols(symbols)

        self.sym_map = {str(s): s for s in self.symbols}

        self.zero_expr = self._to_zero_form(self.original_expr)

    # -------------------------------------
    # Parsing
    # -------------------------------------

    def _parse(self, expr: str):
        return parse_expr(expr, transformations=TRANSFORMATIONS)

    # -------------------------------------
    # Convert equation to zero form
    # -------------------------------------

    def _to_zero_form(self, expr: str):
        if '=' in expr:
            parts = expr.split('=')
            if len(parts) != 2:
                raise ValueError('Equation must contain exactly one "=".')

            left = self._parse(parts[0])
            right = self._parse(parts[1])
            return left - right
        else:
            return self._parse(expr)

    # -------------------------------------
    # Detect unknown variable
    # -------------------------------------

    def _detect_unknown(self, given: dict):
        given_keys = set(given.keys())
        all_keys = set(self.sym_map.keys())

        unknowns = list(all_keys - given_keys)

        if len(unknowns) == 0:
            raise ValueError('No unknown variable to solve for.')

        if len(unknowns) > 1:
            raise ValueError(f'Multiple unknowns detected: {unknowns}')

        return self.sym_map[unknowns[0]]

    # -------------------------------------
    # Solve equation
    # -------------------------------------

    def solve(self, given: dict):
        '''
        Solve the equation for the missing variable.

        Parameters
        ----------
        given : dict[str, float]
            Known variable values.

        Returns
        -------
        float or list[float]
            Computed solution(s).
        '''

        unknown = self._detect_unknown(given)

        solutions = sp.solve(self.zero_expr, unknown)

        if not solutions:
            raise ValueError('No solution found.')

        results = []

        for sol in solutions:
            substituted = sol.subs(given)
            results.append(float(sp.N(substituted)))

        if len(results) == 1:
            return results[0]

        return results

    # -------------------------------------
    # Factor expression
    # -------------------------------------

    def factor(self):
        '''
        Factor the equation or expression symbolically.
        '''
        return sp.factor(self.zero_expr)

    # -------------------------------------
    # Show symbolic solution
    # -------------------------------------

    def symbolic(self, target: str):
        '''
        Return symbolic solution for a specific variable.
        '''
        if target not in self.sym_map:
            raise ValueError(f'Unknown symbol: {target}')

        symbol = self.sym_map[target]
        return sp.solve(self.zero_expr, symbol)