import src.ArrayGen as q
import time


x = time.perf_counter()
# Mandelbrot Sets:
"""Using `.Graphing`"""

# 1. Classic Mandelbrot: z = z^2 + c
formula_square = lambda z, c: z*z + c
m = q.Maths.mandelbrot_set(formula_square)
q.Graphing.mandelbrot(m)

# 2. Cubic Mandelbrot: z = z^3 + c
formula_cube = lambda z, c: z**3 + c
m = q.Maths.mandelbrot_set(formula_cube)
q.Graphing.mandelbrot(m)

# 3. Quartic Mandelbrot: z = z^4 + c
formula_quartic = lambda z, c: z**4 + c
m = q.Maths.mandelbrot_set(formula_quartic)
q.Graphing.mandelbrot(m)

# 4. Mandel bar (Tricorn): z = conjugate(z)^2 + c
formula_mandelbar = lambda z, c: (z.conjugate())**2 + c
m = q.Maths.mandelbrot_set(formula_mandelbar)
q.Graphing.mandelbrot(m)

# 5. Burning Ship fractal: z = (|Re(z)| + i|Im(z)|)^2 + c
formula_burning_ship = lambda z, c: complex(abs(z.real), abs(z.imag))**2 + c
m = q.Maths.mandelbrot_set(formula_burning_ship)
q.Graphing.mandelbrot(m)


y = time.perf_counter()

print(y-x)