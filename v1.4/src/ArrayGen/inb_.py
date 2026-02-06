import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
try:
    import ipywidgets as widgets
    from ipywidgets import interact

except ImportError:
    raise ImportError('Please install ipywidgets first!')

class inb_:
    @staticmethod
    def _mandelbrot_set_plot(
        formula_str="z**2 + c",
        width=300, height=300,
        max_iterations=100,
        xmin=-2.0, xmax=1.0,
        ymin=-1.5, ymax=1.5
    ):
        z, c = sp.symbols('z c')
        try:
            formula_expr = sp.sympify(formula_str)
            custom_formula = sp.lambdify((z, c), formula_expr, modules=["numpy"])
        except Exception as e:
            print("Invalid formula:", e)
            return

        real_axis = np.linspace(xmin, xmax, width)
        imag_axis = np.linspace(ymin, ymax, height)
        C = real_axis[None, :] + 1j * imag_axis[:, None]
        Z = np.zeros(C.shape, dtype=complex)
        N = np.zeros(C.shape, dtype=int)
        mask = np.ones(C.shape, dtype=bool)

        for n in range(max_iterations):
            Z[mask] = custom_formula(Z[mask], C[mask])
            mask_new = (np.abs(Z) <= 2)
            N[mask & ~mask_new] = n
            mask = mask & mask_new
            if not mask.any():
                break
        N[N == 0] = max_iterations

        plt.figure(figsize=(7, 6))
        plt.imshow(N, cmap="hot", extent=[xmin, xmax, ymin, ymax], origin='lower')
        plt.colorbar(label='Iterations')
        plt.title(f"Fractal for: {formula_str}")
        plt.xlabel("Re(c)")
        plt.ylabel("Im(c)")
        plt.show()

    @staticmethod
    def mandelbrot_set_widget():
        interact(
            inb_._mandelbrot_set_plot,
            formula_str=widgets.Text(
                value="z**2 + c",
                description="Iter. Formula:",
                layout=widgets.Layout(width="70%")
            ),
            width=widgets.IntSlider(value=100, min=100, max=600, step=50, description="Width"),
            height=widgets.IntSlider(value=100, min=100, max=600, step=50, description="Height"),
            max_iterations=widgets.IntSlider(value=50, min=10, max=400, step=10, description="Max Iter"),
            xmin=widgets.FloatSlider(value=-2.0, min=-3.0, max=0.0, step=0.1, description="x min"),
            xmax=widgets.FloatSlider(value=1.0, min=0.0, max=3.0, step=0.1, description="x max"),
            ymin=widgets.FloatSlider(value=-1.5, min=-3.0, max=0.0, step=0.1, description="y min"),
            ymax=widgets.FloatSlider(value=1.5, min=0.0, max=3.0, step=0.1, description="y max"),
        )

    @staticmethod
    def _projectile_motion_plot(
        formula_str="x*tan(theta) - (g/(2*u**2*cos(theta)**2))*x**2",
        u=10.0, theta_deg=45.0, g=9.8, x_max=11.0
    ):
        x = sp.symbols('x')
        formula = sp.sympify(formula_str)
        theta = np.deg2rad(theta_deg)
        x_vals = np.linspace(0, x_max, 300)
        y_expr = formula.subs({'theta': theta, 'u': u, 'g': g})
        y_func = sp.lambdify(x, y_expr, modules=['numpy'])
        y_vals = y_func(x_vals)
        plt.figure(figsize=(7, 4))
        plt.plot(x_vals, y_vals, label="Trajectory")
        plt.title(f"Projectile Trajectory\nFormula: {formula_str}")
        plt.xlabel("Horizontal Distance (x)")
        plt.ylabel("Vertical Distance (y)")
        plt.ylim(bottom=0)
        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def projectile_motion_widget():
        interact(
            inb_._projectile_motion_plot,
            formula_str=widgets.Text(
                value="x*tan(theta) - (g/(2*u**2*cos(theta)**2))*x**2",
                description="y(x) Formula:",
                layout=widgets.Layout(width="80%")
            ),
            u=widgets.FloatSlider(min=1, max=50, step=0.1, value=10, description="u (velocity)"),
            theta_deg=widgets.FloatSlider(min=0, max=90, step=1, value=45, description="theta (angle)", continuous_update=False),
            g=widgets.FloatSlider(min=1, max=20, step=0.1, value=9.8, description="g (gravity)"),
            x_max=widgets.FloatSlider(min=5, max=50, step=0.5, value=11, description="Max X")
        )
