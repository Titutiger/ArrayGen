from src.ArrayGen import utils

import numpy as np
import plotly.graph_objects as go

class Chem:
    """
    Chemical visualization utilities for atomic electron shell modeling.

    This class provides a namespace for builder tools related to chemistry.
    See Chem.Builder for specific building functions.

    """
    class Builder:
        """
        Static builder class for constructing atomic electron shell visualizations.

        Methods
        -------
        sphere_points(n)
            Compute evenly distributed points on the surface of a sphere.

        electron_positions(shells, max_electrons, shell_radius_step=1.5)
            Generate electron positions by subshell following the Aufbau principle.

        build_atom(atom='O', level='shell')
            Build electron spatial positions and shell/orbital labels for given atom.

        create_interactive_atom_html(atom='O', filename='atom_all_levels.html')
            Write interactive 3D atomic model (shell, subshell, orbital) to one HTML file.

        """

        ATOM_CONFIGS = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
            'Cl': 17, 'Ar': 18
        }

        AUFBAU = [
            (1, 's', 2),
            (2, 's', 2),
            (2, 'p', 6),
            (3, 's', 2),
            (3, 'p', 6),
            (4, 's', 2),
        ]

        @staticmethod
        def sphere_points(n):
            """
            Evenly distribute n points on the surface of a sphere using the
            Golden Section Spiral (Fibonacci sphere).

            Parameters
            ----------
            n : int
                Number of points (electrons) to distribute.

            Returns
            -------
            points : (n, 3) ndarray
                Cartesian coordinates of points on the sphere.
            """
            pts = []
            inc = np.pi * (3 - np.sqrt(5))
            off = 2 / n
            for k in range(n):
                y = k * off - 1 + (off / 2)
                r = np.sqrt(1 - y * y)
                phi = k * inc
                x, z = np.cos(phi) * r, np.sin(phi) * r
                pts.append([x, y, z])
            return np.array(pts)

        @staticmethod
        def electron_positions(shells, max_electrons, shell_radius_step=1.5):
            """
            Generate electron positions by subshell following the Aufbau order.

            Parameters
            ----------
            shells : list of tuple of (int, str, int)
                List of (n, subshell, electrons_in_subshell).
            max_electrons : int
                Maximum electrons for the atom.
            shell_radius_step : float, optional
                Scaling factor for shell radii (default: 1.5).

            Returns
            -------
            positions : (total_electrons, 3) ndarray
                Cartesian electron positions.
            shell_ids : list
                Shell quantum number for each electron.
            orbital_ids : list
                Subshell label for each electron, e.g. '2p'.
            """
            positions = []
            shell_ids = []
            orbital_ids = []
            e_num = 0
            for shell, subshell, capacity in shells:
                n_e = min(capacity, max_electrons - e_num)
                if n_e == 0:
                    break
                pts = Chem.Builder.sphere_points(n_e) * (shell * shell_radius_step)
                positions.extend(pts)
                shell_ids.extend([shell] * n_e)
                orbital_ids.extend([f"{shell}{subshell}"] * n_e)
                e_num += n_e
            return np.array(positions), shell_ids, orbital_ids

        @staticmethod
        def build_atom(atom='O', level='shell'):
            """
            Build 3D positions and labels representing an atom's electron configuration at a given level.

            Parameters
            ----------
            atom : str, optional
                Atom symbol (e.g. 'O' for oxygen). Default is 'O'.
            level : {'shell', 'subshell', 'orbital'}, optional
                Detail level: group all electrons by shell, by shell+subshell, or unique orbitals (default: 'shell').

            Returns
            -------
            positions : (N, 3) ndarray
                Cartesian coordinates for all electrons.
            shell_labels : list of str
                Shell or shell+subshell (or orbital) label for each electron.
            orbital_labels : list of str
                Subshell or orbital label for each electron.

            Raises
            ------
            ValueError
                If the atom is not supported or level is not valid.
            """
            ATOM_CONFIGS = Chem.Builder.ATOM_CONFIGS
            AUFBAU = Chem.Builder.AUFBAU
            if atom not in ATOM_CONFIGS:
                raise ValueError(f"Unknown atom: {atom}")
            max_electrons = ATOM_CONFIGS[atom]
            shells_to_use = []
            count = 0
            for order in AUFBAU:
                n, subshell, num = order
                if count + num < max_electrons:
                    shells_to_use.append(order)
                    count += num
                else:
                    shells_to_use.append((n, subshell, max_electrons - count))
                    break
            if level == 'shell':
                summary = {}
                for shell, subshell, cap in shells_to_use:
                    summary.setdefault(shell, 0)
                    summary[shell] += cap
                sorted_shells = sorted(summary.items())
                positions = []
                shell_labels = []
                for sh, n in sorted_shells:
                    pts = Chem.Builder.sphere_points(n) * (sh * 2)
                    positions.extend(pts)
                    shell_labels.extend([f"{sh}"] * n)
                positions = np.array(positions)
                orbital_labels = shell_labels
            elif level in ['subshell', 'orbital']:
                positions, shell_labels, orbital_labels = Chem.Builder.electron_positions(shells_to_use, max_electrons)
            else:
                raise ValueError("level must be 'shell', 'subshell', or 'orbital'")
            return positions, shell_labels, orbital_labels

        @staticmethod
        def Electron_Shell(atom='O', filename='atom_all_levels.html'):
            """
            Generate and save a single HTML file containing an interactive Plotly
            visualization of the atomic shells, subshells, and orbitals.

            User can select the view (shell / subshell / orbital) via a dropdown.

            Parameters
            ----------
            atom : str, optional
                Atom symbol, e.g. 'O', 'He', 'Ar'. Must be present in ATOM_CONFIGS.
            filename : str, optional
                Output filename for the HTML file (e.g. 'O_all_levels.html').
            """
            colors = {'shell': 'blue', 'subshell': 'green', 'orbital': 'red'}
            levels = ['shell', 'subshell', 'orbital']
            data = []
            # Prepare data for three levels
            for level_i, level in enumerate(levels):
                positions, shells, orbitals = Chem.Builder.build_atom(atom, level)
                # Render translucent spheres for each shell for visual structure
                unique_shells = sorted(set(shells), key=lambda x: float(str(x).replace('s', '').replace('p', '')))
                for s in unique_shells:
                    R = int(str(s)[0]) * 2 if isinstance(s, str) else int(s) * 2
                    phi, theta = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
                    xs = R * np.sin(theta) * np.cos(phi)
                    ys = R * np.sin(theta) * np.sin(phi)
                    zs = R * np.cos(theta)
                    surf = go.Surface(
                        x=xs, y=ys, z=zs, opacity=0.08,
                        showscale=False,
                        colorscale=[[0, colors[levels[level_i]]], [1, colors[levels[level_i]]]],
                        visible=(level_i == 0)
                    )
                    data.append(surf)
                # Electrons as colored 3D markers
                electron_trace = go.Scatter3d(
                    x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                    mode='markers',
                    marker=dict(size=6, color=list(range(len(positions))),
                                colorscale='Turbo', opacity=0.9),
                    text=orbitals,
                    hovertemplate="Electron in %s orbital: %%{text}<extra></extra>" % level,
                    visible=(level_i == 0)
                )
                data.append(electron_trace)
            # Configure interactive dropdown for level selection
            n_surface_per_level = [len(set(Chem.Builder.build_atom(atom, l)[1])) for l in levels]
            ndata_per_level = [n + 1 for n in n_surface_per_level]
            buttons = []
            i0 = 0
            for level_i, level in enumerate(levels):
                vis = []
                for i in range(len(data)):
                    in_level = (i >= i0) and (i < i0 + ndata_per_level[level_i])
                    vis.append(in_level)
                label = level.capitalize()
                buttons.append(dict(
                    label=label,
                    method='update',
                    args=[{'visible': vis},
                          {'title': f'Atomic model of {atom}: {label} level'}]
                ))
                i0 += ndata_per_level[level_i]
            fig = go.Figure(data=data)
            fig.update_layout(
                updatemenus=[{
                    'buttons': buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.04,
                    'y': 1.17,
                    'xanchor': 'left',
                    'yanchor': 'top'
                }],
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                title=f'Atomic model of {atom}: Shell level',
                margin=dict(l=0, r=0, b=0, t=30),
                width=650,
                height=650
            )

            ext = 'html'
            out_dir = utils.get_output_dir(ext)
            filepath = utils.get_unique_filename(out_dir, base='output', ext=ext)

            fig.write_html(str(filepath))
            print(f"Wrote to {filepath}")
            


if __name__ == "__main__":
    atom = 'Cl'  # Atom symbol (must be present in ATOM_CONFIGS)
    Chem.Builder.Electron_Shell(atom, f"{atom}_all_levels.html")
