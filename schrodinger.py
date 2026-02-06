import numpy as np
from scipy import sparse
from skimage import measure
import torch
from torch import lobpcg
import plotly.graph_objects as go

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid and potential in 3D
N = 120
X, Y, Z = np.mgrid[-25:25:N*1j, -25:25:N*1j, -25:25:N*1j]
dx = np.diff(X[:,0,0])[0]

def get_potential(x, y, z):
    return - dx**2 / np.sqrt(x**2 + y**2 + z**2 + 1e-10)

V = get_potential(X, Y, Z)

# Discretized Hamiltonian matrix
diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
T = -1/2 * sparse.kronsum(sparse.kronsum(D,D), D)
U = sparse.diags(V.reshape(N**3), (0))
H = T + U

# Convert to PyTorch sparse tensor
H = H.tocoo()
H = torch.sparse_coo_tensor(indices=torch.tensor([H.row, H.col]), values=torch.tensor(H.data), size=H.shape).to(device)

# Solve for eigenvectors
eigenvalues, eigenvectors = lobpcg(H, k=4, largest=False)  # k kept small for demonstration

# Extract eigenfunction and surface
def get_e(n):
    return eigenvectors.T[n].reshape((N,N,N)).cpu().numpy()

verts, faces, _, _ = measure.marching_cubes(get_e(3)**2, 1e-6, spacing=(0.1, 0.1, 0.1))
intensity = np.linalg.norm(verts, axis=1)

# 3D Plotly mesh (only output)
fig = go.Figure(
    data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=intensity,
        colorscale='Agsunset',
        opacity=0.5
    )]
)
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='rgb(0, 0, 0)'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False,
    paper_bgcolor='black'
)
fig.show()
