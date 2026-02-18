import src.ArrayGen_v2 as q

X, Y, Z = q.expr('x^2 - y^2 = 1', step=0.1)

import matplotlib.pyplot as plt
plt.contour(X, Y, Z, levels=[0])
plt.gca().set_aspect('equal')
plt.show()