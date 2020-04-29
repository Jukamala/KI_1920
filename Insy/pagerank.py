import numpy as np

a = np.array([[0,1,0,0,0,0],
              [0, 0, 0.5, 0.5, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
              [0, 1, 0, 0, 0, 0],
              [0, 0.5, 0, 0.5, 0, 0]])

P = 0.9 * a + 0.1*1/6

x = np.ones(6) * 1/6
while True:
    x_old = x.copy()
    x = np.dot(x, P)

    if np.linalg.norm(x - x_old) < 1e-5:
        break

print(" ".join(["%.2f" % v for v in x]))


