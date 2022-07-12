import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
import tikzplotlib

from util import write_movie


def rosenbrock(x, a=1., b=100.):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2



if __name__ == '__main__':
    x = np.linspace(-2, 2, num=50)
    y = np.linspace(-1, 3, num=50)
    mx, my = np.meshgrid(x, y)
    x = np.stack((mx, my))
    rose = rosenbrock(x)
    plt.contourf(mx, my, rose,
        locator=ticker.LogLocator(), cmap=cm.viridis)
    plt.colorbar()
    plt.show()

    grad_ros = jax.grad(rosenbrock)

    start_pos = np.array((0.1, 3.))
    step_size = 0.01
    alpha = 0.0 # 0.8, 0.0
    step_total = 600

    pos_list = [start_pos]
    grad = np.array((0.0, 0.0))
    for _ in range(step_total):
        nabla = grad_ros(pos_list[-1])
        grad = grad * alpha - step_size * nabla /np.linalg.norm(nabla)
        pos = pos_list[-1] + grad
        pos_list.append(pos)

    plt.contourf(mx, my, rose, locator=ticker.LogLocator(), cmap=cm.viridis)
    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    plt.show()

    write_movie(mx, my, rose, pos_list, "rosenbrock_gif_momentum", xlim=(-2, 2), ylim=(-1,3))