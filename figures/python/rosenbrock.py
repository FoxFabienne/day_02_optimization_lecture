import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
import tikzplotlib

from util import write_movie


def rosenbrock(x, a=1., b=100.):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


def rosen_grad(x, a=1., b=100.):
    return np.array(
        [-2.*a + 2*x[0] - 4*b*x[1]*x[0] + 4*b*x[0]**3,
        2*b*x[1] - 2*b*x[0]**2]
    )


if __name__ == '__main__':
    x = np.linspace(-2, 2, num=50)
    y = np.linspace(-1, 3, num=50)
    mx, my = np.meshgrid(x, y)
    x = np.stack((mx, my))
    rose = rosenbrock(x)

    jax_grad_ros = jax.grad(rosenbrock)

    grad_field = rosen_grad(x)
    sign = np.sign(grad_field)
    rescaled = sign*np.log(np.abs(grad_field))

    plt.contourf(mx, my, rose,
        locator=ticker.LogLocator(), cmap=cm.viridis)
    plt.quiver(mx, my, rescaled[0], rescaled[1], headwidth=2, headlength=4)
    # plt.savefig('quiver.png', transparent=True, dpi=400)
    plt.show()



    start_pos = np.array((0.1, 3.))
    step_size = 0.01
    alpha = 0.8 # 0.8, 0.0
    step_total = 600

    pos_list = [start_pos]
    grad = np.array((0.0, 0.0))
    for _ in range(step_total):
        #nabla = jax_grad_ros(pos_list[-1])
        nabla = rosen_grad(pos_list[-1])
        grad = grad * alpha - step_size * nabla /np.linalg.norm(nabla)
        pos = pos_list[-1] + grad
        pos_list.append(pos)

    plt.contourf(mx, my, rose, locator=ticker.LogLocator(), cmap=cm.viridis)
    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    plt.show()

    write_movie(mx, my, rose, pos_list, 
        "rosenbrock_gif_momentum", xlim=(-2, 2), ylim=(-1,3))