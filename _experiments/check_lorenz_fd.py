from src.problems.lorenz96 import (
    lorenz96,
    lorenz96_grad,
)
from src.problems.lorenz96_flowmap import (
    lorenz96_flowmap,
    lorenz96_flowmap_grad
)
import numpy as np
from prajwal_lorenz_script import f_lorenz, g_lorenz

if __name__ == '__main__':

    # parameters
    forcing = 1
    tf = 1e-4
    d = 5
    out_components = [0,1]

    # functions
    def f(x):
        f = lorenz96_flowmap(x, forcing, tf)
        return f[:,out_components]

    def g(x):
        g = lorenz96_flowmap_grad(x, forcing, tf)
        return g[:,out_components,:]

    # Generate random initial condition
    x0 = np.ones((2, d))
    I = np.eye(d)
    pert = 1e-4
    grad_exact = g(x0)
    grad_exact_prajwal = g_lorenz(x0)[:,out_components,:]

    # Check finite differences
    for i in range(d): # each derivative
        pertvec = pert * I[i]
        grad_ith_component_fd = (f(x0 + pertvec) - f(x0 - pertvec)) / (2 * pert)
        grad_ith_component_exact = grad_exact[:,:,i]
        grad_ith_component_exact_prajwal = grad_exact_prajwal[:,:,i]

        print(f'Component {i}:')
        print('Exact:')
        print(grad_ith_component_exact)
        print('FD:')
        print(grad_ith_component_fd)
        print('Exact Prajwal:')
        print(grad_ith_component_exact_prajwal)
        print('\n')
