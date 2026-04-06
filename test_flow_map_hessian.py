import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

def van_der_pol(x, mu=1):
    return np.array([
        mu * (x[0] - x[0]**3/3 - x[1]),
        x[0] / mu
    ])

def generate_grid(domain, n=100):
    xgrid = np.linspace(domain[0,0], domain[0,1], n)
    ygrid = np.linspace(domain[1,0], domain[1,1], n)
    xv, yv = np.meshgrid(xgrid, ygrid)
    return xv, yv

def compute_trajectory(x0, tf, mu=1):
    rhs = lambda t, y: van_der_pol(y, mu)
    sol = solve_ivp(rhs, [0, tf], x0, method='RK45', rtol=1e-10)
    return sol.y

def van_der_pol_flow_map(x, dt=0.5, mu=1):
    rhs = lambda t, y: van_der_pol(y, mu)
    sol = solve_ivp(rhs, [0, dt], x, method='RK45')
    return sol.y.T[-1]

def generate_phase_plot():
    domain = np.array([
        [-3, 3],
        [-3, 3]
    ])
    xv, yv = generate_grid(domain, n=50)

    # Compute
    U, V = np.zeros(xv.shape), np.zeros(yv.shape)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            U[i, j], V[i, j] = van_der_pol([xv[i,j], yv[i,j]])

    # Plot
    fig, ax = plt.subplots(figsize=(8,8))
    ax.quiver(xv, yv, U, V, cmap='viridis')

    # Optional: add a trajectory
    y = compute_trajectory(np.array([1,1.5]), 50, 1)
    ax.plot(y[0], y[1], '-b')

    fig.savefig('van_der_pol_phase_plot.png')


def generate_flow_map_plots(dt=np.array([0.1, 0.5, 1.0, 5.0, 10.0, 20.0]), mu=1):

    assert len(dt) == 6

    domain = np.array([
        [-4, 4],
        [-4, 4]
    ])
    xgrid, ygrid = generate_grid(domain, n=100)
    
    # Compute phase plot
    # x_grid_small, y_grid_small = generate_grid(domain, n=50)
    # U, V = np.zeros(x_grid_small.shape), np.zeros(y_grid_small.shape)
    # for i in range(x_grid_small.shape[0]):
    #     for j in range(x_grid_small.shape[1]):
    #         U[i, j], V[i, j] = van_der_pol([x_grid_small[i,j], y_grid_small[i,j]])

    fig = plt.figure(figsize=(12, 10))

    # Outer 3x2 grid
    outer_grid = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.4)

    for k in range(6):
        # Create nested grid inside current outer cell
        # 1 row, 3 columns (Plot 1, Plot 2, and a slim column for the Colorbar)
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_grid[k],
            width_ratios=[1, 1, 0.1], wspace=0.1
        )

        ax1 = fig.add_subplot(inner_grid[0])
        ax2 = fig.add_subplot(inner_grid[1])

        cax = fig.add_subplot(inner_grid[2])

        # Compute flow map
        x_tf, y_tf = np.zeros(xgrid.shape), np.zeros(ygrid.shape)
        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):
                x_tf[i, j], y_tf[i, j] = van_der_pol_flow_map([xgrid[i,j], ygrid[i,j]], dt[k], mu)

        zmin = min(z.min() for z in (x_tf, y_tf))
        zmax = max(z.max() for z in (x_tf, y_tf))
        levels = np.linspace(zmin,zmax,21)
        ax1.contourf(xgrid, ygrid, x_tf, levels=levels, cmap='viridis')
        ax1.set_title(f'$x(\\Delta t = {dt[k]})$')
        cf = ax2.contourf(xgrid, ygrid, y_tf, levels=levels, cmap='viridis')
        ax2.set_title(f'$y(\\Delta t = {dt[k]})$')
        fig.colorbar(cf, cax=cax)

    plt.suptitle(f'Van der Pol Oscillator Flow Maps ($\\mu={mu}$)')
    plt.tight_layout()
    plt.savefig(f'van_der_pol_flow_map_plot_mu_{mu}.png')


if __name__ == '__main__':
    generate_flow_map_plots(mu=1)
