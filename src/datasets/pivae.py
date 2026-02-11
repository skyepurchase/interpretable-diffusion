from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader

from utils.config.datasets import DatasetConfig

# TODO: Make sure this is vectorised!!
# TODO: Work towards saving this dataset, though I don't imagine it getting much use


def make_ODE() -> Callable[[NDArray, NDArray], NDArray]:
    omega = (np.random.randint(0, 1000) * (3.14 - 0.785) / 1000) + 0.785
    xi = np.random.randint(0, 1000) * 8 / 1000
    phi = (np.random.randint(0, 1000) * (6.280 - 3.14) / 1000) + 3.14
    A = np.random.randint(0, 1000) * 40 / 1000

    def force_damped_pendulum(t: NDArray, y: NDArray) -> NDArray:
        theta, theta_dot = y

        dtheta = theta_dot  # first derivative is velocity
        dtheta_dot = (
            (A * omega**2 * np.cos(2 * np.pi * phi * t))
            - (xi * theta_dot)
            - (omega**2 * np.sin(theta))
        )  # second derivative according to force dampened pendulum

        return np.array([dtheta, dtheta_dot])

    return force_damped_pendulum


def get_pendulum(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader]:
    y_0 = (np.random.randint(0, 1000) * (1.57 + 1.57) / 1000) - 1.57
    y0 = [y_0, 0]

    steps = 50
    T = steps * 0.05
    t_span = (0, T)

    sol = solve_ivp(
        fun=make_ODE(),
        t_span=t_span,
        y0=y0,
        t_eval=np.linspace(0, T, steps + 1),
        method="DOP853",
        rtol=1e-3,
        atol=1e-6,
    )

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Subplot for angle
    plt.subplot(1, 1, 1)
    plt.plot(sol.t, sol.y[0])
    plt.title("Force Damped Pendulum Oscillation")
    plt.ylabel(r"$\vartheta(t)$")
    plt.ylim(bottom=-2, top=2)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("test.png", format="png")

    quit(0)
