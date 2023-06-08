import matplotlib.pyplot as plt
import torch

from nodetracker.node.core.solver.core import ODESolver
from nodetracker.node.core.solver.factory import ode_solver_factory


def show_ode_dynamics(ts: torch.Tensor, ode_true: torch.Tensor, ts_hat: torch.Tensor, zs_hat: torch.Tensor, name: str) -> None:
    plt.plot(ts, ode_true, color='red', label='True dynamics')
    plt.scatter(ts_hat, zs_hat, color='blue', s=10, label='Estimated dynamics states')
    plt.title(f'{name} - Sinusoid dynamics')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.show()


def run_ode_test(ode_solver: ODESolver, name: str):
    def cosine_dynamics(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine function

        Args:
            x: Previous state (not relevant but still keeping it as argument for consistency)
            t: Time

        Returns:
            Cosine value of t
        """
        _ = x  # void(x)
        return torch.cos(t)

    ts = torch.linspace(1, 10, 100)
    t0, t1 = torch.tensor(1.0), torch.tensor(10.0)
    z0 = torch.sin(t0)
    ode_true = torch.sin(ts)

    zs_hat, ts_hat = ode_solver(z0, t0, t1, cosine_dynamics, return_all_states=True)

    show_ode_dynamics(
        ts=ts,
        ode_true=ode_true,
        ts_hat=ts_hat,
        zs_hat=zs_hat,
        name=name
    )

def run_global_ode_test(ode_solver: ODESolver, name: str):
    def cosine_dynamics(d: torch.Tensor, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine function

        Args:
            d: Previous state (not relevant but still keeping it as argument for consistency)
            g: Global (static) state
            t: Time

        Returns:
            Cosine value of t
        """
        _, _ = d, g  # void(d, g)
        return torch.cos(t)

    ts = torch.linspace(1, 10, 100)
    t0, t1 = torch.tensor(1.0), torch.tensor(10.0)
    z0 = torch.zeros(2, dtype=torch.float32)
    z0[0] = torch.sin(t0)
    ode_true = torch.sin(ts)

    zs_hat, ts_hat = ode_solver(z0, t0, t1, cosine_dynamics, return_all_states=True)
    zs_hat_dynamic = zs_hat[:, 0]
    zs_hat_global = zs_hat[:, 1]

    assert torch.abs(zs_hat_global - torch.zeros_like(zs_hat_global)).sum() < 1e-3

    show_ode_dynamics(
        ts=ts,
        ode_true=ode_true,
        ts_hat=ts_hat,
        zs_hat=zs_hat_dynamic,
        name=name
    )


if __name__ == '__main__':
    rk4_solver = ode_solver_factory(
        name='rk4',
        params={
            'max_step_size': 0.5
        }
    )
    run_ode_test(rk4_solver, name='RK4')

    euler_solver = ode_solver_factory(
        name='euler',
        params={
            'max_step_size': 0.5
        }
    )
    run_ode_test(rk4_solver, name='Euler')

    euler_global_solver = ode_solver_factory(
        name='euler_global',
        params={
            'max_step_size': 0.25,
            'split_dim': 0
        }
    )
    run_global_ode_test(euler_global_solver, name='EulerWithGlobalState')

    rk4_global_solver = ode_solver_factory(
        name='rk4_global',
        params={
            'max_step_size': 0.25,
            'split_dim': 0
        }
    )
    run_global_ode_test(rk4_global_solver, name='RK4GlobalState')
