"""
Custom NeuralODE that accepts attributes as input in the state change function

State change function is changed from f(z, t, theta) to f(z, t, x, theta)
"""
from typing import Tuple, Any, Callable

import numpy as np
import torch
from torch import nn

from nodetracker.node.custom.solver import DefaultODESolverWithDataPoints


class ODEFWithData(nn.Module):
    """
    Parametrized dynamics function (dz/dt) superclass
    (does not have forward pass defined)
    """
    # noinspection PyMethodMayBeStatic
    def forward_with_grad(self, z: torch.Tensor, t: torch.Tensor, x: torch.Tensor, grad_outputs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs forward pass and calculates:
            - da/dt = df/dz * a
            - da_t/dt = df/dt * a
            - a_theta/dt = df/dtheta * a

        Note: Main difference between this custom

        Args:
            z: State
            t: Time
            x: Data
            grad_outputs: Grad outputs (a)

        Returns:
            z, a_dfdz, a_dfdt, a_dfdtheta
        """
        batch_size = z.shape[0]  # required for postprocessing

        z_end = self.forward(z, t, x)
        a = grad_outputs  # dL/dz(t)

        # Reference: `https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html`
        # - Autograd performs operation J.T @ v (numpy notation)
        #   where J is Jacobian and v is output gradient (example: loss)
        # - In this case: v == a, and J.T == df/dp where p is parameter (z, t or theta)
        #   so we get df/dz * a, df/dt * a and df/dtheta * a
        #   which is exactly what we need for a, a_t and a_theta
        parameters = tuple(self.parameters())
        a_dfdz, a_dfdt, *a_dfdtheta = torch.autograd.grad(
            outputs=(z_end,), # forward pass outputs
            inputs=(z, t) + parameters,  # z, t and theta / x is purposely ignored
            grad_outputs=(a,),  # gradient vector of the loss
            allow_unused=True,
            retain_graph=True
        )

        # grad function automatically sums gradients -> expanding them
        # autograd returns None if variables are not used in forward pass
        # -> replacing None values with zero gradients
        if a_dfdz is None:
            a_dfdz = torch.zeros_like(z).to(z)

        if a_dfdtheta is not None:
            a_dfdtheta = torch.cat([p_grad.flatten() for p_grad in a_dfdtheta]).unsqueeze(0)
            a_dfdtheta = a_dfdtheta.expand(batch_size, -1) / batch_size
        else:
            a_dfdtheta = torch.zeros(batch_size, len(parameters)).to(z)

        if a_dfdt is not None:
            a_dfdt = a_dfdt.expand(batch_size, 1) / batch_size
        else:
            a_dfdt = torch.zeros(batch_size, 1).to(z)

        return z_end, a_dfdz, a_dfdt, a_dfdtheta

    @property
    def flat_parameters(self):
        """
        Returns: Returns flat parameters tensor
        """
        return torch.cat([p.flatten() for p in self.parameters()])


class ODEAdjointWithData(torch.autograd.Function):
    """
    Implementation of Pontryagin Adjoint method
    """
    @staticmethod
    def create_aug_dynamics(batch_size, shape, func) -> Callable:
        """
        Creates augmented dynamics function

        Args:
            batch_size: Batch size
            shape: Shape of z tensor
            func: State change function

        Returns:
            augmented dynamics functions
        """
        # noinspection PyTypeChecker
        n_dim = round(np.prod(shape))

        def aug_dynamics(aug_z: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """
            Augmented state change function

            Args:
                aug_z: flat tensor (z, a, a_t, a_theta) with shape (batch_size, n_dim+ndim+1+n_params)
                    where n_dim is z flat tensor dimension and n_params is theta (params) flat tensor dimensions
                t: Time
                x: Data

            Returns:
                Augmented (z_end, -a_dfdz, -a_dfdt, -a_dfdtheta) as flat tensor
            """
            # Input is flat (z, a, a_t, a_theta) - ignore a_t and a_theta
            z, a = aug_z[:, :n_dim], aug_z[:, n_dim:2*n_dim]

            z = z.view(batch_size, *shape)
            a = a.view(batch_size, *shape)
            with torch.set_grad_enabled(True):
                z = z.requires_grad_()
                t = t.requires_grad_()
                z_end, a_dfdz, a_dfdt, a_dfdtheta = func.forward_with_grad(z, t, x, grad_outputs=a)

            # Output has to be single tensor
            # In adjoint a(t) formula we use negative value of integral
            z_end = z_end.view(batch_size, n_dim) # flatten for concat
            a_dfdz = a_dfdz.view(batch_size, n_dim)  # flatten for concat
            return torch.cat([z_end, -a_dfdz, -a_dfdtheta, -a_dfdt], dim=-1)  # TODO: Reorder dfdt and dfdtheta

        return aug_dynamics

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx: Any, z0: torch.Tensor, t: torch.Tensor, x: torch.Tensor, flat_theta: torch.Tensor, func: ODEFWithData) -> torch.Tensor:
        # z.shape == (time_len, batch_size, *vector_shape)
        # z0.shape == (batch_size, *vector_shape)
        # t.shape == (time_len, 1, 1)
        assert len(t.shape) == 3, f'Expected shape is (time_len, 1, 1) but found {t.shape}'
        assert x.shape[0] == t.shape[0], f'Number of data and time points do not match: {x.shape[0]} > {t.shape[0]}'
        n_steps = t.shape[0]

        zs = [z0]
        with torch.no_grad():
            # ODESolver is used a blackbox (disable autograd)
            for i in range(n_steps - 1):
                z0, _ = DefaultODESolverWithDataPoints.solve(z0, t[i], t[i+1], x[i], func)
                zs.append(z0)

        zs = torch.stack(zs)  # Stack all states into a single vector
        ctx.func = func
        ctx.save_for_backward(zs, t, x, flat_theta)
        return zs

    # noinspection PyMethodOverriding
    @staticmethod
    def jvp(ctx: Any, dLdz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        z, t, x, flat_parameters = ctx.saved_tensors
        time_len, batch_size, *z_shape = z.size()
        # noinspection PyTypeChecker
        n_dim = round(np.prod(z_shape))
        n_params = flat_parameters.shape[0]

        augmented_dynamics = ODEAdjointWithData.create_aug_dynamics(batch_size, z_shape, func)

        dLdz = dLdz.view(time_len, batch_size, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            # Variables to store adjoint states
            a = torch.zeros(batch_size, n_dim).to(dLdz)
            a_theta = torch.zeros(batch_size, n_params).to(dLdz)
            a_t = torch.zeros(time_len, batch_size, 1).to(dLdz)  # adjoint t needs to be stored for all timestamps

            for i_t in range(time_len-1, 0, -1):
                # Iteration steps:
                # 1. Form augmented state
                # 2. Use ODESolve with augmented dynamics
                # Calculate z backward in time (required for a(t) evaluation and augmented state)
                z_i = z[i_t]
                t_i = t[i_t]
                x_i = x[i_t]
                dzdt_i = func(z_i, t_i, x_i).view(batch_size, n_dim)

                # Math details: For augmented state we need (z, a, a_t, a_theta)
                # a_t = dL/dt
                # dL/dt = dL/dz(t) @ dz(t)/dt
                #       = a(t) * func(z, i, theta)
                # Note: we don't need theta in implementation since it is part of func (Module object)
                # Implementation details: L is scalar, t is scalar, z is vector of dimension n
                # hence dLdz shape is (1, n) and dzdt shape is (1, n) and dLdt shape is (1, 1) - vector scalar product
                dLdz_i = dLdz[i_t]
                # bmm is batch matrix multiplication (in parallel)
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), dzdt_i.unsqueeze(-1))[:, 0]

                # Form augmented state z0_aug = (z, a, a_t, a_theta) = (z, dL/dz, dL/dt, dL/dtheta)
                # We need to sum up gradients of loss function for all timestamps backward in time
                # a_theta is set to 0
                a = a + dLdz_i
                a_t[i_t] = a_t[i_t] - dLdt_i
                a_theta_i = torch.zeros(batch_size, n_params).to(z_i)  # We defined dL/dtheta as 0
                z_i = z_i.view(batch_size, n_dim) # flatten for concat
                z0_aug = torch.cat([z_i, a, a_t[i_t], a_theta_i], dim=-1)

                # Solve augmented system backwards
                dzdt_aug, _ = DefaultODESolverWithDataPoints.solve(z0_aug, t_i, t[i_t-1], x_i, augmented_dynamics)

                # Unpack solved backwards augmented system
                a[:] = dzdt_aug[:, n_dim:2*n_dim]
                a_theta[:] += dzdt_aug[:, 2*n_dim:2*n_dim + n_params]
                a_t[i_t-1] = dzdt_aug[:, 2*n_dim + n_params:]

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dzdt_0 = func(z[0], t[0])
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), dzdt_0.view(batch_size, -1).unsqueeze(-1))[:, 0]

            # Adjust adjoint states
            a += dLdz_0
            a_t[0] = a_t[0] - dLdt_0
        return a.view(batch_size, *z_shape), a_t, a_theta, None

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, dLdz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Alias for jvp
        return ODEAdjointWithData.jvp(ctx, dLdz)


class NeuralODE(nn.Module):
    """
    Wrapper for ADJoint function
    """
    def __init__(self, func: ODEFWithData):
        super().__init__()
        self.func = func

    def forward(self, z0: torch.Tensor, t: torch.Tensor, x: torch.Tensor, full_sequence: bool = False) -> torch.Tensor:
        z = ODEAdjointWithData.apply(z0, t, x, self.func.flat_parameters, self.func)

        if full_sequence:
            return z
        return z[-1]


def main() -> None:
    class LinearODEF(ODEFWithData):
        """
        Simple ODEF for testing
        """
        def __init__(self):
            super().__init__()
            self._linear = nn.Linear(5, 2, bias=False)

        def forward(self, z: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            _ = t
            ztx = torch.cat([z, t, x], dim=-1)
            return self._linear(ztx)

    node = NeuralODE(LinearODEF())
    z0 = torch.randn(2).view(1, 2)
    ts = torch.randn(3).view(-1, 1, 1)
    xs = torch.randn(3, 2).view(3, 1, 2)
    print('Last state output shape:', node(z0, ts, xs, full_sequence=False).shape)
    print('Full state sequence output shape:', node(z0, ts, xs, full_sequence=True).shape)


if __name__ == '__main__':
    main()
