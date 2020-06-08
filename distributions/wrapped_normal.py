from typing import Any, Tuple, Optional
import torch
from torch import Tensor
from utils.math_ops import clamp, expand_proj_dims, logsinh
from utils.hyperbolics import parallel_transport_mu0, inverse_sample_projection_mu0, exp_map, inverse_exp_map, lorentz_norm
import torch.distributions

max_clamp_norm = 40

class HyperboloidWrappedNormal(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real_vector,
        "scale": torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, radius:Tensor, loc: Tensor, scale: Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = loc.shape[-1]
        self.radius = radius
        tangent_dim = self.dim - 1

        if scale.shape[-1] > 1 and scale.shape[-1] != tangent_dim:
            raise ValueError("Invalid scale dimension: neither isotropic nor elliptical.")

        if scale.shape[-1] == 1:  # repeat along last dim for (loc.shape[-1] - 1) times.
            s = [1] * len(scale.shape)
            s[-1] = tangent_dim
            scale = scale.repeat(s)  # Expand scalar scale to vector.

        # Loc has to be one dim bigger than scale or equal (in projected spaces).
        assert loc.shape[:-1] == scale.shape[:-1]
        assert tangent_dim == scale.shape[-1]

        self.loc = loc
        self.scale = scale
        self.device = self.loc.device
        smaller_shape = self.loc.shape[:-1] + torch.Size([tangent_dim])
        self.normal = torch.distributions.Normal(torch.zeros(smaller_shape, device=self.device), scale)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale

    def logdet(self, radius: Tensor, mu: Tensor, std: Tensor, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        u = data[0]
        r = lorentz_norm(u, dim=-1) / radius
        n = u.shape[-1] - 1
        logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
        assert torch.isfinite(logdet_partial).all()
        return logdet_partial

    def sample_projection_mu0(self, x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x_expanded = expand_proj_dims(x)
        pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
        pt = clamp(pt, min=-max_clamp_norm, max=max_clamp_norm)
        x_proj = exp_map(pt, at_point=at_point, radius=radius)
        return x_proj, (pt, x)

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        # v ~ N(0, \Sigma)
        v_tilde = self.normal.rsample(shape)
        assert torch.isfinite(v_tilde).all()
        # u = PT_{mu_0 -> mu}([0, v_tilde])
        # z = exp_{mu}(u)
        z, helper_data = self.sample_projection_mu0(v_tilde, at_point=self.loc, radius=self.radius)
        assert torch.isfinite(z).all()
        return z, helper_data

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        z, _ = self.rsample_with_parts(sample_shape)
        return z

    def log_prob_from_parts(self, z: Tensor, data: Optional[Tuple[Tensor, ...]]) -> Tensor:
        if data is None:
            raise ValueError("Additional data cannot be empty for WrappedNormal.")

        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        v = data[1]
        assert torch.isfinite(v).all()
        n_logprob = self.normal.log_prob(v).sum(dim=-1)
        logdet = self.logdet(self.radius, self.loc, self.scale, z, (*data, n_logprob))
        assert n_logprob.shape == logdet.shape
        log_prob = n_logprob - logdet
        assert torch.isfinite(log_prob).all()
        return log_prob

    def log_prob(self, z: Tensor) -> Tensor:
        """Should only be used for p_z, prefer log_prob_from_parts."""
        assert torch.isfinite(z).all()
        data = inverse_sample_projection_mu0(z, at_point=self.loc, radius=self.radius)
        return self.log_prob_from_parts(z, data)

    def rsample_log_prob(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(z, data)
