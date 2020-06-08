from typing import Any, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from utils.math_ops import clamp, acosh, cosh, sinh, sqrt, logsinh, e_i, expand_proj_dims


eps = 1e-7  # TODO: Move this lower for doubles?
max_clamp_norm = 40
### Helper Function For HyperBoloid Model
def _logdet(u: Tensor, radius: Tensor, subdim=0) -> Tensor:
    # det [(\partial / \partial v) proj_{\mu}(v)] = (Rsinh(r) / r)^(n-1)
    r = lorentz_norm(u, dim=-1) / radius
    #TODO: Double check if we need to subtract 1
    n = u.shape[-1] - 1 - subdim
    # n = u.shape[-1]

    logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial

def expmap_logdet(u: Tensor, radius: Tensor, subdim=0) -> Tensor:
    # det [(\partial / \partial v) proj_{\mu}(v)] = (Rsinh(r) / r)^(n-1)
    r = lorentz_norm(u, dim=-1) / radius
    n = u.shape[-1] - 1 - subdim

    logdet_partial = (n - 1) * (torch.log(radius) + logsinh(r) - torch.log(r))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial

def logmap_logdet(u: Tensor, radius: Tensor, subdim=0) -> Tensor:
    # det [(\partial / \partial v) proj_{\mu}(v)] = (r/ Rsinh(r))^(n-1)
    r = lorentz_norm(u, dim=-1) / radius
    #TODO: Double check if we need to subtract 1
    n = u.shape[-1] - 1 - subdim
    # n = u.shape[-1]

    logdet_partial = (1 - n) * (torch.log(radius) + logsinh(r) - torch.log(r))
    assert torch.isfinite(logdet_partial).all()
    return logdet_partial

def proj_vec(u: Tensor, radius: Tensor) -> Tensor:
    # x_1 = \sqrt{||x||^2_2 + R^2}
    x_1 = sqrt(torch.norm(u, p=2, dim=1)**2 + radius**2).unsqueeze(1)
    x = torch.cat((x_1, u), dim=1)
    return x

def proj_vec_to_tang(u: Tensor, radius: Tensor) -> Tensor:
    # \proj(u)_{\mathbb{R^{D+1} \to \mathca{T}_x\mathbb{H}^d
    ipdb.set_trace()
    o = torch.zeros_like(u).to(u.device)
    o[:,0] = radius
    lorentz_prod = lorentz_product(u,o).unsqueeze(1) / radius
    x = u + lorentz_prod * o
    return x

def euc_proj_vec(u: Tensor, radius: Tensor) -> Tensor:
    # x = R*(x)/||x||_L
    r = lorentz_norm(u, dim=-1)
    x = R*(u) / r
    return x

def hyperboloid_dist(x: Tensor, y: Tensor, radius: Tensor) -> Tensor:
    # d = R*acos( -(1/R)^2 <x,y>_L)
    inv_radius = 1. / radius
    lp = -1*(inv_radius**2) * lorentz_product(x, y)
    dist = radius * acosh(lp)
    return dist

def mu_0(shape: Tuple[int, ...], radius: Tensor, **kwargs: Any) -> Tensor:
    return e_i(i=0, shape=shape, **kwargs) * radius

def lorentz_product(x: Tensor, y: Tensor, keepdim: bool = False, dim: int = -1) -> Tensor:
    try:
        m = x * y
    except:
        m = torch.mm(x,y)
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret

def lorentz_norm(x: Tensor, **kwargs: Any) -> Tensor:
    product = lorentz_product(x, x, **kwargs)
    ret = sqrt(product)
    return ret

def parallel_transport_mu0(x: Tensor, dst: Tensor, radius: Tensor) -> Tensor:
    # PT_{mu0 -> dst}(x) = x + <dst, x>_L / (R^2 - <mu0, dst>_L) * (mu0+dst)
    denom = radius * (radius + dst[..., 0:1])  # lorentz_product(mu0, dst, keepdim=True) which is -dst[0]*radius
    lp = lorentz_product(dst, x, keepdim=True)
    coef = lp / denom
    right = torch.cat((dst[..., 0:1] + radius, dst[..., 1:]), dim=-1)  # mu0 + dst
    return x + coef * right

def inverse_parallel_transport_mu0(x: Tensor, src: Tensor, radius: Tensor) -> Tensor:
    # PT_{src -> mu0}(x) = x + <mu0, x>_L / (R^2 - <src, mu0>_L) * (src+mu0)
    denom = (radius + src[..., 0:1])  # lorentz_product(src, mu0, keepdim=True) which is -src[0]*radius
    lp = -x[..., 0:1]  # lorentz_product(mu0, x, keepdim=True) which is -x[0]*radius
    # coef = (lp * radius) / (radius * denom)
    coef = lp / denom
    right = torch.cat((src[..., 0:1] + radius, src[..., 1:]), dim=-1)  # mu0 + src
    return x + coef * right

def exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    x_norm = lorentz_norm(x, keepdim=True) / radius
    x_normed = x / x_norm
    ret = cosh(x_norm) * at_point + sinh(x_norm) * x_normed
    assert torch.isfinite(ret).all()
    return ret

def exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    assert x[..., 0].allclose(torch.zeros_like(x[..., 0]))
    x = x[..., 1:]
    x_norm = torch.norm(x, p=2, keepdim=True, dim=-1) / radius
    x_normed = F.normalize(x, p=2, dim=-1) * radius
    ret = torch.cat((cosh(x_norm) * radius, sinh(x_norm) * x_normed), dim=-1)
    assert torch.isfinite(ret).all()
    return ret

def inverse_exp_map(x: Tensor, at_point: Tensor, radius: Tensor) -> Tensor:
    alpha = -lorentz_product(at_point, x, keepdim=True) / (radius**2)
    coef = acosh(alpha) / sqrt(alpha**2 - 1)
    ret = coef * (x - alpha * at_point)
    return ret

def inverse_exp_map_mu0(x: Tensor, radius: Tensor) -> Tensor:
    alpha = x[..., 0:1] / radius  # -lorentz_product(x, mu0, keepdim=True) / R^2 .. -<x, mu0>_L = x[0] * R
    coef = acosh(alpha) / sqrt(alpha**2 - 1.)
    diff = torch.cat((x[..., 0:1] - alpha * radius, x[..., 1:]), dim=-1)
    return coef * diff

def sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x_expanded = expand_proj_dims(x)
    pt = parallel_transport_mu0(x_expanded, dst=at_point, radius=radius)
    pt = clamp(pt, min=-max_clamp_norm, max=max_clamp_norm)
    x_proj = exp_map(pt, at_point=at_point, radius=radius)
    x_proj = clamp(x_proj, min=-max_clamp_norm, max=max_clamp_norm)
    return x_proj, (pt, x)

def inverse_sample_projection_mu0(x: Tensor, at_point: Tensor, radius: Tensor) -> Tuple[Tensor, Tensor]:
    unmapped = inverse_exp_map(x, at_point=at_point, radius=radius)
    # if torch.isnan(unmapped).any():
        # ipdb.set_trace()
    unmapped = clamp(unmapped, min=-max_clamp_norm, max=max_clamp_norm)
    unpt = inverse_parallel_transport_mu0(unmapped, src=at_point, radius=radius)
    unpt = clamp(unpt, min=-max_clamp_norm, max=max_clamp_norm)
    return unmapped, unpt[..., 1:]

def lorentz_to_poincare(x: Tensor, radius: Tensor) -> Tensor:
    return radius * x[..., 1:] / (radius + x[..., 0:1])
