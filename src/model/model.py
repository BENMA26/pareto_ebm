import os
import copy
import numpy as np
import random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.nn.functional import cosine_similarity
from torch.nn import Dropout
import torch.distributed as dist
from itertools import combinations
import torchvision
import pytorch_lightning as pl
from model.sampler import ReplayBuffer
from model.utils import GaussianSmoothing
from model.constant import CONFLICTS_LIST,ATTRIBUTE_LIST
from sklearn.metrics import roc_auc_score

from torchjd import backward,mtl_backward
from torchjd.aggregation import UPGrad

from PIL import Image
from torchvision import transforms
from model.data import get_color_distortion

@torch.no_grad()
def _solve_min_vGv_lower_bound(G: torch.Tensor,
                              u: torch.Tensor,
                              iters: int = 50) -> torch.Tensor:
    """
    Solve:   min_v  v^T G v    s.t.  v >= u   (elementwise)
    via projected gradient descent on v, projection is v <- max(v, u).
    G: [m, m] SPD
    u: [m]
    return v*: [m]
    """
    m = G.shape[0]
    assert G.shape == (m, m)
    assert u.shape == (m,)

    # Lipschitz constant of grad( v^T G v ) is 2 * lambda_max(G)
    # step = 1 / (2 * lambda_max)
    lam_max = torch.linalg.eigvalsh(G).max().clamp_min(1e-12)
    step = 1.0 / (2.0 * lam_max)

    v = u.clone()  # feasible init
    for _ in range(iters):
        grad = 2.0 * (G @ v)
        v = torch.maximum(u, v - step * grad)
    return v

def upgrad_manual_for_x(target_list,
                        x: torch.Tensor,
                        pref_vector: torch.Tensor | None = None,
                        norm_eps: float = 1e-4,
                        reg_eps: float = 1e-4,
                        qp_iters: int = 50):
    """
    Manual UPGrad for your sampling variable x.

    Steps:
    1) compute per-objective gradients g_i = d target_i / d x
    2) build Gram G from *normalized* gradients (for numerical stability)
    3) for each i: solve w_i = argmin_{v >= e_i} v^T G v
    4) combine weights w_bar = sum_i alpha_i * w_i   (alpha from pref_vector)
    5) aggregated grad = sum_k w_bar[k] * g_k   (use original gradients)
    """
    device = x.device
    m = len(target_list)
    assert m >= 1

    # ---- 1) grads per objective (keep graph for multi grads) ----
    grads = []
    for t in target_list:
        g = torch.autograd.grad(t, x, retain_graph=True, create_graph=False)[0]
        grads.append(g)

    # ---- 2) Gram matrix from normalized gradients ----
    # Normalize each g_i by its L2 norm to improve conditioning (TorchJD also does this). :contentReference[oaicite:2]{index=2}
    flat = [g.reshape(-1) for g in grads]
    norms = torch.stack([f.norm() for f in flat])  # [m]
    flat_hat = [flat[i] / (norms[i] + norm_eps) for i in range(m)]

    G = torch.empty((m, m), device=device, dtype=flat_hat[0].dtype)
    for i in range(m):
        for j in range(i, m):
            val = torch.dot(flat_hat[i], flat_hat[j])
            G[i, j] = val
            G[j, i] = val
    G = G + reg_eps * torch.eye(m, device=device, dtype=G.dtype)  # ensure PD :contentReference[oaicite:3]{index=3}

    # ---- 3) preferences (alpha) ----
    if pref_vector is None:
        alpha = torch.full((m,), 1.0 / m, device=device, dtype=G.dtype)
    else:
        assert pref_vector.shape == (m,)
        # Require only non-negative weights; normalize to sum to 1 for stability.
        alpha = (pref_vector.to(device=device, dtype=G.dtype)).clamp_min(0)
        alpha = alpha / alpha.sum().clamp_min(1e-12)

    # ---- 4) solve w_i QP for each i, then weighted combine ----
    w_bar = torch.zeros((m,), device=device, dtype=G.dtype)
    for i in range(m):
        u = torch.zeros((m,), device=device, dtype=G.dtype)
        u[i] = 1.0  # e_i
        w_i = _solve_min_vGv_lower_bound(G, u, iters=qp_iters)  # eq (5) :contentReference[oaicite:4]{index=4}
        w_bar += alpha[i] * w_i

    # ---- 5) aggregated gradient: J^T w_bar ----
    # Use ORIGINAL grads (not normalized) so scaling behaves correctly.
    agg = torch.zeros_like(x)
    for k in range(m):
        grad_index = grads[k] * w_bar[k].item()
        print(f"{k}: {grad_index.min()} {grad_index.max()}")
        agg.add_(grad_index)
        #agg.add_(grads[k], alpha=w_bar[k].item())

    return agg, w_bar

def upgrad_12_plus_g3(
    target_list,
    x: torch.Tensor,
    pref_vector_12: torch.Tensor | None = None,
    weight_g3: float = 1.0,
    norm_eps: float = 1e-4,
    reg_eps: float = 1e-4,
    qp_iters: int = 50,
    verbose: bool = False,
):
    """
    Use UPGrad on gradients of target_list[0], target_list[1] only,
    then add gradient of target_list[2] (not participating) to form final gradient.

    Returns:
        grad_final:  g_up(1,2) + weight_g3 * g3
        grad_up12:   UPGrad aggregation result from (g1,g2)
        grad3:       gradient of target3
        w_bar12:     weights used in J^T w_bar for UPGrad part (length 2)
    """
    assert len(target_list) >= 3, "Need at least 3 targets: [t1, t2, t3, ...]"
    device = x.device
    dtype = x.dtype

    # ---- 1) compute g1,g2,g3 w.r.t x ----
    t1, t2, t3 = target_list[0], target_list[1], target_list[2]

    g1 = torch.autograd.grad(t1, x, retain_graph=True, create_graph=False)[0]
    g2 = torch.autograd.grad(t2, x, retain_graph=True, create_graph=False)[0]
    g3 = torch.autograd.grad(t3, x, retain_graph=True, create_graph=False)[0]
    g1 = torch.clamp(g1, min=-0.03, max=0.03)
    g2 = torch.clamp(g2, min=-0.03, max=0.03)
    #g3 = torch.clamp(g3, min=-0.03, max=0.03)
    grads12 = [g1, g2]
    m = 2

    # ---- 2) Gram matrix from normalized gradients (ONLY for g1,g2) ----
    flat = [g.reshape(-1) for g in grads12]
    norms = torch.stack([f.norm() for f in flat]).to(device=device, dtype=torch.float32)  # [2]
    flat_hat = [(flat[i] / (norms[i].to(dtype=flat[i].dtype) + norm_eps)) for i in range(m)]

    G = torch.empty((m, m), device=device, dtype=flat_hat[0].dtype)
    for i in range(m):
        for j in range(i, m):
            val = torch.dot(flat_hat[i], flat_hat[j])
            G[i, j] = val
            G[j, i] = val
    G = G + reg_eps * torch.eye(m, device=device, dtype=G.dtype)  # ensure PD

    # ---- 3) preferences on (1,2) only ----
    if pref_vector_12 is None:
        alpha = torch.full((m,), 1.0 / m, device=device, dtype=G.dtype)
    else:
        assert pref_vector_12.shape == (m,)
        alpha = pref_vector_12.to(device=device, dtype=G.dtype).clamp_min(0)
        alpha = alpha / alpha.sum().clamp_min(1e-12)

    # ---- 4) solve QP for i=0,1 then combine -> w_bar12 ----
    w_bar12 = torch.zeros((m,), device=device, dtype=G.dtype)
    for i in range(m):
        u = torch.zeros((m,), device=device, dtype=G.dtype)
        u[i] = 1.0  # e_i
        w_i = _solve_min_vGv_lower_bound(G, u, iters=qp_iters)
        w_bar12 += alpha[i] * w_i

    # ---- 5) UPGrad aggregated gradient from g1,g2 only: J^T w_bar12 ----
    grad_up12 = (g1 * w_bar12[0]) + (g2 * w_bar12[1])

    # ---- 6) add g3 (not participating) ----
    grad_final = grad_up12 + (weight_g3 * g3)

    if verbose:
        # contribution ranges
        c1 = g1 * w_bar12[0]
        c2 = g2 * w_bar12[1]
        print(f"UP(1) contrib: min={c1.min().item():.4e}, max={c1.max().item():.4e}, w={w_bar12[0].item():.4f}")
        print(f"UP(2) contrib: min={c2.min().item():.4e}, max={c2.max().item():.4e}, w={w_bar12[1].item():.4f}")
        c3 = weight_g3 * g3
        print(f"g3  contrib:   min={c3.min().item():.4e}, max={c3.max().item():.4e}, weight_g3={weight_g3}")

    return grad_final, grad_up12, g3, w_bar12

def upgrad_n_plus_final(
    target_list,
    x: torch.Tensor,
    pref_vector_n: torch.Tensor | None = None,
    weight_final: float = 1.0,
    norm_eps: float = 1e-4,
    reg_eps: float = 1e-4,
    qp_iters: int = 50,
    n: int = 2,  # Number of gradients to aggregate
    verbose: bool = False,
):
    """
    Use UPGrad on gradients of target_list[0] to target_list[n-1],
    then add gradient of target_list[n] (not participating) to form final gradient.

    Returns:
        grad_final:  g_up(1,...,n-1) + weight_final * gn
        grad_upn:    UPGrad aggregation result from gradients 0 to n-1
        gn:          gradient of target[n]
        w_barn:      weights used in J^T w_bar for UPGrad part (length n)
    """
    assert len(target_list) >= n + 1, f"Need at least {n+1} targets: [t1, t2, ..., tn, ...]"

    device = x.device
    dtype = x.dtype

    # ---- 1) compute g1, g2, ..., gn w.r.t x ----
    grads = [torch.autograd.grad(target_list[i], x, retain_graph=True, create_graph=False)[0] for i in range(n+1)]
    
    # ---- 2) Gram matrix from normalized gradients (ONLY for g1,...,gn) ----
    flat = [g.reshape(-1) for g in grads[:n]]  # Take first n gradients for aggregation
    norms = torch.stack([f.norm() for f in flat]).to(device=device, dtype=torch.float32)  # [n]
    flat_hat = [(flat[i] / (norms[i].to(dtype=flat[i].dtype) + norm_eps)) for i in range(n)]

    G = torch.empty((n, n), device=device, dtype=flat_hat[0].dtype)
    for i in range(n):
        for j in range(i, n):
            val = torch.dot(flat_hat[i], flat_hat[j])
            G[i, j] = val
            G[j, i] = val
    G = G + reg_eps * torch.eye(n, device=device, dtype=G.dtype)  # ensure PD

    # ---- 3) preferences on (1,...,n) only ----
    if pref_vector_n is None:
        alpha = torch.full((n,), 1.0 / n, device=device, dtype=G.dtype)
    else:
        assert pref_vector_n.shape == (n,)
        alpha = pref_vector_n.to(device=device, dtype=G.dtype).clamp_min(0)
        alpha = alpha / alpha.sum().clamp_min(1e-12)

    # ---- 4) solve QP for i=0,1,...,n-1 then combine -> w_barn ----
    w_barn = torch.zeros((n,), device=device, dtype=G.dtype)
    for i in range(n):
        u = torch.zeros((n,), device=device, dtype=G.dtype)
        u[i] = 1.0  # e_i
        w_i = _solve_min_vGv_lower_bound(G, u, iters=qp_iters)
        w_barn += alpha[i] * w_i

    # ---- 5) UPGrad aggregated gradient from g1,...,gn-1: J^T w_barn ----
    grad_upn = sum(grads[i] * w_barn[i] for i in range(n))

    # ---- 6) add gn (not participating) ----
    grad_final = grad_upn + (weight_final * grads[n])

    if verbose:
        # contribution ranges
        for i in range(n):
            c = grads[i] * w_barn[i]
            print(f"UP({i+1}) contrib: min={c.min().item():.4e}, max={c.max().item():.4e}, w={w_barn[i].item():.4f}")
        c_final = weight_final * grads[n]
        print(f"gn contrib:   min={c_final.min().item():.4e}, max={c_final.max().item():.4e}, weight_final={weight_final}")

    return grad_final, grad_upn, grads[n], w_barn

def upgrad_n(
    target_list,
    x: torch.Tensor,
    pref_vector_n: torch.Tensor | None = None,
    weight_final: float = 1.0,
    norm_eps: float = 1e-4,
    reg_eps: float = 1e-4,
    qp_iters: int = 50,
    n: int = 2,  # Number of gradients to aggregate
    verbose: bool = False,
):
    """
    Use UPGrad on gradients of target_list[0] to target_list[n],
    and aggregate them all (including target[n]) into the final gradient.

    Returns:
        grad_final:  Aggregated gradient from all targets (t1 to tn)
        grad_upn:    UPGrad aggregation result from gradients 0 to n
        gn:          gradient of target[n]
        w_barn:      weights used in J^T w_bar for UPGrad part (length n+1)
    """
    assert len(target_list) >= n, f"Need at least {n+1} targets: [t1, t2, ..., tn, ...]"

    device = x.device
    dtype = x.dtype

    # ---- 1) compute g1, g2, ..., gn w.r.t x ----
    grads = [torch.autograd.grad(target_list[i], x, retain_graph=True, create_graph=False)[0] for i in range(n)]
    grads = [torch.clamp(grad, min=-0.03, max=0.03) for grad in grads]

    # ---- 2) Gram matrix from normalized gradients (all gradients) ----
    flat = [g.reshape(-1) for g in grads]  # Take all gradients for aggregation
    norms = torch.stack([f.norm() for f in flat]).to(device=device, dtype=torch.float32)  # [n+1]
    flat_hat = [(flat[i] / (norms[i].to(dtype=flat[i].dtype) + norm_eps)) for i in range(n)]

    G = torch.empty((n, n), device=device, dtype=flat_hat[0].dtype)
    for i in range(n):
        for j in range(i, n):
            val = torch.dot(flat_hat[i], flat_hat[j])
            G[i, j] = val
            G[j, i] = val
    G = G + reg_eps * torch.eye(n, device=device, dtype=G.dtype)  # ensure PD

    # ---- 3) preferences on (1,...,n) only ----
    if pref_vector_n is None:
        alpha = torch.full((n,), 1.0 / (n), device=device, dtype=G.dtype)
    else:
        assert pref_vector_n.shape == (n,)
        alpha = pref_vector_n.to(device=device, dtype=G.dtype).clamp_min(0)

    # ---- 4) solve QP for i=0,1,...,n then combine -> w_barn ----
    w_barn = torch.zeros((n,), device=device, dtype=G.dtype)
    for i in range(n):
        u = torch.zeros((n,), device=device, dtype=G.dtype)
        u[i] = 1.0  # e_i
        w_i = _solve_min_vGv_lower_bound(G, u, iters=qp_iters)
        w_barn += alpha[i] * w_i

    # ---- 5) UPGrad aggregated gradient from g1,...,gn: J^T w_barn ----
    grad_upn = sum(grads[i] * w_barn[i] for i in range(n))

    # ---- 6) final aggregated gradient is just the UPGrad result ----
    grad_final = grad_upn

    if verbose:
        # contribution ranges
        for i in range(n):
            c = grads[i] * w_barn[i]
            print(f"UP({i+1}) contrib: min={c.min().item():.4e}, max={c.max().item():.4e}, w={w_barn[i].item():.4f}")

    return grad_final, grad_upn, grads[n-1], w_barn

def deq_x(x):
    return (255 * x + torch.rand_like(x)) / 256.

def ema_model_update(model, model_ema, mu=0.9999):
    """
    Update EMA model parameters.
    Args:
        model: main model
        model_ema: EMA model
        mu: EMA decay rate, default 0.999.
    """
    assert 0 <= mu <= 1
    if mu != 1:  # If mu is 1, EMA parameters stay unchanged.
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data

def ema_params_check(model, model_ema):
    """
    Check whether EMA model and main model parameters are identical.
    """
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        if not torch.eq(param_ema.data, param.data).all():
            return False
    return True

def swish(x):
    return x * torch.sigmoid(x)

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

'''
neeural network building blocks
'''
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class residual_block(nn.Module):
    def __init__(self, filters, downsample=True, rescale=True):
        super(residual_block, self).__init__()
        self.downsample = downsample
        self.rescale = rescale

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        self.conv1 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        self.conv2 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.act = swish

        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
                self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
                self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self,x):
        x_orig = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act(x)
        x_out = x + x_orig
        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out

'''
deep residual neural networks for different datasets
'''
class celeba_resnet(nn.Module):
    def __init__(self, filter_dim=64, self_attn=False, multiscale=False):
        super(celeba_resnet, self).__init__()
        self.act = swish

        self.multiscale_flag = multiscale
        self.self_attn_flag = self_attn
        self.filter_dim = filter_dim

        if self.multiscale_flag:
            self.init_mid_model()
            self.init_small_model()

        self.init_main_model()
    
    def init_main_model(self):
        self.conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.res_3a = residual_block(filters=2*self.filter_dim, downsample=False)
        self.res_3b = residual_block(filters=2*self.filter_dim, rescale=True)

        self.res_4a = residual_block(filters=4*self.filter_dim, downsample=False)
        self.res_4b = residual_block(filters=4*self.filter_dim, rescale=True)

        self.energy_map = nn.Linear(self.filter_dim*8, 46)

        self.self_attn = Self_Attn(2 * self.filter_dim, self.act)

    def init_mid_model(self):
        self.mid_conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.mid_res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.mid_res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.mid_res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.mid_res_3a = residual_block(filters=2*self.filter_dim, downsample=False)
        self.mid_res_3b = residual_block(filters=2*self.filter_dim, rescale=True)

        self.mid_energy_map = nn.Linear(self.filter_dim*4, 46)

    def init_small_model(self):
        self.small_conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.small_res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.small_res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.small_res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.small_energy_map = nn.Linear(self.filter_dim*2, 46)

    def main_model(self, x):
        x = self.act(self.conv1(x))

        x = self.res_1a(x)
        x = self.res_1b(x)

        x = self.res_2a(x)
        x = self.res_2b(x)

        if self.self_attn_flag:
            x, _ = self.self_attn(x)

        x = self.res_3a(x)
        x = self.res_3b(x)

        x = self.res_4a(x)
        x = self.res_4b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)
        return energy

    def mid_model(self, x):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x)
        x = self.mid_res_1b(x)

        x = self.mid_res_2a(x)
        x = self.mid_res_2b(x)

        x = self.mid_res_3a(x)
        x = self.mid_res_3b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)

        energy = self.mid_energy_map(x)
        return energy
    
    def small_model(self, x):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x)
        x = self.small_res_1b(x)

        x = self.small_res_2a(x)
        x = self.small_res_2b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)

        energy = self.small_energy_map(x)
        return energy

    def forward(self,x):
        energy = self.main_model(x)
        if self.multiscale_flag:
            large_energy = energy
            mid_energy = self.mid_model(x)
            small_energy = self.small_model(x)
            energy = torch.stack([large_energy, mid_energy, small_energy]).mean(dim=0)
        return energy

class celeba_resnet_single_head(nn.Module):
    def __init__(self, filter_dim=64, self_attn=False, multiscale=False):
        super(celeba_resnet_single_head, self).__init__()
        self.act = swish

        self.multiscale_flag = multiscale
        self.self_attn_flag = self_attn
        self.filter_dim = filter_dim

        if self.multiscale_flag:
            self.init_mid_model()
            self.init_small_model()

        self.init_main_model()
    
    def init_main_model(self):
        self.conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.res_3a = residual_block(filters=2*self.filter_dim, downsample=False)
        self.res_3b = residual_block(filters=2*self.filter_dim, rescale=True)

        self.res_4a = residual_block(filters=4*self.filter_dim, downsample=False)
        self.res_4b = residual_block(filters=4*self.filter_dim, rescale=True)

        self.energy_map = nn.Linear(self.filter_dim*8, 1)

        self.self_attn = Self_Attn(2 * self.filter_dim, self.act)

    def init_mid_model(self):
        self.mid_conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.mid_res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.mid_res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.mid_res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.mid_res_3a = residual_block(filters=2*self.filter_dim, downsample=False)
        self.mid_res_3b = residual_block(filters=2*self.filter_dim, rescale=True)

        self.mid_energy_map = nn.Linear(self.filter_dim*4, 1)

    def init_small_model(self):
        self.small_conv1 = nn.Conv2d(3, self.filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = residual_block(filters=self.filter_dim, downsample=False)
        self.small_res_1b = residual_block(filters=self.filter_dim, rescale=False)

        self.small_res_2a = residual_block(filters=self.filter_dim, downsample=False)
        self.small_res_2b = residual_block(filters=self.filter_dim, rescale=True)

        self.small_energy_map = nn.Linear(self.filter_dim*2, 1)

    def main_model(self, x):
        x = self.act(self.conv1(x))

        x = self.res_1a(x)
        x = self.res_1b(x)

        x = self.res_2a(x)
        x = self.res_2b(x)

        if self.self_attn_flag:
            x, _ = self.self_attn(x)

        x = self.res_3a(x)
        x = self.res_3b(x)

        x = self.res_4a(x)
        x = self.res_4b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)
        return energy

    def mid_model(self, x):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x)
        x = self.mid_res_1b(x)

        x = self.mid_res_2a(x)
        x = self.mid_res_2b(x)

        x = self.mid_res_3a(x)
        x = self.mid_res_3b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)

        energy = self.mid_energy_map(x)
        return energy
    
    def small_model(self, x):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x)
        x = self.small_res_1b(x)

        x = self.small_res_2a(x)
        x = self.small_res_2b(x)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)

        energy = self.small_energy_map(x)
        return energy

    def forward(self,x):

        energy = self.main_model(x)

        if self.multiscale_flag:
            large_energy = energy
            mid_energy = self.mid_model(x)
            small_energy = self.small_model(x)
            energy = torch.stack([large_energy, mid_energy, small_energy]).mean(dim=0)
        return energy

'''
single head classifier trainer for different datasets
'''
class single_head_classifier_lightning_module(pl.LightningModule):
    
    def __init__(self,args):
        super(single_head_classifier_lightning_module, self).__init__()
        self.args = args

        self.model = celeba_resnet_single_head()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.beta1, self.args.beta2))

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_y = batch_y.view(-1, 23)
        conditional_index = self.args.conditional_index
        pos_weight = torch.tensor(self.args.posweight, device=self.device)

        # Forward pass
        logits = self.model(batch_x)
        
        # Extract labels at the specified index (binary label).
        labels = batch_y[:, conditional_index].float()
        
        # Compute binary cross-entropy loss.
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits.squeeze(), labels)
        
        # Log training metrics.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute and log accuracy.
        predictions = (torch.sigmoid(logits) > 0.5).float().squeeze()
        accuracy = (predictions == labels).float().mean()
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_y = batch_y.view(-1, 23)
        conditional_index = self.args.conditional_index
        
        logits = self.model(batch_x)
        
        labels = batch_y[:, conditional_index].float()
        
        if hasattr(self.args, 'pos_weight'):
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        val_loss = loss_fn(logits.squeeze(), labels)
        
        predictions = (torch.sigmoid(logits) > 0.5).float().squeeze()
        
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        from torchmetrics.classification import (
            BinaryF1Score, 
            BinaryPrecision, 
            BinaryRecall,
            BinaryConfusionMatrix
        )
        
        f1 = BinaryF1Score().to(self.device)(predictions, labels)
        self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        
        precision = BinaryPrecision().to(self.device)(predictions, labels)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        
        recall = BinaryRecall().to(self.device)(predictions, labels)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        
        accuracy = (predictions == labels).float().mean()
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)
        
        return {
            'val_loss': val_loss,
            'labels': labels
        }
'''
classifier trainer for different datasets
'''
class classifier_lightning_module(pl.LightningModule):
    def __init__(self, args):
        super(classifier_lightning_module, self).__init__()
        self.args = args
        self.model = celeba_resnet()
        self.loss_fn = nn.CrossEntropyLoss()

        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.beta1, self.args.beta2)
        )

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        logits = self.model(batch_x)     # [B, 46]
        logits = logits.view(-1, 2)      # [B*23, 2]
        batch_y = batch_y.view(-1,)      # [B*23]

        loss = self.loss_fn(logits, batch_y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_eval = self.model_ema
        else:
            model_for_eval = self.model

        logits = model_for_eval(batch_x)      # [B, 46]
        logits = logits.view(-1, 23, 2)       # [B, 23, 2]
        batch_y = batch_y.view(-1, 23)        # [B, 23]

        aurocs = []

        for i in range(23):
            y_true = batch_y[:, i].detach().cpu().numpy()

            y_prob = F.softmax(logits[:, i, :], dim=-1)[:, 1]
            y_prob = y_prob.detach().cpu().numpy()

            if len(set(y_true)) < 2:
                continue

            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)

            self.log(
                f'val_auroc_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}',
                auroc,
                sync_dist=True,
                prog_bar=False
            )

        if len(aurocs) > 0:
            mean_auroc = sum(aurocs) / len(aurocs)
        else:
            mean_auroc = -1

        self.log("val_mean_auroc", mean_auroc, sync_dist=True, prog_bar=True)

'''
pareto classifier trainer for different datasets
'''
class pareto_classifier_lightning_module(pl.LightningModule):
    def __init__(self,args):
        super(pareto_classifier_lightning_module, self).__init__()
        self.args = args
        self.model = celeba_resnet()
        self.loss_fn = nn.CrossEntropyLoss()

        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

        self.automatic_optimization = False

    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.beta1, self.args.beta2))

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        logits = self.model(batch_x)             # [B, 23*2]
        logits = logits.view(-1, 23, 2)          # [B, 23, 2]
        batch_y = batch_y.view(-1, 23)           # [B, 23]

        loss_list = []
        for i in range(23):
            loss_i = self.loss_fn(logits[:, i, :], batch_y[:, i])
            loss_list.append(loss_i)

        opt = self.optimizers()
        opt.zero_grad()

        pref_vector = torch.ones(len(loss_list), device=loss_list[0].device)
        aggregator = UPGrad(pref_vector=pref_vector)
        backward(
            tensors=loss_list,
            aggregator=aggregator,
            retain_graph=False,
            parallel_chunk_size=1
        )

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= dist.get_world_size()

        if self.args.clip_gradients:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.args.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        opt.step()

        for i in range(23):
            self.log(
                f"train_classification_loss_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}",
                loss_list[i],
                sync_dist=True,
                on_step=True,
                on_epoch=True
            )

        mean_loss = sum(loss_list) / len(loss_list)
        self.log("train_mean_loss", mean_loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return mean_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_eval = self.model_ema
        else:
            model_for_eval = self.model

        logits = model_for_eval(batch_x)      # [B, 46]
        logits = logits.view(-1, 23, 2)       # [B, 23, 2]
        batch_y = batch_y.view(-1, 23)        # [B, 23]

        aurocs = []

        for i in range(23):
            y_true = batch_y[:, i].detach().cpu().numpy()

            y_prob = F.softmax(logits[:, i, :], dim=-1)[:, 1]
            y_prob = y_prob.detach().cpu().numpy()

            if len(set(y_true)) < 2:
                continue

            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)

            self.log(
                f'val_auroc_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}',
                auroc,
                sync_dist=True,
                prog_bar=False
            )

        if len(aurocs) > 0:
            mean_auroc = sum(aurocs) / len(aurocs)
        else:
            mean_auroc = -1

        self.log("val_mean_auroc", mean_auroc, sync_dist=True, prog_bar=True)

'''
energy based model trainer for different datasets
'''
class ebm_lightning_module(pl.LightningModule):
    
    def __init__(self,args):
        super(ebm_lightning_module, self).__init__()
        self.args = args

        self.model = celeba_resnet_single_head()

        self.sampler = ReplayBuffer(
            max_size=args.buffer_size,
            replace_prob=args.replace_prob,
            img_shape=(3,args.img_size,args.img_size),
            device=self.device
        )

        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.beta1, self.args.beta2))

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        loss_list = []

        logits = self.model(batch_x)

        batch_x_neg = self.sampler.sample(batch_x.shape[0], train=True).to(batch_x.device)
        batch_x_neg = self.sample(
            batch_x_neg,
            retain_graph=True,
            num_steps=self.args.sgld_steps,
            learning_rate=self.args.sgld_lr,
            sigma=self.args.sgld_sigma,
        ).float().to(batch_x.device)

        batch_x_neg_cd = batch_x_neg.detach()

        self.sampler.update_buffer(batch_x_neg.detach().to("cpu"))

        neg_logits = self.model(batch_x_neg_cd)

        pos_energy = logits
        neg_energy = neg_logits

        cd_loss = pos_energy.mean() - neg_energy.mean()

        if self.args.kl_loss:
            batch_buffer = self.sampler.sample(100, train=True).to(batch_x.device)
            batch_buffer = deq_x(batch_buffer)

            kl_loss_1 = self.model(batch_x_neg).mean()

            dist_matrix = torch.norm(
                batch_x_neg.view(batch_x_neg.shape[0], -1)[None, :, :]
                - batch_buffer.view(batch_buffer.shape[0], -1)[:, None, :],
                p=2,
                dim=-1,
            )
            mins = dist_matrix.min(dim=1).values
            min_fin = mins > 0
            kl_loss_2 = - torch.log(mins[min_fin]).mean()

            loss_list.append(torch.pow(pos_energy,2).mean() + torch.pow(neg_energy,2).mean() + cd_loss + 1 * kl_loss_1 + 1 * 0.3 * kl_loss_2)
        else:
            loss_list.append(torch.pow(pos_energy,2).mean() + torch.pow(neg_energy,2).mean() + cd_loss)

        if self.args.kl_loss:
            self.log("train_kl_loss_1", kl_loss_1, sync_dist=True)
            self.log("train_kl_loss_2", kl_loss_2, sync_dist=True)
        self.log("train_CD_loss", cd_loss, sync_dist=True)
        self.log("train_pos_energy", pos_energy.mean(), sync_dist=True)
        self.log("train_neg_energy", neg_energy.mean(), sync_dist=True)

        return sum(loss_list)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_eval = self.model_ema
        else:
            model_for_eval = self.model

        logits = model_for_eval(batch_x)

    def on_validation_epoch_end(self):
        
        batch_x = torch.rand(32,3,self.args.img_size,self.args.img_size).to(self.device)

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_sampling = self.model_ema
        else:
            model_for_sampling = self.model

        with torch.enable_grad():
            sample = self.sample(
                batch_x,
                num_steps=self.args.sgld_steps,
                learning_rate=self.args.sgld_lr,
                sigma=self.args.sgld_sigma,
                model=model_for_sampling,
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(
                    self.args.save_dir,
                    f"unconditional_sampling_from_noise_epoch_{self.current_epoch}.png"
                )
            )
        
        with torch.enable_grad():
            samples_from_buffer = self.sampler.sample(batch_x.shape[0]).to(batch_x.device)
            sample = self.sample(
                samples_from_buffer,
                num_steps=self.args.sgld_steps,
                learning_rate=self.args.sgld_lr,
                sigma=self.args.sgld_sigma,
                model=model_for_sampling,
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(
                    self.args.save_dir,
                    f"unconditional_sampling_from_buffer_epoch_{self.current_epoch}.png"
                )
            )

    def sample(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy = model(initial_samples).sum()

            # compute grad w.r.t. samples
            if step == num_steps - 1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]
            # Langevin update (do not use .data)
            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()
             
    def sample(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False,model=None):

        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy = model(initial_samples).sum()

            # compute grad w.r.t. samples
            grad = torch.autograd.grad(energy, initial_samples, create_graph=retain_graph)[0]

            # Langevin update (do not use .data)
            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def sample_from_noise(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        warmup_rounds = 10
        warmup_inner_steps = 60

        im_size = initial_samples.shape[-1]  # (N,3,H,W)
        color_transform = get_color_distortion()
        transform = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform
        ])

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for i in range(warmup_rounds):
            for i in range(warmup_inner_steps):

                noise = torch.randn_like(initial_samples)

                energy = model(initial_samples).sum()

                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]

                move = learning_rate * grad
                move = torch.clamp(move, min=-0.03, max=0.03)

                initial_samples = initial_samples - move + sigma * noise
                initial_samples = initial_samples.clamp(0, 1).detach()  
                initial_samples = initial_samples.detach()

                initial_samples = initial_samples.detach().cpu().numpy().transpose((0, 2, 3, 1))
                initial_samples = (initial_samples * 255).astype(np.uint8)
                
                ims = []
                for i in range(initial_samples.shape[0]):
                    im_i = np.array(transform(Image.fromarray(np.array(initial_samples[i]))))
                    ims.append(im_i)

                initial_samples = torch.Tensor(np.array(ims)).cuda().squeeze()
                initial_samples.requires_grad_(requires_grad=True)

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            energy = model(initial_samples).sum()

            if step == num_steps - 1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]
            # Langevin update (do not use .data)
            move = learning_rate * grad
            move = torch.clamp(move, min=-0.03, max=0.03)

            initial_samples = initial_samples - move + sigma * noise
            #initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

'''
joint energy based model trainer for different datasets
'''
class jem_lightning_module(pl.LightningModule):
    
    def __init__(self,args):
        super(jem_lightning_module, self).__init__()
        self.args = args
        self.model = celeba_resnet()

        self.sampler = ReplayBuffer(
            max_size=args.buffer_size,
            replace_prob=args.replace_prob,
            img_shape=(3,args.img_size,args.img_size),
            device=self.device
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.beta1, self.args.beta2)
        )

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        loss_list = []

        logits = self.model(batch_x)
        logits = logits.view(-1, 23, 2)
        batch_y = batch_y.view(-1, 23)

        for i in range(23):
            if self.args.label_smoothing:
                loss_list.append(
                    nn.CrossEntropyLoss(label_smoothing=random.uniform(0.0, 0.5))(
                        logits[:, i, :], batch_y[:, i]
                    )
                )
            else:
                loss_list.append(
                    nn.CrossEntropyLoss()(logits[:, i, :], batch_y[:, i])
                )

        batch_x_neg = self.sampler.sample(batch_x.shape[0], train=True).to(batch_x.device)
        batch_x_neg = self.sample(
            batch_x_neg,
            retain_graph=True,
            num_steps=self.args.sgld_steps,
            learning_rate=self.args.sgld_lr,
            sigma=self.args.sgld_sigma,
        ).float().to(batch_x.device)

        batch_x_neg_cd = batch_x_neg.detach()

        self.sampler.update_buffer(batch_x_neg.detach().to("cpu"))

        neg_logits = self.model(batch_x_neg_cd).view(-1, 23, 2)

        pos_energy = -torch.logsumexp(logits, dim=-1).sum(dim=-1).mean()
        neg_energy = -torch.logsumexp(neg_logits, dim=-1).sum(dim=-1).mean()

        cd_loss = pos_energy - neg_energy

        if self.args.kl_loss:
            batch_buffer = self.sampler.sample(100, train=True).to(batch_x.device)
            batch_buffer = deq_x(batch_buffer)

            kl_loss_1 = - torch.logsumexp(
                self.model(batch_x_neg).view(-1, 23, 2), dim=-1
            ).sum(dim=-1).mean()

            dist_matrix = torch.norm(
                batch_x_neg.view(batch_x_neg.shape[0], -1)[None, :, :]
                - batch_buffer.view(batch_buffer.shape[0], -1)[:, None, :],
                p=2,
                dim=-1,
            )
            mins = dist_matrix.min(dim=1).values
            min_fin = mins > 0
            kl_loss_2 = - torch.log(mins[min_fin]).mean()

            loss_list.append(self.args.cd_weight * cd_loss + 1 * kl_loss_1 + 1 * 0.3 * kl_loss_2)
        else:
            loss_list.append(self.args.cd_weight * cd_loss)

        for i in range(23):
            self.log(
                f"train_classification_loss_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}",
                loss_list[i],
                sync_dist=True,
            )
        if self.args.kl_loss:
            self.log("train_kl_loss_1", kl_loss_1, sync_dist=True)
            self.log("train_kl_loss_2", kl_loss_2, sync_dist=True)
        self.log("train_CD_loss", cd_loss, sync_dist=True)
        self.log("train_pos_energy", pos_energy, sync_dist=True)
        self.log("train_neg_energy", neg_energy, sync_dist=True)

        return sum(loss_list)
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()
        
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        logits = self.model(batch_x)  # [B, 23*2]
        logits = logits.view(-1, 23, 2)  
        batch_y = batch_y.view(-1, 23)

        aurocs = []

        for i in range(23):
            y_true = batch_y[:, i].detach().cpu().numpy()
            
            y_prob = F.softmax(logits[:, i, :], dim=-1)[:, 1]
            y_prob = y_prob.detach().cpu().numpy()

            if len(set(y_true)) < 2:
                continue

            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)

            self.log(
                f'val_auroc_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}',
                auroc,
                sync_dist=True,
                prog_bar=False
            )

        if len(aurocs) > 0:
            mean_auroc = sum(aurocs) / len(aurocs)
            self.log("val_mean_auroc", mean_auroc, sync_dist=True, prog_bar=True)
        else:
            self.log("val_mean_auroc", -1, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self):
        
        batch_x = torch.rand(32,3,self.args.img_size,self.args.img_size).to(self.device)

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_sampling = self.model_ema
        else:
            model_for_sampling = self.model

        with torch.enable_grad():
            sample = self.sample(
                batch_x, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(
                    self.args.save_dir,
                    f"unconditional_sampling_from_noise_epoch_{self.current_epoch}.png"
                )
            )

        with torch.enable_grad():
            samples_from_buffer = self.sampler.sample(batch_x.shape[0]).to(batch_x.device)
            sample = self.sample(
                samples_from_buffer, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(self.args.save_dir,
                             f"unconditional_sampling_from_buffer_epoch_{self.current_epoch}.png")
            )
    
    def sample(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False,model=None):

        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy = - torch.logsumexp(model(initial_samples).view(-1,23,2),dim=-1).sum(dim=-1).sum()

            # compute grad w.r.t. samples

            if step == num_steps - 1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]

            # Langevin update (do not use .data)
            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def sample_from_noise(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        warmup_rounds = 10
        warmup_inner_steps = 60

        im_size = initial_samples.shape[-1]  # (N,3,H,W)
        color_transform = get_color_distortion()
        transform = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform
        ])

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for i in range(warmup_rounds):
            for i in range(warmup_inner_steps):

                noise = torch.randn_like(initial_samples)

                energy = - torch.logsumexp(model(initial_samples).view(-1,23,2),dim=-1).sum(dim=-1).sum()

                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]

                move = learning_rate * grad
                move = torch.clamp(move, min=-0.03, max=0.03)

                initial_samples = initial_samples - move + sigma * noise
                initial_samples = initial_samples.clamp(0, 1).detach()  
                initial_samples = initial_samples.detach()

                initial_samples = initial_samples.detach().cpu().numpy().transpose((0, 2, 3, 1))
                initial_samples = (initial_samples * 255).astype(np.uint8)
                
                ims = []
                for i in range(initial_samples.shape[0]):
                    im_i = np.array(transform(Image.fromarray(np.array(initial_samples[i]))))
                    ims.append(im_i)

                initial_samples = torch.Tensor(np.array(ims)).cuda().squeeze()
                initial_samples.requires_grad_(requires_grad=True)

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy = - torch.logsumexp(model(initial_samples).view(-1,23,2),dim=-1).sum(dim=-1).sum()

            # compute grad w.r.t. samples
            if step == num_steps-1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]
            # Langevin update (do not use .data)

            move = learning_rate * grad
            move = torch.clamp(move, min=-0.03, max=0.03)

            initial_samples = initial_samples - move + sigma * noise
            #initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def conditional_sample(self, conditional_index, conditional_label, initial_samples,
                           num_steps, learning_rate, sigma, retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        unconditional_index = [i for i in range(23) if i!=conditional_index]
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[:, conditional_index, 0 if conditional_index else 1]
            unconditional_logits = logits[:, unconditional_index,:]
            energy = - conditional_logits.sum() - torch.logsumexp(
                unconditional_logits, dim=-1
            ).sum(dim=-1).sum()

            grad = torch.autograd.grad(
                energy, initial_samples, create_graph=retain_graph
            )[0]

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, 
                                num_steps, learning_rate, sigma,retain_graph=False, return_samples_each_step=False,model=None):

        if model is None:
            model = self.model

        unconditional_index_list = [i for i in range(23) if i not in conditional_index_list]
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[:, conditional_index_list,
                                        [0 if conditional_index else 1 for conditional_index in conditional_label_list]]
            unconditional_logits = logits[:,unconditional_index_list,:]
            energy = - conditional_logits.sum() - torch.logsumexp(
                unconditional_logits, dim=-1
            ).sum(dim=-1).sum()

            grad = torch.autograd.grad(
                energy, initial_samples, create_graph=retain_graph
            )[0]

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list,
                                        initial_samples, num_steps, learning_rate, sigma, retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[0, conditional_index_list,:].squeeze(0)
            target_list = [
                torch.log(torch.softmax(logit,dim=-1).squeeze()[0])
                if conditional_label_list[idx]
                else torch.log(torch.softmax(logit,dim=-1).squeeze()[1])
                for idx,logit in enumerate(conditional_logits)
            ]
            target_list.append(- torch.logsumexp(logits,dim=-1).sum(dim=-1).mean())

            pref_vector = torch.ones(len(target_list))
            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors = target_list,
                aggregator = aggregator,
                retain_graph = True,
                parallel_chunk_size = 1
            )
            grad = initial_samples.grad.data

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    # argmax (log p(x),log p(y1|x),log p(y2|x)) learning rate 0.1
    def pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False,model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # log p(y|x)
            target_list = [
                torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 0]).sum()
                if conditional_label_list[idx]
                else torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 1]).sum()
                for idx in range(num_attributes)
            ]
            # log p(x)
            target_list.append(torch.logsumexp(logits.view(-1, 46), dim=-1).sum())

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1

            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0

            # pref_vector size = len(target_list)
            pref_vector = torch.ones(len(target_list), device=initial_samples.device)

            pref_vector[0] = w_cond  
            pref_vector[1] = w_cond  
            pref_vector[2] = w_gen 
            # ----------------------------------------------------------

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            # compute grad w.r.t. samples
            grad = initial_samples.grad

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate pareto
    def latent_jem_pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # E(Y|X)

            target_list = [
                conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum()
                if conditional_label_list[idx]
                else conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum()
                for idx in range(num_attributes)
            ]
            # E(X)
            target_list.append(torch.logsumexp(logits,dim=-1).sum(dim=-1).sum())

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1

            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0

            # pref_vector size = len(target_list)
            pref_vector = torch.ones(len(target_list), device=initial_samples.device)

            pref_vector[0] = w_cond   
            pref_vector[1] = w_cond  
            pref_vector[2] = w_gen

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            # compute grad w.r.t. samples
            grad = initial_samples.grad

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate normal
    def latent_jem_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1

            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0

            # E(Y|X)
            target_list = [
                w_cond * (conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                if conditional_label_list[idx]
                else w_cond * (conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                for idx in range(num_attributes)
            ]
            # E(X)
            target_list.append(w_gen * torch.logsumexp(logits,dim=-1).sum(dim=-1).sum())
    
            grad = torch.autograd.grad(sum(target_list), initial_samples, create_graph=retain_graph)[0]

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

'''
pareto energy based model trainer for different datasets
'''
class pareto_jem_lightning_module(pl.LightningModule):
    
    def __init__(self,args):
        super(pareto_jem_lightning_module, self).__init__()
        self.args = args
        self.model = celeba_resnet()
        self.sampler = ReplayBuffer(max_size=args.buffer_size, replace_prob=args.replace_prob, img_shape=(3,args.img_size,args.img_size), device=self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.automatic_optimization = False
    
        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None
    
    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):  
        return torch.optim.Adam(self.model.parameters(), 
                                lr=self.args.lr, 
                                betas=(self.args.beta1, self.args.beta2))

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        loss_list = []

        logits = self.model(batch_x)
        logits = logits.view(-1, 23, 2)
        batch_y = batch_y.view(-1, 23)

        for i in range(23):
            if self.args.label_smoothing:
                loss_list.append(
                    nn.CrossEntropyLoss(label_smoothing=random.uniform(0.0, 0.5))(
                        logits[:, i, :], batch_y[:, i]
                    )
                )
            else:
                loss_list.append(
                    nn.CrossEntropyLoss()(logits[:, i, :], batch_y[:, i])
                )

        batch_x_neg = self.sampler.sample(batch_x.shape[0], train=True).to(batch_x.device)
        batch_x_neg = self.sample(
            batch_x_neg,
            retain_graph=True,
            num_steps=self.args.sgld_steps,
            learning_rate=self.args.sgld_lr,
            sigma=self.args.sgld_sigma,
        ).float().to(batch_x.device)

        batch_x_neg_cd = batch_x_neg.detach()

        self.sampler.update_buffer(batch_x_neg.detach().to("cpu"))

        neg_logits = self.model(batch_x_neg_cd).view(-1, 23, 2)

        pos_energy = -torch.logsumexp(logits, dim=-1).sum(dim=-1).mean()
        neg_energy = -torch.logsumexp(neg_logits, dim=-1).sum(dim=-1).mean()

        cd_loss = pos_energy - neg_energy

        if self.args.kl_loss:
            batch_buffer = self.sampler.sample(100, train=True).to(batch_x.device)
            batch_buffer = deq_x(batch_buffer)

            kl_loss_1 = -torch.logsumexp(
                self.model(batch_x_neg).view(-1, 23, 2), dim=-1
            ).sum(dim=-1).mean()

            dist_matrix = torch.norm(
                batch_x_neg.view(batch_x_neg.shape[0], -1)[None, :, :]
                - batch_buffer.view(batch_buffer.shape[0], -1)[:, None, :],
                p=2,
                dim=-1,
            )
            mins = dist_matrix.min(dim=1).values
            min_fin = mins > 0
            kl_loss_2 = -torch.log(mins[min_fin]).mean()

            loss_list.append(cd_loss + 0.001 * kl_loss_1 + 0.001 * 0.3 * kl_loss_2)
        else:
            loss_list.append(cd_loss)

        opt = self.optimizers()
        opt.zero_grad()
        pref_vector = torch.ones(len(loss_list))
        pref_vector[-1] = 100
        aggregator = UPGrad(pref_vector=pref_vector)
        backward(
            tensors=loss_list,
            aggregator=aggregator,
            retain_graph=True,
            parallel_chunk_size=1,
        )

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= dist.get_world_size()

        if self.args.clip_gradients:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.args.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        opt.step()

        for i in range(23):
            self.log(
                f"train_classification_loss_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}",
                loss_list[i],
                sync_dist=True,
            )
        if self.args.kl_loss:
            self.log("train_kl_loss_1", kl_loss_1, sync_dist=True)
            self.log("train_kl_loss_2", kl_loss_2, sync_dist=True)
        self.log("train_CD_loss", cd_loss, sync_dist=True)
        self.log("train_pos_energy", pos_energy, sync_dist=True)
        self.log("train_neg_energy", neg_energy, sync_dist=True)

        return cd_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()
        
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        logits = self.model(batch_x)  # [B, 23*2]
        logits = logits.view(-1, 23, 2)  
        batch_y = batch_y.view(-1, 23)

        aurocs = []

        for i in range(23):
            y_true = batch_y[:, i].detach().cpu().numpy()
            
            y_prob = F.softmax(logits[:, i, :], dim=-1)[:, 1]
            y_prob = y_prob.detach().cpu().numpy()

            if len(set(y_true)) < 2:
                continue

            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)

            self.log(
                f'val_auroc_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}',
                auroc,
                sync_dist=True,
                prog_bar=False
            )

        if len(aurocs) > 0:
            mean_auroc = sum(aurocs) / len(aurocs)
            self.log("val_mean_auroc", mean_auroc, sync_dist=True, prog_bar=True)
        else:
            self.log("val_mean_auroc", -1, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self):
        
        batch_x = torch.rand(32,3,self.args.img_size,self.args.img_size).to(self.device)

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_sampling = self.model_ema
        else:
            model_for_sampling = self.model

        with torch.enable_grad():
            sample = self.sample(
                batch_x, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(
                    self.args.save_dir,
                    f"unconditional_sampling_from_noise_epoch_{self.current_epoch}.png"
                )
            )

        with torch.enable_grad():
            samples_from_buffer = self.sampler.sample(batch_x.shape[0]).to(batch_x.device)
            sample = self.sample(
                samples_from_buffer, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(self.args.save_dir,
                             f"unconditional_sampling_from_buffer_epoch_{self.current_epoch}.png")
            )

    def sample(self, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        warmup_rounds = 10
        warmup_inner_steps = 60

        im_size = initial_samples.shape[-1]  # (N,3,H,W)
        color_transform = get_color_distortion()
        transform = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform
        ])

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for i in range(warmup_rounds):
            for i in range(warmup_inner_steps):

                noise = torch.randn_like(initial_samples)

                energy = - torch.logsumexp(model(initial_samples).view(-1,23,2),dim=-1).sum(dim=-1).sum()

                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]

                move = learning_rate * grad
                move = torch.clamp(move, min=-0.03, max=0.03)

                initial_samples = initial_samples - move + sigma * noise
                initial_samples = initial_samples.clamp(0, 1).detach()  
                initial_samples = initial_samples.detach()

                initial_samples = initial_samples.detach().cpu().numpy().transpose((0, 2, 3, 1))
                initial_samples = (initial_samples * 255).astype(np.uint8)
                
                ims = []
                for i in range(initial_samples.shape[0]):
                    im_i = np.array(transform(Image.fromarray(np.array(initial_samples[i]))))
                    ims.append(im_i)

                initial_samples = torch.Tensor(np.array(ims)).cuda().squeeze()
                initial_samples.requires_grad_(requires_grad=True)

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)

            # compute energy
            energy = - torch.logsumexp(model(initial_samples).view(-1,23,2),dim=-1).sum(dim=-1).sum()

            # compute grad w.r.t. samples
            if step == num_steps-1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]
            # Langevin update (do not use .data)

            move = learning_rate * grad
            move = torch.clamp(move, min=-0.03, max=0.03)

            initial_samples = initial_samples - move + sigma * noise
            #initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()  # detach old graph
            initial_samples.requires_grad_(True)  # re-enable grad

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def conditional_sample(self, conditional_index, conditional_label, initial_samples,
                           num_steps, learning_rate, sigma, retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        unconditional_index = [i for i in range(23) if i!=conditional_index]
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[:, conditional_index, 0 if conditional_index else 1]
            unconditional_logits = logits[:, unconditional_index,:]
            energy = - conditional_logits.sum() - torch.logsumexp(
                unconditional_logits, dim=-1
            ).sum(dim=-1).sum()

            grad = torch.autograd.grad(
                energy, initial_samples, create_graph=retain_graph
            )[0]

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, 
                                num_steps, learning_rate, sigma,retain_graph=False, return_samples_each_step=False,model=None):

        if model is None:
            model = self.model

        unconditional_index_list = [i for i in range(23) if i not in conditional_index_list]
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[:, conditional_index_list,
                                        [0 if conditional_index else 1 for conditional_index in conditional_label_list]]
            unconditional_logits = logits[:,unconditional_index_list,:]
            energy = - conditional_logits.sum() - torch.logsumexp(
                unconditional_logits, dim=-1
            ).sum(dim=-1).sum()

            grad = torch.autograd.grad(
                energy, initial_samples, create_graph=retain_graph
            )[0]

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list,
                                        initial_samples, num_steps, learning_rate, sigma, retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            logits = model(initial_samples).view(-1,23,2)
            conditional_logits = logits[0, conditional_index_list,:].squeeze(0)
            target_list = [
                torch.log(torch.softmax(logit,dim=-1).squeeze()[0])
                if conditional_label_list[idx]
                else torch.log(torch.softmax(logit,dim=-1).squeeze()[1])
                for idx,logit in enumerate(conditional_logits)
            ]
            target_list.append(- torch.logsumexp(logits,dim=-1).sum(dim=-1).mean())

            pref_vector = torch.ones(len(target_list))
            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors = target_list,
                aggregator = aggregator,
                retain_graph = True,
                parallel_chunk_size = 1
            )
            grad = initial_samples.grad.data

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    # argmax (log p(x),log p(y1|x),log p(y2|x)) learning rate 0.1
    def pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False,model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # log p(y|x)
            target_list = [
                torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 0]).sum()
                if conditional_label_list[idx]
                else torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 1]).sum()
                for idx in range(num_attributes)
            ]
            # log p(x)
            target_list.append(torch.logsumexp(logits.view(-1, 46), dim=-1).sum())

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            w_cond = 1.0 - 0.9 * t

            w_gen = 0.1 + 0.9 * t

            pref_vector = torch.ones(len(target_list), device=initial_samples.device)

            pref_vector[0] = w_cond  
            pref_vector[1] = w_cond  
            pref_vector[2] = w_gen 
            # ----------------------------------------------------------

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            # compute grad w.r.t. samples
            grad = initial_samples.grad

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate pareto
    def latent_jem_pareto_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # E(Y|X)

            target_list = [
                conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum()
                if conditional_label_list[idx]
                else conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum()
                for idx in range(num_attributes)
            ]
            # E(X)
            target_list.append(torch.logsumexp(logits,dim=-1).sum(dim=-1).sum())

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1

            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0

            # pref_vector size = len(target_list)
            pref_vector = torch.ones(len(target_list), device=initial_samples.device)

            pref_vector[0] = w_cond   
            pref_vector[1] = w_cond  
            pref_vector[2] = w_gen

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            # compute grad w.r.t. samples
            grad = initial_samples.grad

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate normal
    def latent_jem_multi_conditional_sample(self, conditional_index_list, conditional_label_list, initial_samples, num_steps, learning_rate, sigma, 
            retain_graph=False, return_samples_each_step=False, model=None):
        
        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)   # Correct usage

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1
            w_cond = 1
            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0
            w_gen = 1
            # E(Y|X)
            target_list = [
                w_cond * (conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                if conditional_label_list[idx]
                else w_cond * (conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                for idx in range(num_attributes)
            ]
            # E(X)
            target_list.append(w_gen * torch.logsumexp(logits,dim=-1).sum(dim=-1).sum())
    
            grad = torch.autograd.grad(sum(target_list), initial_samples, create_graph=retain_graph)[0]

            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

    def two_stage_pareto_multi_conditional_sample(
        self,
        conditional_index_list,
        conditional_label_list,
        initial_samples,
        num_steps,
        learning_rate,
        sigma,
        retain_graph=False,
        return_samples_each_step=False,
        model=None
    ):
        if model is None:
            model = self.model

        # -------------------------
        # Stage 1 config (user-provided hyperparameters)
        # -------------------------
        warmup_rounds = 20
        warmup_inner_steps = 60

        im_size = initial_samples.shape[-1]  # (N,3,H,W)
        color_transform = get_color_distortion()
        # Note: transform here excludes ToTensor; assumes it can handle torch.Tensor.
        transform = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform
        ])

        # -------------------------
        # init
        # -------------------------
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        # -------------------------
        # one step update (logic kept unchanged)
        # -------------------------
        def one_step_update(x, step, total_steps):
            # reset noise
            noise = torch.randn_like(x)
            logits = model(x).view(-1, 23, 2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # log p(y|x)
            target_list = [
                torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 0]).sum()
                if conditional_label_list[idx]
                else torch.log(torch.softmax(conditional_logits[:, idx, :], dim=-1)[:, 1]).sum()
                for idx in range(num_attributes)
            ]
            # log p(x)
            target_list.append(torch.logsumexp(logits.view(-1, 46), dim=-1).sum())

            if total_steps > 1:
                t = step / (total_steps - 1)
            else:
                t = 0.0

            w_cond = 1.0 - 0.9 * t
            w_gen  = 0.1 + 0.9 * t

            pref_vector = torch.ones(len(target_list), device=x.device)
            pref_vector[0] = w_cond
            pref_vector[1] = w_cond
            pref_vector[2] = w_gen

            # ---- Key: clear old grad to avoid accumulation. ----
            if x.grad is not None:
                x.grad.zero_()

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            grad = x.grad

            # Langevin update
            x = x + learning_rate * grad + sigma * noise
            x = x.clamp(0, 1).detach()
            x.requires_grad_(True)
            return x

        # -------------------------
        # Stage 1: warmup (with transform)
        # -------------------------
        total_warmup_steps = warmup_rounds * warmup_inner_steps
        global_step = 0

        for r in range(warmup_rounds):
            for s in range(warmup_inner_steps):
                initial_samples = one_step_update(initial_samples, global_step, total_warmup_steps)
                global_step += 1

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

            # After each warmup round, apply one transform (Stage 1 only).
            # Use CPU tensors for augmentation to avoid torchvision GPU support mismatch.
            x_cpu = initial_samples.detach().cpu()  # (N,3,H,W) float in [0,1]
            ims = []
            for k in range(x_cpu.shape[0]):
                ims.append(transform(x_cpu[k]))  # Expected output remains torch.Tensor (3,H,W).

            initial_samples = torch.stack(ims, dim=0).to(self.device, non_blocking=True)
            initial_samples = initial_samples.clamp(0, 1).detach().squeeze()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        # -------------------------
        # Stage 2: refine (no transform)
        # -------------------------
        for step in range(num_steps):
            initial_samples = one_step_update(initial_samples, step, num_steps)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        # -------------------------
        # return (unchanged)
        # -------------------------
        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate pareto
    def two_stage_latent_jem_pareto_multi_conditional_sample(
        self,
        conditional_index_list,
        conditional_label_list,
        initial_samples,
        num_steps,
        learning_rate,
        sigma,
        retain_graph=False,
        return_samples_each_step=False,
        model=None
    ):
        if model is None:
            model = self.model

        # ---------------------------
        # Stage-1 transform (PIL pipeline, safe with ToTensor)
        # ---------------------------
        im_size = initial_samples.shape[-1]  # assume (N,3,H,W)
        color_transform = get_color_distortion()
        transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(),color_transform])

        # warmup hyperparameters: fixed 10x20 (function signature unchanged)
        warmup_rounds = 20
        warmup_inner_steps = 30

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        def one_step_update(x, step, total_steps, final=False):
            # reset noise
            noise = torch.randn_like(x)
            logits = model(x).view(-1, 23, 2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            # E(Y|X)
            target_list = [
                (conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                if conditional_label_list[idx]
                else
                (conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                for idx in range(num_attributes)
            ]
            # E(X)
            if final:
                target_list.append(torch.logsumexp(logits, dim=-1).sum(dim=-1).sum())

            # pref_vector size = len(target_list)
            pref_vector = torch.ones(len(target_list), device=x.device)

            # ---- Key: clear old grad to avoid accumulation. ----
            if x.grad is not None:
                x.grad.zero_()

            aggregator = UPGrad(pref_vector=pref_vector)
            backward(
                tensors=target_list,
                aggregator=aggregator,
                retain_graph=True,
                parallel_chunk_size=1
            )

            # compute grad w.r.t. samples
            grad = x.grad
            print(torch.max(grad).item())
            print(torch.min(grad).item())
            # Langevin update
            x = x + learning_rate * grad + sigma * noise
            x = x.clamp(0, 1).detach()
            x.requires_grad_(True)
            return x

        total_warmup_steps = warmup_rounds * warmup_inner_steps
        global_step = 0

        for r in range(warmup_rounds):
            for s in range(warmup_inner_steps):
                initial_samples = one_step_update(initial_samples, global_step, total_warmup_steps, final=True)
                global_step += 1

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

            x_uint8 = (
                initial_samples.detach()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(0, 2, 3, 1)   # N,H,W,C
                .cpu()
                .numpy()
            )

            ims = []
            for k in range(x_uint8.shape[0]):
                pil = Image.fromarray(x_uint8[k])
                t = transform(pil)
                ims.append(t)

            initial_samples = torch.stack(ims, dim=0).to(self.device, non_blocking=True)
            initial_samples = initial_samples.clamp(0, 1).squeeze().detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        for step in range(num_steps):
            initial_samples = one_step_update(initial_samples, step, num_steps, final=True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    def two_stage_latent_jem_pareto_multi_conditional_sample_upgrad(
        self,
        conditional_index_list,
        conditional_label_list,
        initial_samples,
        num_steps,
        learning_rate,
        sigma,
        retain_graph=False,
        return_samples_each_step=False,
        model=None
    ):
        if model is None:
            model = self.model

        # ---------------------------
        # Stage-1 transform (PIL pipeline, safe with ToTensor)
        # ---------------------------
        im_size = initial_samples.shape[-1]  # assume (N,3,H,W)
        color_transform = get_color_distortion()
        transform = transforms.Compose([transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(),color_transform])

        # warmup hyperparameters: fixed 10x20 (function signature unchanged)
        warmup_rounds = 20
        warmup_inner_steps = 60

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        def one_step_update(x, step, total_steps, final=False):
            # reset noise
            noise = torch.randn_like(x)
            x.requires_grad_(True)

            logits = model(x).view(-1, 23, 2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K
            # E(Y|X)
            if not final:
                dis_weight = 1
            else:
                #dis_weight = 1 - step / total_steps
                dis_weight = 1 / (1 + np.log(1 + step))  # logarithmic decay: slower at first, faster later
            target_list = [
                dis_weight * (conditional_logits[:, idx, 0].sum() - conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                if conditional_label_list[idx]
                else
                dis_weight * (conditional_logits[:, idx, 1].sum() - conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                for idx in range(num_attributes)
            ]
            # E(X)

            target_list.append(torch.logsumexp(logits, dim=-1).sum(dim=-1).sum())

            pref_vector = torch.ones(len(target_list)-1, device=x.device)
            # ---- Key: clear old grad to avoid accumulation. ----
            if x.grad is not None:
                x.grad.zero_()

            pref_vector = torch.ones(len(target_list)-1, device=x.device)
            grad, grad_up12, g3, _ = upgrad_n_plus_final(
                    target_list=target_list,
                    x=x,
                    pref_vector_n=pref_vector,
                    norm_eps=1e-4,
                    reg_eps=1e-4,
                    qp_iters=50,
                    verbose=True,
                    n = len(pref_vector))
            # Langevin update
            x = x + learning_rate * grad + sigma * noise
            x = x.clamp(0, 1).detach()
            x.requires_grad_(True)
            return x

        total_warmup_steps = warmup_rounds * warmup_inner_steps
        global_step = 0

        for r in range(warmup_rounds):
            for s in range(warmup_inner_steps):

                initial_samples = one_step_update(initial_samples, global_step, total_warmup_steps, final=True)
                global_step += 1

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

            x_uint8 = (
                initial_samples.detach()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(0, 2, 3, 1)   # N,H,W,C
                .cpu()
                .numpy()
            )

            ims = []
            for k in range(x_uint8.shape[0]):
                pil = Image.fromarray(x_uint8[k])
                t = transform(pil)
                ims.append(t)

            initial_samples = torch.stack(ims, dim=0).to(self.device, non_blocking=True)
            initial_samples = initial_samples.clamp(0, 1).squeeze().detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        for step in range(num_steps):
            initial_samples = one_step_update(initial_samples, step, num_steps, final=True)
            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()

    # argmin (E(Y1|X),E(Y2|X),E(X)) learning rate normal
    def two_stage_latent_jem_multi_conditional_sample(
        self,
        conditional_index_list,
        conditional_label_list,
        initial_samples,
        num_steps,
        learning_rate,
        sigma,
        retain_graph=False,
        return_samples_each_step=False,
        model=None
    ):
        if model is None:
            model = self.model

        # ---------------------------
        # Stage-1 transform (PIL pipeline, safe with ToTensor)
        # ---------------------------
        im_size = initial_samples.shape[-1]  # assume (N,3,H,W) and H=W
        color_transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        warmup_transform = transforms.Compose([
            transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform,
            transforms.ToTensor(),  # PIL -> Tensor in [0,1]
        ])

        # warmup hyperparameters: fixed constants following prior style (signature unchanged)
        warmup_rounds = 20
        warmup_inner_steps = 30

        # ---------------------------
        # init
        # ---------------------------
        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        # ---------------------------
        # one Langevin step (logic kept as original)
        # ---------------------------
        def one_step_update(x, step, total_steps):
            noise = torch.randn_like(x)
            logits = model(x).view(-1, 23, 2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            if total_steps > 1:
                t = step / (total_steps - 1)
            else:
                t = 0.0

            w_cond = 1.0 - 0.9 * t
            w_gen  = 0.1 + 0.9 * t
            w_cond = 1
            w_gen = 1
            target_list = [
                w_cond * (
                    (conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                    if conditional_label_list[idx]
                    else
                    (conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :], dim=-1).sum())
                )
                for idx in range(num_attributes)
            ]
            target_list.append(w_gen * torch.logsumexp(logits, dim=-1).sum(dim=-1).sum())

            grad = torch.autograd.grad(sum(target_list), x, create_graph=retain_graph)[0]
            grad = torch.clamp(grad, min=-0.03, max=0.03)
            print(torch.max(grad))
            print(torch.min(grad))

            x = x + learning_rate * grad + sigma * noise
            x = x.clamp(0, 1).detach()
            x.requires_grad_(True)
            return x

        # ---------------------------
        # Stage 1: warm up (with transform)
        # ---------------------------
        total_warmup_steps = warmup_rounds * warmup_inner_steps
        global_step = 0

        for r in range(warmup_rounds):
            for s in range(warmup_inner_steps):
                initial_samples = one_step_update(initial_samples, global_step, total_warmup_steps)
                global_step += 1

                if return_samples_each_step:
                    sample_list.append(initial_samples.detach().cpu())

            # After each warmup round, apply one transform augmentation (Stage 1 only).
            # Tensor(CUDA) -> uint8 numpy -> PIL -> transform -> Tensor(CPU) -> stack -> CUDA
            x_uint8 = (
                initial_samples.detach()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(0, 2, 3, 1)   # N,H,W,C
                .cpu()
                .numpy()
            )

            ims = []
            for k in range(x_uint8.shape[0]):
                pil = Image.fromarray(x_uint8[k])
                t = warmup_transform(pil)  # torch.Tensor (3,H,W) on CPU
                ims.append(t)

            initial_samples = torch.stack(ims, dim=0).to(self.device, non_blocking=True)
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        # ---------------------------
        # Stage 2: refine (no transform) - exactly the original loop
        # ---------------------------
        for step in range(num_steps * 2):
            # reset noise
            noise = torch.randn_like(initial_samples)
            logits = model(initial_samples).view(-1,23,2)  # N, 23, 2

            conditional_logits = logits[:, conditional_index_list, :]  # N, K, 2
            num_attributes = conditional_logits.shape[1]  # K

            if num_steps > 1:
                t = step / (num_steps - 1)
            else:
                t = 0.0

            # 1 and 2 decay linearly from 1 to 0.1: w = 1 - 0.9 * t
            w_cond = 1.0 - 0.9 * t         # step=0 -> 1.0, step=last -> 0.1
            w_cond = 1
            # 3 increases linearly from 0.1 to 1: w = 0.1 + 0.9 * t
            w_gen = 0.1 + 0.9 * t          # step=0 -> 0.1, step=last -> 1.0
            w_gen = 1
            # E(Y|X)
            target_list = [
                w_cond * (conditional_logits[:, idx, 0].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                if conditional_label_list[idx]
                else w_cond * (conditional_logits[:, idx, 1].sum() - torch.logsumexp(conditional_logits[:, idx, :],dim=-1).sum())
                for idx in range(num_attributes)
            ]
            # E(X)
            target_list.append(w_gen * torch.logsumexp(logits,dim=-1).sum(dim=-1).sum())
    
            grad = torch.autograd.grad(sum(target_list), initial_samples, create_graph=retain_graph)[0]
            grad = torch.clamp(grad, min=-0.03, max=0.03)
            print(torch.max(grad))
            print(torch.min(grad))
            # Langevin update
            initial_samples = initial_samples + learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list

        elif retain_graph:
            return initial_samples

        else:
            return initial_samples.detach().cpu()

'''
gibbs energy based model trainer for different datasets
'''
class gibbs_jem_lightning_module(pl.LightningModule):

    def __init__(self, args):
        super(gibbs_jem_lightning_module, self).__init__()
        self.args = args
        self.model = celeba_resnet()

        self.sampler = ReplayBuffer(
            max_size=args.buffer_size,
            replace_prob=args.replace_prob,
            img_shape=(3, args.img_size, args.img_size),
            device=self.device
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.use_ema   = getattr(args, "use_ema", False)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_eval  = getattr(args, "ema_eval", True)

        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

    def _update_ema(self):
        if (not self.use_ema) or (self.model_ema is None):
            return
        ema_model_update(self.model, self.model_ema, mu=self.ema_decay)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2)
        )

    @staticmethod
    def _energy_xy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        PoJ joint energy:  E(x,y) = - sum_i logits_i[y_i]
        logits: [B,23,2]
        y:      [B,23] (0/1)
        return: [B]
        """
        y = y.long()
        picked = logits.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # [B,23]
        return -picked.sum(dim=-1)  # [B]

    @torch.no_grad()
    def _sample_y_given_x(self, x: torch.Tensor, model=None) -> torch.Tensor:
        """
        y ~ p(y|x), fully factorized:
            y_i ~ Categorical(logits=logits[:,i,:])
        return y: [B,23]
        """
        if model is None:
            model = self.model
        logits = model(x).view(-1, 23, 2)
        y = torch.distributions.Categorical(logits=logits).sample()
        return y

    def _langevin_x_given_y(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
        model=None,
        num_steps: int = 60,
        step_size: float = 1e-3,
        sigma: float = 0.01,
        clamp: bool = True,
    ) -> torch.Tensor:
        """
        Langevin sample for x | y using E(x,y) = -sum logits[y].
        (Z(y) is const wrt x, so grad wrt x is just grad of the score.)
        """
        if model is None:
            model = self.model

        x = x_init.detach()
        for _ in range(num_steps):
            x.requires_grad_(True)

            logits = model(x).view(-1, 23, 2)
            e = self._energy_xy_from_logits(logits, y).sum()  # scalar
            grad = torch.autograd.grad(e, x, create_graph=False, retain_graph=False)[0]

            with torch.no_grad():
                noise = torch.randn_like(x)
                x = x - step_size * grad + sigma * noise
                if clamp:
                    x = torch.clamp(x, 0.0, 1.0)

            x = x.detach()

        return x

    def gibbs_sample_x_y(
        self,
        x_init: torch.Tensor,
        model_for_gibbs=None,
        gibbs_steps: int = 1,
        gibbs_k_steps: int = 60,
        gibbs_n_steps: int = 1,
        step_size: float = 1e-3,
        sigma: float = 0.01,
    ):
        """
        Gibbs chain:
          repeat gibbs_steps times:
            y <- sample p(y|x)   (repeat gibbs_n_steps)
            x <- Langevin for p(x|y) (gibbs_k_steps steps)
        """
        if model_for_gibbs is None:
            model_for_gibbs = self.model

        # Use eval mode during sampling to avoid BN-stat pollution by negative samples; keep train mode for training.
        was_training = model_for_gibbs.training
        model_for_gibbs.eval()

        x = x_init.detach()
        y = self._sample_y_given_x(x, model=model_for_gibbs)

        for _ in range(gibbs_steps):
            for _ in range(gibbs_n_steps):
                y = self._sample_y_given_x(x, model=model_for_gibbs)

            x = self._langevin_x_given_y(
                x_init=x,
                y=y,
                model=model_for_gibbs,
                num_steps=gibbs_k_steps,
                step_size=step_size,
                sigma=sigma,
                clamp=True,
            )

        if was_training:
            model_for_gibbs.train()

        return x.detach(), y.detach()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        loss_list = []

        logits = self.model(batch_x).view(-1, 23, 2)
        batch_y = batch_y.view(-1, 23).long()

        B = batch_x.shape[0]
        batch_x_init = self.sampler.sample(B, train=True).to(batch_x.device)

        gibbs_steps   = getattr(self.args, "gibbs_steps", 5)
        gibbs_k_steps = getattr(self.args, "gibbs_k_steps", 1)
        gibbs_n_steps = getattr(self.args, "gibbs_n_steps", 1)
        step_size     = getattr(self.args, "step_size", 1.0)
        sigma         = getattr(self.args, "sigma", 0.01)

        use_ema_for_sampling = getattr(self.args, "use_ema_for_sampling", False)

        if use_ema_for_sampling and self.use_ema and (self.model_ema is not None):
            model_for_gibbs = self.model_ema
        else:
            model_for_gibbs = self.model

        # Langevin requires gradients, so do not use no_grad.
        batch_x_neg, batch_y_neg = self.gibbs_sample_x_y(
            x_init=batch_x_init,
            model_for_gibbs=model_for_gibbs,
            gibbs_steps=gibbs_steps,
            gibbs_k_steps=gibbs_k_steps,
            gibbs_n_steps=gibbs_n_steps,
            step_size=step_size,
            sigma=sigma,
        )

        # Write back to replay buffer (store only x; keep buffer structure unchanged as much as possible).
        self.sampler.update_buffer(batch_x_neg.detach().to("cpu"))

        # 3) CD: switch to joint-energy contrast (closer to Gibbs-JEM).
        #     pos uses real (x,y), neg uses Gibbs-chain terminal (x~, y~).
        neg_logits = self.model(batch_x_neg.detach()).view(-1, 23, 2)

        pos_energy = self._energy_xy_from_logits(logits, batch_y).mean()
        neg_energy = self._energy_xy_from_logits(neg_logits, batch_y_neg).mean()

        cd_loss = pos_energy - neg_energy

        # 4) KL regularization block: keep behavior close to original (still using uncond energy for distance regularization).
        if self.args.kl_loss:
            batch_buffer = self.sampler.sample(100, train=True).to(batch_x.device)
            batch_buffer = deq_x(batch_buffer)

            kl_loss_1 = -torch.logsumexp(
                self.model(batch_x_neg).view(-1, 23, 2), dim=-1
            ).sum(dim=-1).mean()

            dist_matrix = torch.norm(
                batch_x_neg.view(batch_x_neg.shape[0], -1)[None, :, :]
                - batch_buffer.view(batch_buffer.shape[0], -1)[:, None, :],
                p=2,
                dim=-1,
            )
            mins = dist_matrix.min(dim=1).values
            min_fin = mins > 0
            kl_loss_2 = -torch.log(mins[min_fin]).mean()

            loss_list.append(cd_loss + 1 * kl_loss_1 + 1 * 0.3 * kl_loss_2)
        else:
            loss_list.append(cd_loss)

        if self.args.kl_loss:
            self.log("train_kl_loss_1", kl_loss_1, sync_dist=True)
            self.log("train_kl_loss_2", kl_loss_2, sync_dist=True)

        self.log("train_CD_loss", cd_loss, sync_dist=True)
        self.log("train_pos_energy", pos_energy, sync_dist=True)
        self.log("train_neg_energy", neg_energy, sync_dist=True)

        return sum(loss_list)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_ema()

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        logits = self.model(batch_x).view(-1, 23, 2)
        batch_y = batch_y.view(-1, 23)

        aurocs = []
        for i in range(23):
            y_true = batch_y[:, i].detach().cpu().numpy()
            y_prob = F.softmax(logits[:, i, :], dim=-1)[:, 1].detach().cpu().numpy()
            if len(set(y_true)) < 2:
                continue
            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)
            self.log(
                f'val_auroc_{ATTRIBUTE_LIST[self.args.label_list[i]][0]}',
                auroc,
                sync_dist=True,
                prog_bar=False
            )

        if len(aurocs) > 0:
            mean_auroc = sum(aurocs) / len(aurocs)
            self.log("val_mean_auroc", mean_auroc, sync_dist=True, prog_bar=True)
        else:
            self.log("val_mean_auroc", -1, sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self):
        batch_x = torch.rand(32, 3, self.args.img_size, self.args.img_size).to(self.device)

        if self.use_ema and self.ema_eval and (self.model_ema is not None):
            model_for_sampling = self.model_ema
        else:
            model_for_sampling = self.model

        with torch.enable_grad():
            sample = self.sample(
                batch_x, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(
                    self.args.save_dir,
                    f"unconditional_sampling_from_noise_epoch_{self.current_epoch}.png"
                )
            )

        with torch.enable_grad():
            samples_from_buffer = self.sampler.sample(batch_x.shape[0]).to(batch_x.device)
            sample = self.sample(
                samples_from_buffer, num_steps=self.args.sgld_steps, learning_rate=self.args.sgld_lr, sigma=self.args.sgld_sigma,
                model=model_for_sampling
            ).float().to(batch_x.device)
            imgs_to_save = torchvision.utils.make_grid(sample, nrow=10)
            torchvision.utils.save_image(
                imgs_to_save,
                os.path.join(self.args.save_dir,
                             f"unconditional_sampling_from_buffer_epoch_{self.current_epoch}.png")
            )

    def sample(self, initial_samples, num_steps, learning_rate, sigma,
               retain_graph=False, return_samples_each_step=False, model=None):

        if model is None:
            model = self.model

        initial_samples = initial_samples.clone().detach().to(self.device)
        initial_samples.requires_grad_(True)

        if return_samples_each_step:
            sample_list = [initial_samples.detach().cpu()]

        for step in range(num_steps):
            noise = torch.randn_like(initial_samples)

            energy = -torch.logsumexp(model(initial_samples).view(-1, 23, 2), dim=-1).sum(dim=-1).sum()

            if step == num_steps - 1 and retain_graph:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=True)[0]
            else:
                grad = torch.autograd.grad(energy, initial_samples, create_graph=False)[0]

            initial_samples = initial_samples - learning_rate * grad + sigma * noise
            initial_samples = initial_samples.clamp(0, 1).detach()
            initial_samples.requires_grad_(True)

            if return_samples_each_step:
                sample_list.append(initial_samples.detach().cpu())

        if return_samples_each_step:
            return sample_list
        elif retain_graph:
            return initial_samples
        else:
            return initial_samples.detach().cpu()
