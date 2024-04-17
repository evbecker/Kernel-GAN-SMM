eps = 1e-10
import torch
from torch import acos, pi, einsum, clip
from torch.nn.functional import cosine_similarity
domain_check = lambda u: clip(u, -1+eps, 1-eps)
kappa_0 = lambda u: (1-acos(domain_check(u))/pi)
kappa_0_ = lambda u: (1/(1-domain_check(u).pow(2)).sqrt()/pi)
kappa_1 = lambda u: (u*kappa_0(u) + (1-domain_check(u).pow(2)).sqrt()/pi)
kappa_1_ = kappa_0
import time

torch.set_default_dtype(torch.float32)

def norm_M(X,M):
    return (X*(X @ M)).sum(dim=-1).sqrt()

def cosine_similarity_M(X,Z,M):
    nx,dx=X.shape
    nz,dz=Z.shape
    assert dx==dx
    return (X @ M @ Z.T) /norm_M(X,M).view(nx,1)/norm_M(Z,M).view(nz)
    #return einsum('na,mb,ab->nm',X,Z,M)/norm_M(X,M).view(nx,1)/norm_M(Z,M).view(nz)

def ntk_relu(X, Z=None, depth=1, bias=0., M=None):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.

    depth  (int): number of weight layers of the network
    bias (float): (default=0.)
    """
    nx, d = X.shape
    if M is None: M = torch.eye(d).to(X.device)
    norm_x = norm_M(X,M).view(nx,1)
    if Z is not None:
        nz, dz = Z.shape
        assert d==dz
        norm_z = norm_M(Z,M).view(nz)
    else:
        nz, norm_z, Z = nx, norm_x.view(nx), X
    S = cosine_similarity_M(X, Z, M)
    Q = S + bias**2/norm_x/norm_z
    for k in range(1, depth):
        Q = Q * kappa_0(S) +  bias**2/norm_x/norm_z
        S = kappa_1(S)
        Q = S + Q
    return S*norm_x*norm_z, Q*norm_x*norm_z

def grad1_ntk_relu(X, Z=None, depth=1, bias=0, M=None):
    """
    Returns the gradient of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.

    depth  (int): number of weight layers of the network
    bias (float): (default=0.)
    """
    nx, d = X.shape
    if M is None: M = torch.eye(d).to(X.device)

    norm_x = norm_M(X,M).view(nx,1,1)
    if Z is not None:
        nz, dz = Z.shape
        assert d==dz
        norm_z = norm_M(Z,M).view(nz,1)
    else:
        nz, norm_z, Z = nx, norm_x.view(nx,1), X
    S = cosine_similarity_M(X, Z, M).view(nx,nz,1)
    S_ = (Z @ M/norm_z/norm_x - S * (X @ M).view(nx,1,d)/norm_x.pow(2))
    Q = S + bias**2/norm_x/norm_z
    Q_ = S_ - bias**2*(X @ M).view(nx,1,d)/norm_x.pow(3)/norm_z
    if Z is X: print('Z is X')
    for k in range(1, depth):
        Q_ = Q_*kappa_0(S) + S_ *Q*kappa_0_(S) - bias**2*(X @ M).view(nx,1,d)/norm_x.pow(3)/norm_z
        S_ = S_ * kappa_1_(S)
        Q_ = S_ + Q_
        Q = Q * kappa_0(S) + bias**2/norm_x/norm_z
        S = kappa_1(S)
        Q =  S + Q
    return ((S*norm_x*norm_z).squeeze(),
        (Q*norm_x*norm_z).squeeze(),
        S_*norm_x*norm_z + S*(X @ M).view(nx,1,d)*norm_z/norm_x,
        Q_*norm_x*norm_z + Q*(X @ M).view(nx,1,d)*norm_z/norm_x)

def grad1vp_ntk_relu(X, Z=None, V=None, depth=1, bias=0, M=None):
    """
    Returns the product of matrix `V` with
    gradient of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.
    equivalent to [grad_x (K(x,Z) @ V) for x=xi in X]
    input shapes::
        X (n, d)        Z (m, d)        V (m, c)
    output shape::
        (c, n, d)

    depth  (int): number of weight layers of the network
    bias (float): (default=0.)
    """
    nx, d = X.shape
    if M is None: M = torch.eye(d).to(X.device)
    norm_x = norm_M(X,M).view(nx,1)
    if Z is not None:
        nz, dz = Z.shape
        assert d==dz
        norm_z = norm_M(Z,M)
    else: nz, norm_z, Z = nx, norm_x.view(nx), X
    if V is None:
        V=torch.ones(nz,1)
        c=1
    elif len(V.shape)==1:
        nv, c = V.shape[0], 1
        V = V.reshape(-1,1)
        assert nv==nz
    else:
        nv, c = V.shape
        assert nv==nz

    S = cosine_similarity_M(X, Z, M)
    Pi_km1 = 1
    for k in range(1, depth):
        Pi_km1 = Pi_km1 * kappa_1_(S)
        S = kappa_1(S)

    t = V.reshape(-1, c, 1) @ (Z @ M).reshape(-1, 1, d)
    # m x c x d
    # print(t.shape)
    t = torch.swapaxes(t, 1, 0)
    g1 = Pi_km1 @ t
    #g1 = einsum('nm,mc,md->cnd', Pi_km1, V, Z@M)
    g2 = (X @ M)*(X*g1).sum(-1).view(c,nx,1)/norm_x.pow(2)
    g3 = (X @ M) /norm_x * (S @ (V * norm_z.unsqueeze(-1))).T.unsqueeze(-1)
    #g3 = (X @ M)/norm_x*einsum('nm,mc,m->cn',S,V,norm_z).unsqueeze(-1)
    return (g1 - g2 + g3).squeeze()

if __name__ == "__main__":
    n, m, d, c = 10000, 10000, 3072, 10
    depth,bias=3,0.

    X = torch.randn(n, d).cuda()
    Z = torch.randn(m, d).cuda()
    V = torch.randn(m, c).cuda()

    #SigmaXZ_v1 = einsum('nmd,mc->cnd',
    #    grad1_ntk_relu(X,Z,depth=depth,bias=bias)[2], V)
    start = time.time()
    SigmaXZ_v2 = grad1vp_ntk_relu(X,Z,V=V,depth=depth,bias=bias)
    end = time.time()
    print(end - start)
    #print(torch.allclose(SigmaXZ_v1, SigmaXZ_v2))
    #print((SigmaXZ_v1 - SigmaXZ_v2).abs().max())