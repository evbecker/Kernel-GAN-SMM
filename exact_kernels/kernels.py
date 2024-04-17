import torch
from functools import partial
from ntk import grad1vp_ntk_relu, ntk_relu

### Distances ###
def euclidean(X, Z=None):
    '''
    inp: X (nx, d), Z (nz,d)
    out: D (nx,nz) where D_ij = ||x_i-z_j||_2^2
    '''
    X_norm2 = X.pow(2).sum(-1)
    Z, Z_norm2 = (Z, Z.pow(2).sum(-1)) if Z is not None else (X, X_norm2)
    X_dot_Z = X @ Z.T
    return X_norm2[:,None] -2*X_dot_Z + Z_norm2[None,:]

def mahalanobis_old(X, M=None, Z=None):
    '''
    inp: X (nx, d), M (d,d), Z (nz,d)
    out: D (nx,nz) where D_ij = ||x_i-z_j||_M^2
    '''
    M = M if M is not None else torch.eye(X.shape[1])
    X_norm2 = torch.diag(X@M@X.T)
    Z, Z_norm2 = (Z, torch.diag(Z@M@Z.T)) if Z is not None else (X, X_norm2)
    X_dot_Z = X @ M @ Z.T
    return X_norm2[:,None] -2*X_dot_Z + Z_norm2[None,:]

def mahalanobis(X, M=None, Z=None):
    '''
    inp: X (nx, d), M (d,d), Z (nz,d)
    out: D (nx,nz) where D_ij = ||x_i-z_j||_M^2
    '''
    M = M if M is not None else torch.eye(X.shape[1])
    X_norm2 = torch.einsum('nd,dn->n',X@M,X.T) # diag of X@M@X.T
    Z, Z_norm2 = (Z,  torch.einsum('nd,dn->n',Z@M,Z.T)) if Z is not None else (X, X_norm2)
    X_dot_Z = X @ M @ Z.T
    return X_norm2[:,None] -2*X_dot_Z + Z_norm2[None,:]

### Kernels ###    
def gaussian(X, Z=None, bandwidth=1.):
    return euclidean(X,Z).div(-2*bandwidth**2).exp()

def m_gaussian(X, M=None, Z=None, bandwidth=1.):
    return mahalanobis(X, M, Z).div(-2*bandwidth**2).exp()

def laplacian(X, Z=None, bandwidth=1.):
    return euclidean(X,Z).sqrt().div(-bandwidth).exp()

def m_laplacian(X, M=None, Z=None, bandwidth=1.):
    return mahalanobis(X, M, Z).sqrt().div(-bandwidth).exp()

def rq(X, Z=None, bandwidth=1., mixing=1.):
    return (1+euclidean(X,Z).div(2*mixing*bandwidth**2)).pow(-mixing)

def m_rq(X, M=None, Z=None, bandwidth=1., mixing=1.):
    return (1+mahalanobis(X,M,Z).div(2*mixing*bandwidth**2)).pow(-mixing)

### Kernel Gradients ###
def gaussian_grad_vp(X, M=None, Z=None, v=None, bandwidth=1.):
    M = M if M is not None else torch.eye(X.shape[1])
    v = v if v is not None else torch.ones(Z.shape[0])
    KXZ = m_gaussian(X, M, Z, bandwidth)
    return  (KXZ @ (Z@M * v[:,None]) - (KXZ @ v)[:,None] * X@M).div(bandwidth**2)

def old_laplacian_grad_vp(X, Z=None, v=None, bandwidth=1.):
    v = v if v is not None else torch.ones(Z.shape[0])
    KXZ = laplacian(X, Z, bandwidth)/euclidean(X,Z).sqrt()
    return  (KXZ @ (Z * v[:,None]) - (KXZ @ v)[:,None] * X).div(bandwidth).nan_to_num(0.0,0.0,0.0)

def laplacian_grad_vp(X, M=None, Z=None, v=None, bandwidth=1.):
    M = M if M is not None else torch.eye(X.shape[1])
    v = v if v is not None else torch.ones(Z.shape[0])
    KXZ = m_laplacian(X, M, Z, bandwidth)/mahalanobis(X,M,Z).sqrt()
    return  (KXZ @ (Z@M * v[:,None]) - (KXZ @ v)[:,None] * X@M).div(bandwidth).nan_to_num(0.0,0.0,0.0)

def rq_grad_vp(X, M=None, Z=None, v=None, bandwidth=1., mixing=1.):
    v = v if v is not None else torch.ones(Z.shape[0])
    KXZ = m_rq(X, M, Z, bandwidth, mixing).pow(1+1/mixing)
    return  (KXZ @ (Z@M * v[:,None]) - (KXZ @ v)[:,None] * X@M).div(bandwidth**2)

### Kernel Decorators ###
def old_uvp_decorator(grad_vp):
    # gradient for each point is now weighted by scalar u_i  
    def grad1_uvp_wrapper(X, M=None, Z=None, u=None, v=None, **kwargs):
        if u is None:
            u = torch.ones(X.shape[0], 1)
        return  u.reshape(-1, 1) * grad_vp(X, Z, v, **kwargs)
    
    return grad1_uvp_wrapper

def uvp_decorator(grad_vp):
    # gradient for each point is now weighted by scalar u_i  
    def grad1_uvp_wrapper(X, M=None, Z=None, u=None, v=None, **kwargs):
        if u is None:
            u = torch.ones(X.shape[0], 1)
        return  u.reshape(-1, 1) * grad_vp(X, M, Z, v, **kwargs)
    
    return grad1_uvp_wrapper

def multiscale_decorator(grad1_uvp):
    #  fcn is now the weighted sum of grad1_uvp at different bandwidths
    def multiscale_wrapper(X, M=None, Z=None, u=None, v=None, width_params=[1], weights=None, **kwargs):
        if weights is None:
            weights = torch.ones(len(width_params))/len(width_params)
        else:
            weights = weights/torch.sum(weights)
        grad_parts = torch.stack([grad1_uvp(X,M,Z,u,v,bandwidth=width,**kwargs) for width in width_params])
        return torch.einsum('a, abc->bc', weights, grad_parts)
    
    return multiscale_wrapper

class GaussKernel():
    def __init__(self, d, width_param=1, M=None, weights=None):
        self.d = d
        self.width_param = width_param
        self.M = M if M is not None else torch.eye(self.d)
        self.weights = weights
        self.multiscale = isinstance(width_param, list)
        # choosing grad and eval settings
        self.grad1_uvp = uvp_decorator(gaussian_grad_vp)
        if self.multiscale: # using a multiscale kernel
            self.grad1_uvp = multiscale_decorator(self.grad1_uvp)
            self._grad = partial(self.grad1_uvp, width_params=self.width_param, weights=self.weights)
        else:
            self._grad = partial(self.grad1_uvp, bandwidth=self.width_param)
        self._eval = partial(m_gaussian, bandwidth=self.width_param) #TODO fix ms_eval

    def grad(self, X, Z=None, u=None, v=None):
        """
        grad_i = u[i] * [sum_{j} v[j] * K(X[i], Z[j])]
        ::input shapes:
            X (nx, d), Z (nz, d)
            u (nx,), v (nz,)
        ::intermediate shapes
            K(X,Z) (nx, nz)
        ::output shape
            (nx, d)
        """
        return self._grad(X,self.M,Z,u,v)

    def eval(self, X, M=None, Z=None):
        """
        ::output shape
            K(x,z) (nx,nz)
        """
        return self._eval(X,self.M,Z)


### General Kernel Class ###
class Kernel():
    def __init__(self, kernel_type, width_param=1, mix_param=1, depth=1, weights=None):
        self.kernel_type=kernel_type
        self.multiscale = isinstance(width_param, list)
        if kernel_type=='gaussian':
            grad1_uvp = uvp_decorator(gaussian_grad_vp)
            self._grad = partial(grad1_uvp, bandwidth=width_param)
            self._eval = partial(m_gaussian, bandwidth=width_param)
        elif kernel_type=='laplacian':
            grad1_uvp = uvp_decorator(laplacian_grad_vp)
            self._grad = partial(grad1_uvp, bandwidth=width_param)
            self._eval = partial(m_laplacian, bandwidth=width_param)
        elif kernel_type=='rq':
            grad1_uvp = uvp_decorator(rq_grad_vp)
            self._grad = partial(grad1_uvp, bandwidth=width_param, mixing=mix_param)
            self._eval = partial(rq, bandwidth=width_param, mixing=mix_param)
        elif kernel_type=='nngp': #TODO, this might be broken with new mahalanobis distances
            grad1_uvp = old_uvp_decorator(grad1vp_ntk_relu)
            self._grad = partial(grad1_uvp, depth=depth, bias=0.)
            self._eval = partial(ntk_relu, depth=depth, bias=0.)
        elif kernel_type=='gaussian_ms':
            grad1_uvp = multiscale_decorator(uvp_decorator(gaussian_grad_vp))
            self.grad = partial(grad1_uvp, width_params=width_param, weights=weights)
            self.eval = partial(gaussian, bandwidth=width_param) #TODO fix ms_eval
        elif kernel_type=='laplacian_ms':
            grad1_uvp = multiscale_decorator(uvp_decorator(laplacian_grad_vp))
            self.grad = partial(grad1_uvp, width_params=width_param, weights=weights)
            self.eval = partial(laplacian, bandwidth=width_param) #TODO fix ms_eval
        else:
            raise ValueError('not a valid kernel type')

    def grad(self, X, M=None, Z=None, u=None, v=None):
        """
        grad_i = u[i] * [sum_{j} v[j] * K(X[i], Z[j])]
        ::input shapes:
            X (nx, d), Z (nz, d), M (d, d)
            u (nx,), v (nz,)
        ::intermediate shapes
            K(X,Z) (nx, nz)
        ::output shape
            (nx, d)
        """
        return self._grad(X,M,Z,u,v)

    def eval(self, X, M=None, Z=None):
        """
        ::output shape
            K(x,z) (nx,nz)
        """
        return self._eval(X,M,Z)
