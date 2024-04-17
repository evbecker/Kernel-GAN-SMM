import numpy as np
import torch
import torch.distributions as distr

# dataset 'factory'
def get_distr(shape, **kwargs):
    '''
    input: dataset distribution params
    output: dataset object
    '''
    distributions= {'point':Point, 'ring':Ring, 'gauss':Normal, 'ringMOG':RingMOG}
    return distributions[shape](**kwargs)


# single true point at the origin in n-dimensions
class Point():
    def __init__(self, dim=2, offset=None, value=1, device=torch.device('cpu')):
        self.dim = dim
        assert(offset is None or len(offset)==dim)
        self.offset = offset
        self.value = value
        self.device = device

    def sample_n(self, n_samples):
        if self.offset is not None:
            tensorX = torch.zeros(n_samples,self.dim) + torch.tensor(self.offset) 
        else:
            tensorX = torch.zeros(n_samples,self.dim)
        tensorY = self.value*torch.ones(n_samples)
        return tensorX.to(self.device), tensorY.to(self.device)

# essentially a wrapper for returning normal tensors plus label
class Normal():
    def __init__(self, dim=2, loc=None, scale=None, value=1, device=torch.device('cpu')):
        self.dim = dim
        self.loc = np.zeros(dim) if loc is None else loc
        self.scale = np.ones(dim) if scale is None else scale
        self.value = value
        self.device = device

    def sample_n(self, n_samples):
        X = np.random.normal(self.loc, self.scale, size=(n_samples, self.dim))
        tensorX = torch.tensor(X, dtype=torch.float)
        tensorY = self.value*torch.ones(n_samples)
        return tensorX.to(self.device), tensorY.to(self.device)


# loosely based off of https://github.com/poolio/unrolled_gan/blob/master/Unrolled%20GAN%20demo.ipynb
class RingMOG():
    def __init__(self, dim=2, n_mixture=8, std=0.01, radius=1.0,
                 value=1, device=torch.device('cpu')):
        # to avoid generating two points at 0 and 2pi rads
        # we disregard the last theta
        thetas = np.linspace(0, 2 * np.pi, n_mixture+1)[:-1]
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.centers = torch.Tensor(np.stack((xs,ys)).T)
        if isinstance(std, list):
          self.stds = torch.Tensor(std)
        else:
          self.stds = torch.ones((n_mixture,dim))*std
        self.mix = distr.Categorical(torch.ones(n_mixture,))
        self.comp = distr.Independent(distr.Normal(self.centers, self.stds), 1)
        self.mog = distr.mixture_same_family.MixtureSameFamily(self.mix, self.comp)
        self.value = value
        self.device = device

    def sample_n(self, n_samples):
        tensorX = self.mog.sample((n_samples,))
        tensorY = self.value*torch.ones(n_samples)
        return tensorX.to(self.device), tensorY.to(self.device)


class Ring():
    def __init__(self, dim=2, n_mixture=8, offset=None, std=0, radius=1.0, 
                 value=1, device=torch.device('cpu')):
        # to avoid generating two points at 0 and 2pi rads
        # we disregard the last theta
        if (dim==2):
            thetas = np.linspace(0, 2 * np.pi, n_mixture+1)[:-1]
            xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
            self.centers = torch.Tensor(np.stack((xs,ys)).T)
        else:
            X = np.random.normal(0,1, (n_mixture,dim))
            Xnorm = np.sqrt(np.sum(X**2, axis=1))
            X = radius* X / Xnorm[:,None]
            self.centers = torch.Tensor(X)
            
        self.n_mixture = n_mixture
        self.dim=dim
        assert(offset is None or len(offset)==dim)
        self.offset = torch.zeros(dim) if offset is None else torch.Tensor(offset)
        self.std= 0 if std is None else std
        self.value = value
        self.device = device

    def sample_n(self, n):
        # tensorX is a nxd matrix where
        # n = n_per_mix*n_mixture and
        # d is the dimension
        n_per_mix = n//self.n_mixture
        tensorX = self.centers.repeat(n_per_mix, 1) + self.offset.repeat(n,1)
        noise = torch.tensor(np.random.normal(loc=0, scale=self.std, size=(n, self.dim)))
        tensorX+=noise
        tensorY = self.value*torch.ones(self.n_mixture*n_per_mix)
        return tensorX.to(self.device), tensorY.to(self.device)