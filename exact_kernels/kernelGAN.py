from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class KernelGAN():
    def __init__(self, G, D, device=torch.device('cpu')):
        # initializing generator and discriminator params
        # G is a generator with generate and update methods
        # KD is a discriminator kernel object with gradient method
        self.G = G
        self.n_gen, self.d = G.n_gen, G.d
        self.D = D
        self.device = device

    def train(self, X_true, p_true, lr_d = 1e-3, lr_g = 1e-3, T = 100, lam = 1e-3, 
              e_threshold=0, RFM=False, resample_interval= 0, log_interval=50, log_grad=False):
        logging.debug(f'Training {self.D.DK.kernel_type} with T={T}, lr_g={lr_g}, lr_d={lr_d}, lam={lam}')
        # distribution + hyperparam initialization
        self.T, self.lam, self.lr_d, self.lr_g = T, lam, lr_d, lr_g
        self.X_true, self.p_true  = X_true.to(self.device), p_true.to(self.device)
        self.G.init_points(T)
        self.D.init_points(self.G.X_gen[0], self.G.p_gen, X_true, p_true, T, e_threshold)
        self.avg_norm_grad = []
        self.M_hist = []
        # calculating generated point trajectories
        curr_X_gen = self.G.generate()
        for t in tqdm(range(T-1), miniters=log_interval):
            # generator update
            gradD = self.D.grad(curr_X_gen, self.G.p_gen)
            self.G.update(t, gradD)
            # discriminator update
            curr_X_gen = self.G.generate()
            self.D.update(curr_X_gen)
            # optional resample noise in generator
            if resample_interval > 0 and t%resample_interval==0:
                self.G.resample()
            # optional feature update in discriminator
            if RFM and t%log_interval==log_interval-1:
                self.D.update_features()
            # logging additional info
            if log_grad and t%log_interval==0:
                self.M_hist.append(self.D.M.clone())
                self.avg_norm_grad.append(gradD.norm(dim=1).mean())
    
    def get_grad_field(self, t, xlim=[-2,2], ylim=[-2,2], nplt=100):
        """
        Calculates the gradient field in 2D
        ::inputs:
            timestep t, x/y limits, num points to plot (per row/col)
        ::outputs:
            vectors x,y,u,v; point i has (xi,yi) location and (ui,vi) gradient 
        """
        assert(self.d==2)
        x0 = torch.linspace(xlim[0], xlim[1], nplt)
        x1 = torch.linspace(ylim[0], ylim[1], nplt)
        X0mat, X1mat = torch.meshgrid(x0, x1, indexing='xy') 
        X = torch.column_stack((X0mat.ravel(), X1mat.ravel())) #(nxd)
        p = torch.ones(X.shape[0])
        # TODO: convert this code to a discriminator grad fcn call
        g1=self.D.DK.grad(X, M=self.D.M, Z=self.X_true, u=p, v=self.p_true)
        g2=self.D.DK.grad(X,  M=self.D.M, Z=self.G.X_gen[:t+1].reshape((t+1)*self.n_gen,self.d), u=p, 
                v=(self.G.p_gen*(1-self.lam*self.lr_d)**torch.arange(t,-1,-1).reshape(-1,1)).ravel())
        grad_field = self.lr_g * self.D.g1_weight(self.lr_d,t, self.lam) * g1 - self.lr_g * self.lr_d * g2

        return X[:,0], X[:,1], grad_field[:,0], grad_field[:,1]
    
class KernelDiscriminator():
    def __init__(self, DK, d, lam=1e-3, lr_d=1e-3, device=torch.device('cpu')):
        # setting hyperparameters
        self.device = device
        self.DK=DK
        self.d = d
        self.M = torch.eye(self.d).to(device)
        self.lr_d=lr_d
        self.lam=lam
        # change gradient when regularizer is 0 to avoid div0 errors
        if lam==0:
            self.g1_weight = lambda lr_d, t, lam: lr_d*t
        else:
            self.g1_weight = lambda lr_d, t, lam: (1-(1 - lr_d * lam)**(t))/lam

    def init_points(self, X_gen_init, p_gen, X_true, p_true, T=100, e_threshold=0):
        # initializing discriminator based on true and gen points
        self.n_gen, _ = X_gen_init.shape
        self.p_gen = p_gen.to(self.device)
        self.X_true = X_true.to(self.device)
        self.p_true = p_true.to(self.device)
        # will forget past generated points once their weight falls below a threshold
        self.t = 0
        self.max_mem = T if e_threshold==0 else int(np.log(e_threshold)/np.log(1-self.lam*self.lr_d))
        self._buffer_len = 0
        self.X_gen_hist = torch.zeros((self.max_mem, self.n_gen, self.d), device=self.device)
        self.update(X_gen_init)

    def grad(self, X, p):
        # input: X (n, d) points, p (n) weights;   out: gradD at X (n,d)
        # g1 are (unweighted) grads at curr X coming from Xtrue
        # g2 are grads at curr X coming from Xgen (including past points)
        g1=self.DK.grad(X, M=self.M, Z=self.X_true, u=p, v=self.p_true)
        g2=self.DK.grad(X, M=self.M, Z=self.X_gen_hist[:self._buffer_len].reshape((self._buffer_len)*self.n_gen,self.d), 
                        u=p, v=(self.p_gen*(1-self.lam*self.lr_d)**torch.arange(self._buffer_len-1,-1,-1, device=self.device).reshape(-1,1)).ravel())
        gradD = self.g1_weight(self.lr_d,self.t,self.lam) * g1 - self.lr_d * g2

        return gradD
    
    def update(self, curr_X_gen):
        # input is (ngen, d) set of current gen points
        # add this to 'end' of internal buffer [t-delta, ..., t-1, t]
        if self._buffer_len < self.max_mem:
            # still room in buffer
            self._buffer_len+=1
            self.X_gen_hist[self._buffer_len-1, :, :] = curr_X_gen
        else:
            # need to delete old point 
            self.X_gen_hist = torch.cat((self.X_gen_hist[1:,:,:], curr_X_gen[None,:,:]),0)
        # need to keep track of total time for weight calculations
        self.t+=1
    
    def update_features(self):
        grad_xgen = self.grad(self.X_gen_hist[self._buffer_len-1], self.p_gen)
        grad_xtrue = self.grad(self.X_true, self.p_true) 
        self.M = grad_xgen.T@grad_xgen/self.n_gen - grad_xtrue.T@grad_xtrue/self.X_true.shape[0]

class PointGenerator():
    def __init__(self, X_gen_init, p_gen, device=torch.device('cpu')):
        self.X_gen_init = X_gen_init.to(device)
        self.n_gen, self.d = X_gen_init.shape
        self.p_gen = p_gen.to(device)
        self.t = 0
        self.device=device

    def init_points(self, T):
        self.X_gen = torch.zeros(T,self.n_gen,self.d, device=self.device)
        self.X_gen[0,:,:] = self.X_gen_init
    
    def generate(self):
        return self.X_gen[self.t]

    def update(self, t, lr_g, gradD):
        self.X_gen[t+1] = self.X_gen[t] + lr_g * gradD
        self.t=t+1

class NormalGenerator():
    def __init__(self, X_gen_init, p_gen, device=torch.device('cpu')):
        self.X_gen_init = X_gen_init.to(device)
        self.n_gen, self.d = X_gen_init.shape
        self.p_gen = p_gen.to(device)
        self.t = 0
        self.device=device

    def init_points(self, T):
        self.X_gen = torch.zeros(T,self.n_gen,self.d, device=self.device)
        self.X_gen[0,:,:] = self.X_gen_init
        self.mu=torch.mean(self.X_gen_init,axis=0)
        self.Cov=torch.cov(self.X_gen_init.T)
    
    def generate(self):
        return self.X_gen[self.t]

    # def update(self, t, lr_g, gradD):
    #     X = self.X_gen[t] + lr_g * gradD
    #     mu = torch.mean(X, axis=0)
    #     Cov = torch.cov(X.T)+1e-5*torch.eye(self.d)
    #     m = torch.distributions.MultivariateNormal(mu, Cov)
    #     self.X_gen[t+1] = m.sample((self.n_gen,))
    #     self.t=t+1

    def update(self, t, lr_g, gradD):
        proposed_X = self.X_gen[t] + lr_g * gradD
        delta_mu = torch.mean(proposed_X, axis=0)-torch.mean(self.X_gen[t], axis=0)
        delta_Cov = torch.eye(self.d)*1e-5 + torch.cov(proposed_X.T)-torch.cov(self.X_gen[t].T)
        self.mu+=delta_mu
        self.Cov+=delta_Cov
        m = torch.distributions.MultivariateNormal(self.mu, self.Cov)
        self.X_gen[t+1] = m.sample((self.n_gen,))
        self.t=t+1

class LinearGenerator():
    def __init__(self, d, Z_gen, p_gen, device=torch.device('cpu')):
        self.d = d
        self.n_gen, self.z_dim = Z_gen.shape
        self.Z_gen = Z_gen.to(device)
        self.ZZT = self.Z_gen@self.Z_gen.T
        self.W = (torch.rand(size=(self.z_dim, d))/(self.z_dim)).to(device)
        self.X_gen_init = self.Z_gen@self.W
        self.p_gen = p_gen.to(device)
        self.device=device

    def init_points(self, T):
        self.X_gen = torch.zeros(T,self.n_gen,self.d, device=self.device)
        self.X_gen[0,:,:] = self.X_gen_init

    def generate(self, Z=None):
        if Z==None:
            return self.Z_gen@self.W
        else:
            return Z@self.W

    def update(self, t, lr_g, gradx_D):
        self.X_gen[t+1] = self.X_gen[t] + lr_g * self.ZZT@gradx_D
        self.W = self.W - lr_g*self.Z_gen.T@gradx_D # TODO: not sure whether this sign is correct

class ReLUGenerator():
    def __init__(self, d, Z_gen, p_gen, device=torch.device('cpu')):
        self.d = d
        self.n_gen, self.z_dim = Z_gen.shape
        self.Z_gen = Z_gen.to(device)
        self.ZZT = self.Z_gen@self.Z_gen.T
        # normalizing so that generated points are approx 0-1 values
        self.W = (torch.rand(size=(self.z_dim, d))/(self.z_dim)).to(device)
        self.p_gen = p_gen.to(device)
        self.device=device

    def generate(self, Z=None):
        if Z==None:
            return self.Z_gen@self.W
        else:
            return Z@self.W

    def init_points(self, T):
        self.X_gen = torch.zeros(T,self.n_gen,self.d, device=self.device)
        self.X_gen[0,:,:] = self.generate()

    def update(self, t, lr_g, gradx_D):
        self.X_gen[t+1] = self.X_gen[t] + lr_g * self.ZZT@gradx_D
        self.W = self.W - lr_g*self.Z_gen.T@((self.Z_gen@self.W>0.0).float()*gradx_D) # TODO: not sure whether this sign is correct

class DCGenerator(nn.Module):
    def __init__(self, d, Z, p_gen, lr_g, device=torch.device('cpu')):
        super(DCGenerator, self).__init__()
        self.device=device
        self.d = d
        self.n_gen, self.z_dim = Z.shape
        self.Z = Z.to(device)
        self.p_gen = p_gen.to(device)
        self.lr_g=lr_g

        ngf = 64
        self.network = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        ).to(device)

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr_g)

    def generate(self, input=None, flatten=True):
        if input==None:
            input=self.Z
        out = self.network(input[:,:,None,None])
        
        return out.reshape(-1,784) if flatten else out

    def resample(self):
        # update latent Z
        self.Z = torch.normal(mean=torch.zeros(self.Z.shape)).to(self.device)

    def init_points(self, T):
        # create X_gen history tensor (stored on CPU)
        self.X_gen = torch.zeros(T, self.n_gen, self.d, device='cpu')
        self.X_gen[0,:,:] = self.generate().detach().clone().cpu()

    def update(self, t, gradx_D):
        # manually setting gradients dL/dx 
        # then backproping over the rest of the computation graph
        self.optimizer.zero_grad()
        X = self.network(self.Z[:,:,None,None]).reshape(-1,784)
        X.backward(-gradx_D)
        self.optimizer.step()
        self.X_gen[t+1,:,:] = self.generate().detach().clone().cpu()





