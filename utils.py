import ot
from tqdm import tqdm
import torch
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def wasserstein2(Xhist, Xtrue, log_interval, device=torch.device('cpu')):
    '''
    inp: (t x n x d) point trajectories, (n' x d) targets
    out: t-vector of wasserstein distances
    '''
    T, ngen = Xhist.shape[:2]
    ntrue = Xtrue.shape[0]
    Wdist = torch.zeros(T//log_interval)
    # assume uniform distribution on samples
    a, b = torch.ones(ngen, device=device)/ngen, torch.ones(ntrue, device=device)/ntrue
#     print('Calculating Wass2 History')
    for i, sample_idx in enumerate(tqdm(range(0, T, log_interval), miniters=5)):
            # loss matrix M
            M = ot.dist(Xhist[sample_idx,:,:], Xtrue)
            EMD = ot.emd(a,b,M)
            Wdist[i] = torch.sum(EMD*M)
    return Wdist

# Adapted from Matplotlib documentation
def confidence_ellipse(x, y, ax, n_std=3.0, cov=None, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if cov is None:
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)