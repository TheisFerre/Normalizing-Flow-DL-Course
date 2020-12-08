import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class AffineCoupling(nn.Module):
    """
    Affine coupling layer described in "RealNVP" [Dinh et al, 2016]
    
    A mask is used to split the streams
    """
    def __init__(self, in_features, hidden_features=128, batchnorm=False):
        super(AffineCoupling, self).__init__()

        if batchnorm:
            self.scale = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.BatchNorm1d(num_features=hidden_features),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_features, in_features),
            )
            
            self.translate = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.BatchNorm1d(num_features=hidden_features),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_features, in_features)
            )
        else:
            self.scale = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, in_features),
            )
            
            self.translate = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_features, hidden_features),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_features, in_features)
            )
        
    def forward(self, x, mask=None):
        """
        Forward propagation (inference) y = f(x) = x1 + (x2 * s(x1) + t(x1))
        """ 
        x_ = mask * x

        s = self.scale(x_) * (1 - mask)
        t = self.translate(x_) * (1 - mask)

        y = x_ + (1 - mask) * (x * torch.exp(s) + t)
        jacobian = torch.sum(s, -1, keepdim=True)
        
        return y, jacobian

    def inverse(self, y, mask=None):
        """
        Inverse propagation (generation) x = f^(-1)(y) = y1 + (y2 - t(y1)) / s(y1))
        """
        y_ = mask * y

        s = self.scale(y_) * (1 - mask)
        t = self.translate(y_) * (1 - mask)

        x = y_ + (1 - mask) * ((y - t) * torch.exp(-s))
        jacobian = torch.sum(s, -1, keepdim=True)

        return x, jacobian

class RealNVP(nn.Module):
    """
    Real non-volume preserving network (RealNVP)
    is just a stack of affine coupling layers.
    """
    def __init__(self, in_features, prior=None, hidden_features=256, depth=6, batchnorm=False):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList([AffineCoupling(in_features, hidden_features, batchnorm) for _ in range(depth)])
        
        # A binary mask is used to make splitting of streams simple
        num_mask = int(in_features / 2)
        base_mask = torch.FloatTensor([[0, 1]]).repeat(1, num_mask)
        self.mask = nn.Parameter(base_mask.repeat((depth, 1)), requires_grad=False)
        
        # Alternate masking
        indices = [i for i in range(depth) if i%2]
        self.mask[indices] = 1 - self.mask[indices]
        if prior is None:
            self.prior = torch.distributions.Normal(torch.zeros(in_features).to(device), torch.ones(in_features).to(device))
        else:
            self.prior = prior
        
    def forward(self, x):
        """
        Forward propagation (inference) y = f(x)
        
        Gather the jacobian, log det(ab) = log det(a) + log det(b)
        """
        jacobian = torch.zeros_like(x)
        for mask, layer in zip(self.mask, self.layers):
            x, log_det_J = layer.forward(x, mask)
            jacobian += log_det_J
        
        return x, jacobian
    
    def inverse(self, z):
        """
        Inverse propagation (generation) x = f^(-1)(y)
        """
        jacobian = torch.zeros_like(z)
        # Run the operation in reverse
        for mask, layer in zip(reversed(self.mask), reversed(self.layers)):
            z, log_det_J = layer.inverse(z, mask)
            jacobian -= log_det_J

        return z, jacobian
    
    def log_likelihood(self, x):
        
        z_hat, log_det_jac = self.forward(x)
        
        log_likelihood_point = torch.mean(self.prior.log_prob(z_hat) + log_det_jac, dim=-1)
        
        return log_likelihood_point

    
class ConditionalAffineCoupling(nn.Module):
    """
    Affine coupling layer described in "RealNVP" [Dinh et al, 2016]
    
    A mask is used to split the streams
    """
    def __init__(self, in_features, context_dim, hidden_features=256):
        super(ConditionalAffineCoupling, self).__init__()

        self.scale = nn.Sequential(
            nn.Linear(in_features+context_dim, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, in_features),
        )
        
        self.translate = nn.Sequential(
            nn.Linear(in_features+context_dim, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, in_features)
        )
        
    def forward(self, x, cond, mask=None):
        """
        Forward propagation (inference) y = f(x) = x1 + (x2 * s(x1) + t(x1))
        """ 
        x_ = mask * x
        
        input_x = torch.cat([x_, cond], dim=-1)

        s = self.scale(input_x) * (1 - mask)
        t = self.translate(input_x) * (1 - mask)

        y = x_ + (1 - mask) * (x * torch.exp(s) + t)
        jacobian = torch.sum(s, -1, keepdim=True)
        
        return y, jacobian

    def inverse(self, z, cond, mask=None):
        """
        Inverse propagation (generation) x = f^(-1)(y) = y1 + (y2 - t(y1)) / s(y1))
        """
        z_ = mask * z
        
        input_z = torch.cat([z_, cond], dim=-1)

        s = self.scale(input_z) * (1 - mask)
        t = self.translate(input_z) * (1 - mask)

        x = y_ + (1 - mask) * ((z - t) * torch.exp(-s))
        jacobian = torch.sum(s, -1, keepdim=True)

        return x, jacobian

    
class ConditionalRealNVP(nn.Module):
    """
    Real non-volume preserving network (RealNVP)
    is just a stack of affine coupling layers.
    """
    def __init__(self, in_features, context_dim, prior=None, hidden_features=256, depth=6):
        super(ConditionalRealNVP, self).__init__()
        self.layers = nn.ModuleList([ConditionalAffineCoupling(
            in_features, 
            context_dim, 
            hidden_features) for _ in range(depth)])
        
        # A binary mask is used to make splitting of streams simple
        base_mask = torch.FloatTensor([[0, 1]])
        self.mask = nn.Parameter(base_mask.repeat((depth, 1)), requires_grad=False)
        
        # Alternate masking
        indices = [i for i in range(depth) if i%2]
        self.mask[indices] = 1 - self.mask[indices]
        if prior is None:
            self.prior = PriorConditioner(input_dim=context_dim, hidden_dim=256, output_dim=in_features)
        else:
            self.prior = prior
        
    def forward(self, x, cond):
        """
        Forward propagation (inference) y = f(x)
        
        Gather the jacobian, log det(ab) = log det(a) + log det(b)
        """
        jacobian = torch.zeros_like(x)
        for mask, layer in zip(self.mask, self.layers):
            x, log_det_J = layer.forward(x, cond, mask)
            jacobian += log_det_J
        
        return x, jacobian
    
    def inverse(self, z, cond):
        """
        Inverse propagation (generation) x = f^(-1)(y)
        """
        jacobian = torch.zeros_like(z)
        # Run the operation in reverse
        for mask, layer in zip(reversed(self.mask), reversed(self.layers)):
            z, log_det_J = layer.inverse(z, cond, mask)
            jacobian -= log_det_J

        return z, jacobian
    
    def log_likelihood(self, x, cond):
        
        input_x = torch.cat([x, cond], dim=-1)
        
        z_hat, log_det_jac = self.forward(x, cond)
        
        prior_loc, prior_scale = self.prior(cond)
        
        dist = Normal(prior_loc, prior_scale)
        
        log_likelihood_point = torch.mean(dist.log_prob(z_hat) + log_det_jac, dim=-1)
        
        return log_likelihood_point

    
class PriorConditioner(nn.Module):
    """
    This PyTorch Module implements the neural network used to condition the
    base distribution of a NF as in Eq.(13).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PriorConditioner, self).__init__()

        #initialize linear transformations
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)

        #initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Given input x=[z_{t-1}, h_t], this method outputs mean and 
        std.dev of the diagonal gaussian base distribution of a NF.
        """
        hidden = self.relu(self.lin_input_to_hidden(x))
        hidden = self.relu(self.lin_hidden_to_hidden(hidden))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden))
        return loc, scale
    

    
###### TAKEN FROM https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py ######
class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh).to(device)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh).to(device)
        
    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.zeros((x.shape[0], self.dim))  
        z[:,::2] = z0
        z[:,1::2] = z1
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.zeros((z.shape[0], self.dim))  
        x[:,::2] = x0
        x[:,1::2] = x1
        log_det = torch.sum(-s, dim=1)
        return x, log_det

    
class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
        
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        loc, scale = self.prior(x[:, -self.prior.input_dim:])
        dist = Normal(loc, scale)
        z_last = zs[-1]
        prior_logprob = dist.log_prob(z_last[:, 0:self.prior.output_dim]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples, condition):
        loc, scale = self.prior(condition)
        dist = Normal(loc, scale)
        z = torch.squeeze(dist.sample((num_samples,)))
        
        cond = torch.squeeze(condition).repeat(num_samples).reshape(-1, self.prior.output_dim)
        z_back = torch.cat([z, cond], dim=1)
        xs, _ = self.flow.backward(z_back)
        return xs
    
    def log_likelihood(self, x):
        zs, prior_logprob, log_det = self.forward(x)
        
        return prior_logprob + log_det

class PriorConditioner(nn.Module):
    """
    This PyTorch Module implements the neural network used to condition the
    base distribution of a NF as in Eq.(13).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PriorConditioner, self).__init__()

        #initialize linear transformations
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)

        #initialize non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Given input x=[z_{t-1}, h_t], this method outputs mean and 
        std.dev of the diagonal gaussian base distribution of a NF.
        """
        hidden = self.relu(self.lin_input_to_hidden(x))
        hidden = self.relu(self.lin_hidden_to_hidden(hidden))
        loc = self.lin_hidden_to_loc(hidden)
        scale = self.softplus(self.lin_hidden_to_scale(hidden))
        return loc, scale

    
def Z_values(func, bounds, mu_lng, sig_lng, mu_lat, sig_lat, cond=None):
    """
    Args:
        func: Log Likelihood hood for model of interest
        bounds: Tuple of bounds (minx, maxx, miny, maxy)
        mu_lng: mu value of unnormalized longitude values
        sig_lng: std value of unnormalized longitude values
        mu_lat: mu value of unnormalized latitude values
        sig_lat: std value of unnormalized longitude values
        cond: Conditioned data. Only applicable for Conditional Normalizing Flow
    
    Returns:
        X, Y, Z: (X, Y) coordinates with corresponding Negative Log-Likelihood of model.
    """
    minx = bounds[0]
    maxx = bounds[1]
    miny = bounds[2]
    maxy = bounds[3]
    x_linspace = np.linspace(minx, maxx, num=150)
    y_linspace = np.linspace(miny, maxy, num=150)
    
    X, Y = np.meshgrid(x_linspace, y_linspace)
    X_stand, Y_stand = (X-mu_lng)/sig_lng, (Y-mu_lat)/sig_lat
    
    XX = np.array([X_stand.ravel(), Y_stand.ravel()]).T

    XX = torch.Tensor(XX)
    
    if cond is not None:
        cond_len = len(cond)
        cond = cond.repeat(len(XX)).reshape(-1, cond_len)

        Z = torch.cat([XX, cond], dim=1)
    else:
        Z = XX
    
    Z = -func(Z)
    Z = Z.reshape(X.shape)
    Z = Z.detach().cpu().clone().numpy()
    
    return X, Y, Z


def contour_plot_map(mapplot, X, Y, Z, **kwargs):
    """
    Args:
        mapplot: Geopandas ploting object
        X: Numpy.Array Matrix with shape (N, N)
        Y: Numpy.Array Matrix with shape (N, N)
        Z: Numpy.Array Matrix with shape (N, N). Contains Log Likelihood
        **kwargs: Additional Keyword Arguments for plotting the contours.
    
    Returns:
        GeoPandas plotting object. Can be used to further add plots
    """
    CS = mapplot.contourf(
        X, 
        Y, 
        Z, 
        **kwargs
    )
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    return mapplot
    

    
    
    

class NormalizingFlow_RAW(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel_RAW(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
    
    def log_likelihood(self, x):
        zs, prior_logprob, log_det = self.forward(x)
        
        return prior_logprob + log_det
        