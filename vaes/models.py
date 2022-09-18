import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

def sample_from_discretized_mix_logistic(l):
    """
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    def to_one_hot(tensor, n):
        one_hot = torch.zeros(tensor.size() + (n,))
        one_hot = one_hot.to(tensor.device)
        one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.)
        return one_hot

    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel,
                                       dim=4),
                             min=-7.)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel,
                       dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = nn.Parameter(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0,
                                 min=-1.),
                     max=1.)
    x2 = torch.clamp(torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 +
                                 coeffs[:, :, :, 2] * x1,
                                 min=-1.),
                     max=1.)

    out = torch.cat([
        x0.view(xs[:-1] + [1]),
        x1.view(xs[:-1] + [1]),
        x2.view(xs[:-1] + [1])
    ],
                    dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def discretized_mix_logistic_loss(x, l):
    """
    log-likelihood for mixture of discretized logistics, assumes the data
    has been rescaled to [-1,1] interval
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    # channels last
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)

    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [int(y) for y in x.size()]
    # predicted distribution, e.g. (B,32,32,100)
    ls = [int(y) for y in l.size()]


    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(
        xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + nn.Parameter(torch.zeros(xs + [nr_mix]).to(x.device),
                                       requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(
        xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(
              xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below
    # for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which
    # never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero
    # instead of selecting: this requires use to use some ugly tricks to avoid
    # potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as
    # output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (
        1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + torch.log_softmax(logit_probs,
                                                                dim=-1)
    log_probs = torch.logsumexp(log_probs, dim=-1)

    # return -torch.sum(log_probs)
    loss_sep = -log_probs.sum((1, 2))  # keep batch dimension
    return loss_sep


class SimpleHalMoonNN(nn.Module):
    """ Simple NN classifier for HalfMoon
    """

    def __init__(self, x_dim=2, y_dim=2, h_dim=[8, 16, 8, 4]):
        super(SimpleHalMoonNN, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        h_neurons = [x_dim, *h_dim]

        h_linear_layers = [nn.Linear(h_neurons[i - 1], h_neurons[i]) for i in range(1, len(h_neurons))]

        self.cls_hidden = nn.ModuleList(h_linear_layers)
        self.logits = nn.Linear(h_dim[-1], y_dim)

    def forward(self, x, logits=True):
        for layer in self.cls_hidden:
            x = F.relu(layer(x))
        lg = self.logits(x)
        if logits:
            return lg
        logprobs = F.log_softmax(lg, dim=-1)
        return logprobs



class VAE_HalfMoon(nn.Module):
  def __init__(self, device):
    super(VAE_HalfMoon, self).__init__()

    self.device = device
    self.x_dims = 2
    self.z_dims = 20
    self.y_dims = 2
    self.px_sigma = 0.08

    # Layers for q(z|x,y):
    self.qz_fc = nn.Sequential(
                    nn.Linear(in_features=self.x_dims+self.y_dims, out_features=128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )

    self.qz_mu = nn.Linear(in_features=128, out_features=self.z_dims)
    self.qz_pre_sp = nn.Linear(in_features=128, out_features=self.z_dims)

    # Layers for p(x,y|z):
    self.pxy_fc = nn.Sequential(
                    nn.Linear(in_features=self.z_dims, out_features=128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
    self.px_mu = nn.Linear(in_features=128, out_features=self.x_dims)
    # self.px_pre_sp = nn.Linear(in_features=128, out_features=self.x_dims)
    self.py_logits = nn.Linear(in_features=128, out_features=self.y_dims)

  def q_z(self, x, y):
    # y is one_hot
    h = torch.cat((x.view(x.size(0), -1), y), dim=-1)
    h = self.qz_fc(h)
    z_mu = self.qz_mu(h)
    z_pre_sp = self.qz_pre_sp(h)
    z_std = F.softplus(z_pre_sp)
    return self.reparameterize(z_mu, z_std), z_mu, z_std

  def p_xy(self, z):
    h = self.pxy_fc(z)
    x_mu = self.px_mu(h)
    # x_pre_sp = self.px_pre_sp(h)
    # x_std = F.softplus(x_pre_sp)
    x_std = torch.ones_like(x_mu) * self.px_sigma
    y_logit = self.py_logits(h)
    return self.reparameterize(x_mu, x_std), x_mu, x_std, y_logit

  def reparameterize(self, mu, std):
    eps = torch.randn(mu.size())
    eps = eps.to(self.device)

    return mu + eps * std

  def sample(self, num=10):
    # sample latent vectors from the normal distribution
    z = torch.randn(num, self.z_dims)
    z = z.to(self.device)

    x, x_mu, x_std, y_logit = self.p_xy(z)

    return x, y_logit

  def reconstruction(self, x, y):
    z, _, _ = self.q_z(x, y)
    x_hat, x_mu, x_std, y_logit = self.p_xy(z)

    return x_hat, y_logit

  def forward(self, x, y):
    elbo = 0
    z, qz_mu, qz_std = self.q_z(x, y)

    x_hat, px_mu, px_std, py_logit = self.p_xy(z)

    # For likelihood : <log p(x|z)>_q :
    px = D.normal.Normal(px_mu, px_std)
    px = D.independent.Independent(px, 1)
    elbo += px.log_prob(x)
    # For likelihood : <log p(y|z)>_q :
    py = D.categorical.Categorical(logits=py_logit)
    elbo += py.log_prob(torch.argmax(y, dim=1))

    qz = D.normal.Normal(qz_mu, qz_std)
    qz = D.independent.Independent(qz, 1)
    pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
    pz = D.independent.Independent(pz, 1)

    # For: - KL[qz || pz]
    elbo -= D.kl.kl_divergence(qz, pz)

    # # For : <log p(z)>_q
    # elbo += pz.log_prob(z)

    # # For : -<log q(z|x,y)>_q
    # elbo -= qz.log_prob(z)

    return -elbo.mean(), elbo


class VAE_MNIST(nn.Module):
  def __init__(self, device):
    super(VAE_MNIST, self).__init__()

    self.device = device
    self.c = 16
    self.z_dims = 16
    self.y_dim = 2
    self.x_h = 28
    self.x_w = 28
    # self.px_sigma = 0.4
    self.n_components = 10

    # Layers for q(z|x,y):
    self.qz_conv1 = nn.Conv2d(in_channels=1+self.y_dim, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
    self.qz_conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
    self.qz_mu = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)
    self.qz_pre_sp = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)

    # Layers for p(x,y|z):
    self.px_l1 = nn.Linear(in_features=self.z_dims, out_features=self.c*2*7*7)
    self.px_conv1 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
    self.px_mol = nn.ConvTranspose2d(in_channels=self.c, out_channels=10 * self.n_components, kernel_size=4, stride=2, padding=1)
    # self.px_mu = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
    # self.px_pre_sp = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
    self.py_logits = nn.Linear(in_features=3136, out_features=self.y_dim)

  def q_z(self, x, y):
    # y is binary, add y as a channel
    y = y.view(-1, self.y_dim, 1, 1).repeat(1, 1, self.x_h, self.x_w)
    h = torch.cat([x, y], dim=1)
    h = F.relu(self.qz_conv1(h))
    h = F.relu(self.qz_conv2(h))
    h = h.view(h.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
    z_mu = self.qz_mu(h)
    z_pre_sp = self.qz_pre_sp(h)
    z_std = F.softplus(z_pre_sp)
    return self.reparameterize(z_mu, z_std), z_mu, z_std

  def p_xy(self, z):
    h = self.px_l1(z)
    h = h.view(h.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
    h = F.relu(self.px_conv1(h))
    y_logit = self.py_logits(h.view(h.size(0), -1))
    # x_mu = self.px_mu(h)
    # x_pre_sp = self.px_pre_sp(h)
    # x_std = F.softplus(x_pre_sp)
    # x_std = torch.ones_like(x_mu) * self.px_sigma
    # return self.reparameterize(x_mu, x_std), x_mu, x_std, y_logit

    x_mol = self.px_mol(h)

    x_repara = sample_from_discretized_mix_logistic(x_mol)
    x_repara = (x_repara + 1) / 2
    x_repara = x_repara.clamp(min=0., max=1.)

    return x_repara, x_mol, y_logit

  def reparameterize(self, mu, std):
    eps = torch.randn(mu.size())
    eps = eps.to(self.device)

    return mu + eps * std

  def sample(self, num=10):
    # sample latent vectors from the normal distribution
    z = torch.randn(num, self.z_dims)
    z = z.to(self.device)

    # x, x_mu, x_std, y_logit = self.p_xy(z)
    x, x_mol, y_logit = self.p_xy(z)

    return x, y_logit

  def reconstruction(self, x, y):
    z, _, _ = self.q_z(x, y)
    # x_hat, x_mu, x_std, y_logit = self.p_xy(z)
    x_hat, x_mol, y_logit = self.p_xy(z)

    return x_hat, y_logit

  def loglikelihood_MoL(self, x, x_mol):
    x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
    ll = -discretized_mix_logistic_loss(x, x_mol)
    return ll

  def forward(self, x, y):

    elbo = 0
    z, qz_mu, qz_std = self.q_z(x, y)

    # x_hat, px_mu, px_std, py_logit = self.p_xy(z)
    x_hat, px_mol, py_logit = self.p_xy(z)
    elbo += self.loglikelihood_MoL(x, px_mol)

    # # For likelihood : <log p(x|z)>_q :
    # px = D.normal.Normal(px_mu, px_std)
    # px = D.independent.Independent(px, 3)
    # elbo += px.log_prob(x)

    # For likelihood : <log p(y|z)>_q :
    py = D.categorical.Categorical(logits=py_logit)
    elbo += py.log_prob(torch.argmax(y, dim=1))

    qz = D.normal.Normal(qz_mu, qz_std)
    qz = D.independent.Independent(qz, 1)
    pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
    pz = D.independent.Independent(pz, 1)

    # For: - KL[qz || pz]
    elbo -= D.kl.kl_divergence(qz, pz)

    # # For : <log p(z)>_q
    # elbo += pz.log_prob(z)

    # # For : -<log q(z|x,y)>_q
    # elbo -= qz.log_prob(z)

    return -elbo.mean(), elbo



class VAE_bMNIST(nn.Module):
  def __init__(self, device):
    super(VAE_bMNIST, self).__init__()

    self.device = device
    self.c = 16
    self.z_dims = 16
    self.y_dim = 2
    self.x_h = 28
    self.x_w = 28

    # Layers for q(z|x,y):
    self.qz_conv = nn.Sequential(
                        nn.Conv2d(in_channels=1+self.y_dim, out_channels=self.c, kernel_size=4, stride=2, padding=1), # out: c x 14 x 14
                        nn.ReLU(),
                        nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1), # out: c x 7 x 7
                        nn.ReLU()
                    )
    self.qz_mu = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)
    self.qz_pre_sp = nn.Linear(in_features=self.c*2*7*7, out_features=self.z_dims)

    # Layers for p(x,y|z):
    self.px_l1 = nn.Linear(in_features=self.z_dims, out_features=self.c*2*7*7)
    self.px_conv1 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
    self.px_bern = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
    self.py_logits = nn.Linear(in_features=3136, out_features=self.y_dim)

  def q_z(self, x, y):
    # y is binary, add y as a channel
    y = y.view(-1, self.y_dim, 1, 1).repeat(1, 1, self.x_h, self.x_w)
    h = torch.cat([x, y], dim=1)
    h = self.qz_conv(h)
    h = h.view(h.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
    z_mu = self.qz_mu(h)
    z_pre_sp = self.qz_pre_sp(h)
    z_std = F.softplus(z_pre_sp)
    return self.reparameterize(z_mu, z_std), z_mu, z_std

  def p_xy(self, z):
    h = self.px_l1(z)
    h = h.view(h.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
    h = F.relu(self.px_conv1(h))
    y_logit = self.py_logits(h.view(h.size(0), -1))
    x_bern = torch.sigmoid(self.px_bern(h))
    return x_bern, y_logit

  def reparameterize(self, mu, std):
    eps = torch.randn(mu.size())
    eps = eps.to(self.device)

    return mu + eps * std

  def sample(self, num=10):
    # sample latent vectors from the normal distribution
    z = torch.randn(num, self.z_dims)
    z = z.to(self.device)

    x_bern, y_logit = self.p_xy(z)

    return x_bern, y_logit

  def reconstruction(self, x, y):
    z, _, _ = self.q_z(x, y)
    # x_hat, x_mu, x_std, y_logit = self.p_xy(z)
    x_bern, y_logit = self.p_xy(z)

    return x_bern, y_logit

  def loglikelihood_MoL(self, x, x_mol):
    x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
    ll = -discretized_mix_logistic_loss(x, x_mol)
    return ll

  def forward(self, x, y):

    elbo = 0
    z, qz_mu, qz_std = self.q_z(x, y)

    x_bern, py_logit = self.p_xy(z)

    # # For likelihood : <log p(x|z)>_q :
    elbo += torch.sum(torch.flatten(x * torch.log(x_bern + 1e-8)
                                + (1 - x) * torch.log(1 - x_bern + 1e-8),
                                start_dim=1),
                        dim=-1)

    # For likelihood : <log p(y|z)>_q :
    py = D.categorical.Categorical(logits=py_logit)
    elbo += py.log_prob(torch.argmax(y, dim=1))

    qz = D.normal.Normal(qz_mu, qz_std)
    qz = D.independent.Independent(qz, 1)
    pz = D.normal.Normal(torch.zeros_like(z), torch.ones_like(z))
    pz = D.independent.Independent(pz, 1)

    # For: - KL[qz || pz]
    elbo -= D.kl.kl_divergence(qz, pz)

    # # For : <log p(z)>_q
    # elbo += pz.log_prob(z)

    # # For : -<log q(z|x,y)>_q
    # elbo -= qz.log_prob(z)

    return -elbo.mean(), elbo
    