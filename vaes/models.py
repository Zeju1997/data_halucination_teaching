import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

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



class VAE(nn.Module):
  def __init__(self, device):
    super(VAE, self).__init__()

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