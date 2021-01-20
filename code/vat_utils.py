import torch
import torch.nn.functional as F
from torch.nn import Variable


def get_normalized_vector(d):
    B, T, D = d.shape
    d = d.view(B, -1)
    d /= (1e-12 + torch.max(torch.abs(d), dim=1, keepdim=True)[0])
    # d /= (1e-12 + torch.max(torch.abs(d), dim=1, keepdim=True)[0])

    d /= torch.sqrt(1e-6 + torch.sum(d**2, dim=1, keepdim=True))
    # d /= torch.sqrt(1e-6 + torch.sum(d**2, dim=1, keepdim=True))
    d = d.view(B, T, D)
    return d

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) -
                         F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl) # F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def vat_loss(embedder, model, batch, perturb_norm_length=5.0,
            small_constant_for_finite_diff=1e-1, Ip=1, p_logit=None):
    
    embedded = embedder(batch)
    d = torch.randn(embedded.shape).float()
    d = d.transpose(0, 1).contiguous()
    d = get_normalized_vector(d).transpose(0, 1).contiguous()

    for ip in range(Ip):
        x_d = Variable(embedded.data + (small_constant_for_finite_diff * d),
                       requires_grad=True)
        x_d.retain_grad()

        p_d_logit = model(x_d, initial_embed=True)
        




