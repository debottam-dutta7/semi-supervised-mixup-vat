import torch
import torch.nn.functional as F
from torch.autograd import Variable
#from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer

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
    dd = torch.randn(embedded.shape).float().cuda()
    dd = dd.transpose(0, 1).contiguous()
    dd = get_normalized_vector(dd).transpose(0, 1).contiguous()

    for ip in range(Ip):
        x_d = Variable(embedded.data + (small_constant_for_finite_diff * dd),
                       requires_grad=True)
        x_d.retain_grad()
        p_d_logit = model(batch, pre_embeddings=x_d) # batch is just for compatibility with implementation
        kl_loss = kl_categorical(Variable(p_logit.data), p_d_logit) 
        kl_loss.backward()
        dd = x_d.grad.data.transpose(0,1).contiguous()
        dd = get_normalized_vector(dd).transpose(0,1).contiguous()

    x_adv = embedded + (perturb_norm_length * Variable(dd))
    p_adv_logit = model(batch, pre_embeddings=x_adv)
    
    return kl_categorical(Variable(p_logit.data), p_adv_logit)



        
        




