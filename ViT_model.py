import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
def kron(A, B):
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)
    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#         self.dim_head = dim_head
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_q = nn.Parameter(torch.randn(dim, inner_dim))
#         self.to_k = nn.Parameter(torch.randn(dim, inner_dim))
#         self.to_v = nn.Parameter(torch.randn(dim, inner_dim))
#         self.projection = nn.Parameter(torch.randn(inner_dim, dim))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         q,k,v = x @ self.to_q, x @ self.to_k, x @ self.to_v
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)  
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = out @ self.projection
#         out = self.dropout(out)
#         return out
#     def get_approx(self, x):
#         Qmat = (x @ self.to_q.detach())  
#         Kmat = (x @ self.to_k.detach())
        
        
#         Qmat, Kmat = map(lambda t: rearrange(t, 'b d (h dk) -> b h d dk', h = self.heads), (Qmat, Kmat))

#         Amat = self.attend((self.dim_head**-0.5) * torch.matmul(Qmat, Kmat.transpose(-1,-2)))
#         # print(Amat.shape)
#         # print(self.projection.shape)
#         projs = rearrange(self.projection.detach(), '(h dk) d -> h dk d', h = self.heads)
#         Vmat = self.to_v.detach()
#         Vmat = rearrange(Vmat, 'd (h dk) -> h d dk', h = self.heads)
#         Bmat = torch.matmul(Vmat, projs)
#         # Bmat = Bmat.transpose(-1,-2)
#         Amat = torch.mean(Amat, dim = 0)
#         Amat = Amat.chunk(self.heads, dim = 0)
#         Bmat = Bmat.chunk(self.heads, dim = 0)
#         mat = kron(Bmat[0].squeeze().T, Amat[0].squeeze())
#         sum = kron(Bmat[0].squeeze().T, Amat[0].squeeze())
#         krons = [mat]
#         for head in range(1, self.heads):
#             mat = kron(Bmat[head].squeeze().T, Amat[head].squeeze())
#             krons.append(mat)
#             sum += kron(Bmat[head].squeeze().T, Amat[head].squeeze())
#         # print('Kron trace  ', torch.trace(sum))
#         krons.append(sum)
#         return krons
    
#Now we define an attention class in which x_q has a different random shuffle for each head:


class MHRSQ(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        shuffles = []
        for head in range(heads):
            perm = random.sample(range(l*dim_head), (l)*dim_head)
            TP = torch.tensor(perm, requires_grad = False)
            self.register_buffer(f'perm_{head}', TP)
            shuffles.append(TP)
        self.shuffles  = shuffles

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        flat_q = q.flatten(start_dim = 2)
        for head in range(self.heads):
            flat_q[:, head] = flat_q[:, head, self.shuffles[head]]
        shuffled_q = flat_q.view(q.size())
        dots = torch.matmul(shuffled_q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Q_shuffle2(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        
        perm = random.sample(range(l*dim_head), (l)*dim_head)
        self.TP = torch.tensor(perm, requires_grad = False)
        self.register_buffer(f'perm', self.TP)
        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        flat_q = q.flatten(start_dim = 2)
        flat_q = flat_q[:, :, self.TP]
        shuffled_q = flat_q.view(q.size())
        dots = torch.matmul(shuffled_q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class K_shuffle2(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        
        perm = random.sample(range(l*dim_head), (l)*dim_head)
        self.TP = torch.tensor(perm, requires_grad = False)
        self.register_buffer(f'perm', self.TP)
        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        flat_k = k.flatten(start_dim = 2)
        flat_k = flat_k[:, :, self.TP]
        shuffled_k = flat_k.view(k.size())
        dots = torch.matmul(q, shuffled_k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class V_shuffle2(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        
        perm = random.sample(range(l*dim_head), (l)*dim_head)
        self.TP = torch.tensor(perm, requires_grad = False)
        self.register_buffer(f'perm', self.TP)
        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        flat_v = v.flatten(start_dim = 2)
        flat_v = flat_v[:, :, self.TP]
        shuffled_v = flat_v.view(v.size())
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, shuffled_v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)





class Multi_Head_Q_Shuffle(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.W_qs = nn.ModuleList([nn.Linear(dim, dim_head, bias = False) for head in range(heads)])
        self.W_ks = nn.ModuleList([nn.Linear(dim, dim_head, bias = False) for head in range(heads)])
        self.W_vs = nn.ModuleList([nn.Linear(dim, dim_head, bias = False) for head in range(heads)])
        shuffles = []
        for head in range(heads-1):
            perm = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
            TP = torch.tensor(perm, requires_grad = False)
            self.register_buffer(f'perm_{head}', TP)
            shuffles.append(TP)
        self.shuffles  = shuffles
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 
    def forward(self, x):
        qs = []
        ks = []
        vs = []
        for head in range(self.heads):
            ks.append(self.W_ks[head](x))
            vs.append(self.W_vs[head](x))
            if head == 0:
                qs.append(self.W_qs[head](x))
            else:
                flat_x = x.flatten(start_dim=1)
                flat_x_shuffled = flat_x[:, self.shuffles[head-1]]
                shuffled_x = flat_x_shuffled.view(x.size())
                qs.append(self.W_qs[head](shuffled_x))
        mats = []
        for head in range(self.heads):
            mat = torch.matmul(qs[head], ks[head].transpose(-1, -2)) * self.scale
            mat = self.attend(mat)
            mat = torch.matmul(mat, vs[head])
            mats.append(mat)
        out = torch.cat(mats, dim = -1)
        out = self.to_out(out)
        return out






        
class All_Same_Shuffle(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        perm = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        self.perm = torch.tensor(perm, requires_grad = False)

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_x_shuffled = flat_x[:, self.perm]

        shuffled_x = flat_x_shuffled.view(x.size())
        
        qkv = self.to_qkv(shuffled_x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransposedAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(4, dim = -1)
        q, k, v, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q.transpose(-1,-2), k) * self.scale

        attn = self.attend(dots)
        
        v = v.transpose(-1,-2)

        v = torch.matmul(v, v2)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    
class QuadraticAttention(nn.Module):
    def __init__(self, dim, l,  heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        #inner_dim = dim_head *  heads
        #project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.middle_matrix = nn.Linear(dim, l, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        out = torch.matmul(self.middle_matrix(x), x)
        return out

class AllRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm2 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm3 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permk = torch.tensor(perm2, requires_grad = False)
        self.permv = torch.tensor(perm3, requires_grad = False)
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xk = flat_x[:, self.permk]
        flat_xv = flat_x[:, self.permv]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xk = flat_xk.view(x.size())
        shuffled_xv = flat_xv.view(x.size())


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class QKRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm2 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permk = torch.tensor(perm2, requires_grad = False)
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xk = flat_x[:, self.permk]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xk = flat_xk.view(x.size())


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class QRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permq = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]


        shuffled_xq = flat_xq.view(x.size())
    


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class KRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permk = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xk = flat_x[:, self.permk]


        shuffled_xk = flat_xk.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class VRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xv = flat_x[:, self.permv]


        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class KVRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permk = torch.tensor(perm1, requires_grad = False)
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xk = flat_x[:, self.permk]
        flat_xv = flat_x[:, self.permv]


        shuffled_xk = flat_xk.view(x.size())
        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class QVRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xv = flat_x[:, self.permv]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class ShuffleRowAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        perm = [0] + random.sample(range(1, l), l-1)
        self.perm = torch.tensor(perm)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = q[:,:, self.perm]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class QuarticAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 5, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(5, dim = -1)
        q1, k1, q2, k2, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) 
        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) 
        dots = torch.matmul(dots1, dots2)
        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class NoSoftMaxAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) 

        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        


class Simple_Kron(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.feature_mixer1 = nn.Linear(dim, dim)
        self.token_mixer1 = nn.Linear(num_patches+1, num_patches+1)
        self.activation = nn.GELU()
    def forward(self, x):
        x = self.token_mixer1(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.feature_mixer1(x)
        x = self.activation(x)
        return x




# class Simple_Kron_Mixer(nn.Module):
#     def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head
#         # project_out = not (heads == 1 and dim_head == dim)
#         self.heads = heads
#         self.A_list = nn.ParameterList([nn.Parameter(torch.randn(num_patches, num_patches)) for head in range(self.heads)])
#         self.B_list = nn.ParameterList([nn.Parameter(torch.randn(dim, inner_dim)) for head in range(self.heads)])

#     def forward(self, x):
#         out = self.A_list[0] @ x @ self.B_list[0]
#         for head in range(1,self.heads):
#             out += self.A_list[head] @ x @ self.B_list[head]
#         return out
#     def get_approx(self):
#         sum = kron(self.B_list[0].detach().T, self.A_list[0].detach())
#         for head in range(1,self.heads):
#             sum += kron(self.B_list[head].detach().T, self.A_list[head].detach())
        # return sum

class Transformer(nn.Module):
    def __init__(self, attention_type, dim, depth, heads, dim_head, mlp_dim, num_patches, fixed_size, dropout = 0.):
        super().__init__()
        #def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        self.layers = nn.ModuleList([])
        self.type = None
        if attention_type == 'mhrsq':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, MHRSQ(dim, l = num_patches+1,heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'mixer':
            self.type = 'mixer'
            for _ in range(depth):
                self.layers.append(
                    MixerLayer(num_patches = num_patches+1, hidden_size = dim, hidden_s = num_patches+1,hidden_c = 512 , drop_p = dropout, off_act = False)
                )
        elif attention_type == 'kron':
            self.type = 'mixer'
            for _ in range(depth):
                self.layers.append(
                    PreNorm(dim, Simple_Kron(dim = dim, num_patches = num_patches, )),
                    #PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                )
        elif attention_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'quartic':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QuarticAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'no_softmax':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, NoSoftMaxAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'row_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, ShuffleRowAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'allrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, AllRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qkrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QKRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Q_shuffle2(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'quadratic':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QuadraticAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'krandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, K_shuffle2(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'vrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, V_shuffle2(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'kvrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, KVRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qvrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QVRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'transposed':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'all_same_shuffle':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, All_Same_Shuffle(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'multi_head_q':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Multi_Head_Q_Shuffle(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))




    def forward(self, x):
        if self.type == 'mixer':
            for layer in self.layers:
                x = layer(x)
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, attention_type, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., fixed_size = False, pre_layers = 2):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.fixed_size = fixed_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(attention_type = attention_type, dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim,
                                        dropout = dropout, num_patches = num_patches, fixed_size = self.fixed_size)
        
        self.pool = pool
        #self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
            
        self.atn_type = attention_type

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(64 + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #x = self.to_latent(x)
        return self.mlp_head(x)
