import sys

import torch.nn as nn
import torch.nn.functional as F
import torch, math


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class AttFlat(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, glimpses=1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)

        self.linear_merge = nn.Linear(
            hidden_dim * glimpses,
            out_dim
        )

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -65504.0
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class Ctx_MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, up_dim=25, img_size=25, k_caption=5):
        super(Ctx_MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_r)

        # 新加的
        self.linear_c = nn.Linear(hidden_dim, hidden_dim)
        self.up_sample = nn.Linear(k_caption, up_dim, bias=False)
        # self.up_sample_conv = nn.Conv1d(k_caption, up_dim, kernel_size=3, stride=1, padding=1) 
        self.tau = nn.Parameter(torch.tensor(3.0))

        self.norm = nn.LayerNorm(img_size + 15)

    def forward(self, v, k, q, c, mask=None):
        b, n, s = q.shape

        c = self.linear_c(c).view(b, -1, self.head, self.head_size).transpose(1, 2)

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, c, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, c, mask):
        d_k = query.size(-1)
        b, head, d_c = c.size(0), c.size(1), c.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)

        c = self.up_sample(c.reshape(b * head, *c.shape[2:]).transpose(-2, -1)).transpose(-2, -1)
        # c = self.up_sample_conv(c.reshape(b * head, *c.shape[2:]))
        c = c.resize(b, head, *c.shape[1:])
        c_scores = torch.matmul(c, key.transpose(-2, -1))
        c_scores = c_scores / math.sqrt(d_c)
        c_scores = c_scores * torch.tanh(self.tau)

        scores += c_scores
        scores = scores / (1 + torch.tanh(self.tau))
        scores = self.norm(scores)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FCA(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, up_dim=25, img_size=25, k_caption=5):
        super().__init__()
        self.CA = Ctx_MHAtt(hidden_dim, dropout_r, head, up_dim, img_size, k_caption)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, ctx):
        # x做Q，[x y]做KV
        K = V = torch.cat([x, y], dim=1)
        Q = x
        att = self.norm(self.dropout(self.CA(K, V, Q, ctx)) + x)
        return att


class Fusion(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, img_size=25, avg_pool=True, k_caption=5):
        super(Fusion, self).__init__()
        self.avg_pool = avg_pool

        self.FCA_img = FCA(hidden_dim, dropout_r, head, img_size, img_size, k_caption)
        self.ffn_img = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.norm_img = nn.LayerNorm(hidden_dim)

        self.FCA_que = FCA(hidden_dim, dropout_r, head, 15, img_size, k_caption)
        self.ffn_que = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.norm_que = nn.LayerNorm(hidden_dim)

        self.norm_total = nn.LayerNorm(hidden_dim)

        self.gate = nn.Parameter(torch.tensor(3.0))

    def forward(self, que, img, ctx):
        b = img.shape[0]

        if ctx.dim() == 4:
            ctx = torch.mean(ctx, dim=2)  # [80, 3, 640]

        img_CA = self.FCA_img(img, que, ctx)
        img_ctx = self.norm_img(self.ffn_img(img_CA) + img_CA)

        que_CA = self.FCA_que(que, img, ctx)
        que_CA = self.norm_que(self.ffn_que(que_CA) + que_CA)

        img_ctx = torch.sum(img_ctx, dim=1)
        que_CA = torch.sum(que_CA, dim=1)

        img_ctx_que = self.norm_total((2 - torch.tanh(self.gate)) * img_ctx + torch.tanh(self.gate) * que_CA)

        # if self.avg_pool:
        #     img_ctx_que = self.multi_pool(img_ctx_que.permute(0, 2, 1)).view(b, -1)

        return img_ctx_que
