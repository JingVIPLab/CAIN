import sys

import torch.nn as nn
import torch.nn.functional as F
import torch, math


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


class SA(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super().__init__()
        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask=None):
        x = self.norm(x + self.dropout(self.mhatt(x, x, x, x_mask)))
        return x


class CA(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super().__init__()
        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, mask=None):
        x = self.norm(x + self.dropout(self.mhatt(y, y, x, mask)))
        return x


class FFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super().__init__()
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Decoder, self).__init__()

        self.SA = SA(hidden_dim, dropout_r, head)
        self.CA = CA(hidden_dim, dropout_r, head)
        self.FFN = FFN(hidden_dim, dropout_r)

        self.SA_five = SA(hidden_dim, dropout_r, head)
        self.CA_five = CA(hidden_dim, dropout_r, head)
        self.FFN_five = FFN(hidden_dim, dropout_r)

        self.SA_nine = SA(hidden_dim, dropout_r, head)
        self.CA_nine = CA(hidden_dim, dropout_r, head)
        self.FFN_nine = FFN(hidden_dim, dropout_r)

        self.a = nn.Parameter(torch.tensor(3.0))
        self.b = nn.Parameter(torch.tensor(3.0))

    def forward(self, img, que, img_mask, que_mask, five_mask, nine_mask):
        img_five = self.SA_five(img, five_mask) * torch.tanh(self.a)
        img_five = self.CA_five(img_five, que, que_mask)
        img_five = self.FFN_five(img_five)

        img_nine = self.SA_nine(img, nine_mask) * torch.tanh(self.b)
        img_nine = self.CA_nine(img_nine, que, que_mask)
        img_nine = self.FFN_nine(img_nine)

        img = self.SA(img, img_mask)
        img = (img + img_five + img_nine) / (1 + torch.tanh(self.a) + torch.tanh(self.b))
        img = self.CA(img, que, que_mask)
        img = self.FFN(img)

        return img


class CtxDecoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(CtxDecoder, self).__init__()
        self.SA = SA(hidden_dim, dropout_r, head)
        self.CA = CA(hidden_dim, dropout_r, head)
        self.FFN = FFN(hidden_dim, dropout_r)

    def forward(self, ctx, que):
        ctx = self.SA(ctx)
        ctx = self.CA(ctx, que)
        ctx = self.FFN(ctx)
        return ctx


class Transformer(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, avg_pool=True, k_captions=3):
        super(Transformer, self).__init__()
        self.avg_pool = avg_pool
        self.k = k_captions

        self.enc_list = nn.ModuleList([Encoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.dec_list = nn.ModuleList([Decoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.ctx_list = nn.ModuleList([CtxDecoder(hidden_dim, dropout_r, head) for _ in range(1)])

        if avg_pool:
            self.img_avgpool = nn.AdaptiveAvgPool1d(1)
            self.que_avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img, que, img_mask, que_mask, five_mask, nine_mask, ctx):
        b, n, c = img.shape

        for enc in self.enc_list:
            que = enc(que, que_mask)  # (30, 15, 768)

        for dec in self.dec_list:
            img = dec(img, que, img_mask, que_mask, five_mask, nine_mask)  # (30, 49, 768)

        if ctx.dim() == 4:
            ctx = torch.mean(ctx, dim=2)  # [80, 3, 640]
        for ctx_dec in self.ctx_list:
            ctx = ctx_dec(ctx, que)

        if self.avg_pool:
            img = self.img_avgpool(img.permute(0, 2, 1)).view(b, -1)
            que = self.que_avgpool(que.permute(0, 2, 1)).view(b, -1)

        return img, que, ctx
