

import torch.nn as nn
import torch.nn.functional as F
import torch, math

from einops import rearrange, repeat

class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim,hidden_dim)
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
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, out_dim)

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

class AttProto(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, seq_len=15, glimpses=1):
        super(AttProto, self).__init__()
        self.glimpses = glimpses

        # self.token_weight = nn.Parameter(torch.randn(1, seq_len, self.glimpses))
        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)

        self.linear_merge = nn.Linear(
            hidden_dim * glimpses,
            out_dim
        )

    def forward(self, x, x_mask=None):
        b, n, c = x.shape
        # weight = repeat(self.token_weight, '() n d -> b n d', b = b)
        # att = torch.add(self.mlp(x), weight)
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

        return x_atted.unsqueeze(1)


class GuideProto(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, seq_len=49, guide_len=15, glimpses=1):
        super(GuideProto, self).__init__()
        self.glimpses = glimpses

        self.token_weight = nn.Parameter(torch.randn(1, seq_len, self.glimpses))
        self.linear_y = nn.Linear(hidden_dim, seq_len)
        self.linear_yp = nn.Linear(guide_len, self.glimpses)
        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)

        self.linear_merge = nn.Linear(
            hidden_dim * glimpses,
            out_dim
        )

    def forward(self, x, y, x_mask=None):
        b, n, c = x.shape
        # weight = repeat(self.token_weight, '() n d -> b n d', b = b)
        att_y = self.linear_yp(self.linear_y(y).transpose(1,2))
        
        # att = torch.add(self.mlp(x), att_y + weight)
        att = torch.add(self.mlp(x), att_y)
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

        return x_atted.unsqueeze(1)


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

        self.mhatt1 = MHAtt(hidden_dim, dropout_r, head)
        self.mhatt2 = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        # self.dropout4 = nn.Dropout(dropout_r)
        # self.norm4 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, avg_pool=True):
        super(Transformer, self).__init__()
        self.avg_pool = avg_pool

        # self.que_proto = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # self.img_proto = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # self.que_pool = AttProto(hidden_dim, dropout_r, hidden_dim)
        # self.img_pool = GuideProto(hidden_dim, dropout_r, hidden_dim)

        self.que_pool = AttFlat(hidden_dim, dropout_r, hidden_dim)
        self.img_pool = AttFlat(hidden_dim, dropout_r, hidden_dim)

        self.enc_list = nn.ModuleList([Encoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.dec_list = nn.ModuleList([Decoder(hidden_dim, dropout_r, head) for _ in range(1)])

        # self.l_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim * 2)
        # self.i_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim * 2)

        if avg_pool:
            # self.img_avgpool = nn.AdaptiveAvgPool1d(1)
            # self.que_avgpool = nn.AdaptiveAvgPool1d(1)
            # self.que_flatten = AttProto(hidden_dim, dropout_r, hidden_dim)
            # self.img_flatten = GuideProto(hidden_dim, dropout_r, hidden_dim)
            self.que_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
            self.img_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
        

    def forward(self, img, que, img_mask, que_mask):
        b, n, c = img.shape
        proto_mask = torch.ones((b,1,1,1)).cuda() == 0
        # que_proto = repeat(self.que_proto, '() n d -> b n d', b = b)
        # img_proto = repeat(self.img_proto, '() n d -> b n d', b = b)

        # que_proto = que_proto + self.que_pool(que, que_mask)
        # img_proto = img_proto + self.img_pool(img, que, img_mask)

        que_proto = self.que_pool(que, que_mask).unsqueeze(1)
        img_proto = self.img_pool(img, img_mask).unsqueeze(1)

        que_mask = torch.cat((que_mask, proto_mask), dim=-1)
        img_mask = torch.cat((img_mask, proto_mask), dim=-1)

        que = torch.cat((que_proto, que), dim=1)
        img = torch.cat((img_proto, img), dim=1)

        for enc in self.enc_list:
            que = enc(que, que_mask)
        
        for dec in self.dec_list:
            img = dec(img, que, img_mask, que_mask)

        if self.avg_pool:
            img_proto = self.img_flatten(img[:, 1:, :], img_mask[:, :, :, 1:])
            que_proto = self.que_flatten(que[:, 1:, :], que_mask[:, :, :, 1:])

        return img_proto, que_proto

