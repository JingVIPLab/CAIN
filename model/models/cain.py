import math
import sys
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from model.dataloader.fsl_vqa import proc_ques
from model.networks.lstm import LSTM
from model.networks.idea_final_transformer import Transformer
from model.networks.fusion import Fusion

# context
import h5py


def get_padding(backbone="Res12", part_num=5):
    padding = {}
    if part_num == 5:
        if backbone == "Res12":
            # 范围为3*3
            size = 3
            padding[0] = (0, 2, 0, 2)
            padding[1] = (2, 0, 0, 2)
            padding[2] = (0, 2, 2, 0)
            padding[3] = (2, 0, 2, 0)
            padding[4] = (1, 1, 1, 1)
        elif backbone == "SwinT":
            # 范围为4*4
            size = 4
            padding[0] = (0, 3, 0, 3)
            padding[1] = (3, 0, 0, 3)
            padding[2] = (0, 3, 3, 0)
            padding[3] = (3, 0, 3, 0)
            padding[4] = (2, 1, 2, 1)
        else:  # Vits
            # 范围为7*7
            size = 7
            padding[0] = (0, 7, 0, 7)
            padding[1] = (7, 0, 0, 7)
            padding[2] = (0, 7, 7, 0)
            padding[3] = (7, 0, 7, 0)
            padding[4] = (4, 3, 4, 3)

    elif part_num == 9:
        if backbone == "Res12":
            # 范围为2*2
            size = 2
            padding[0] = (0, 3, 0, 3)
            padding[1] = (2, 1, 0, 3)
            padding[2] = (3, 0, 0, 3)
            padding[3] = (0, 3, 2, 1)
            padding[4] = (2, 1, 2, 1)
            padding[5] = (3, 0, 2, 1)
            padding[6] = (0, 3, 3, 0)
            padding[7] = (2, 1, 3, 0)
            padding[8] = (3, 0, 3, 0)

        elif backbone == "SwinT":
            # 范围为3*3
            size = 3
            padding[0] = (0, 4, 0, 4)
            padding[1] = (2, 2, 0, 4)
            padding[2] = (4, 0, 0, 4)
            padding[3] = (0, 4, 2, 2)
            padding[4] = (2, 2, 2, 2)
            padding[5] = (4, 0, 2, 2)
            padding[6] = (0, 4, 4, 0)
            padding[7] = (2, 2, 4, 0)
            padding[8] = (4, 0, 4, 0)

        else:  # Vits
            # 范围为5*5
            size = 5
            padding[0] = (0, 9, 0, 9)
            padding[1] = (5, 4, 0, 9)
            padding[2] = (9, 0, 0, 9)
            padding[3] = (0, 9, 5, 4)
            padding[4] = (5, 4, 5, 4)
            padding[5] = (9, 0, 5, 4)
            padding[6] = (0, 9, 9, 0)
            padding[7] = (5, 4, 9, 0)
            padding[8] = (9, 0, 9, 0)

    return size, padding


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class FewShotModel(nn.Module):
    def __init__(self, args, hidden_dim=768):
        super().__init__()
        self.args = args

        # resnet12 => 640
        if args.backbone_class == 'Res12':
            from model.networks.res12 import ResNet
            # self.encoder = ResNet()
            self.encoder = ResNet(avg_pool=False)
            hidden_dim = 640
            img_size = 25
        elif args.backbone_class == 'SwinT':
            from model.networks.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer(window_size=7, embed_dim=96, depths=[2, 2, 6, 2],
                                           num_heads=[3, 6, 12, 24], mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
            hidden_dim = 768
            img_size = 49
        elif args.backbone_class == 'VitS':
            from model.networks.vision_transformer import VisionTransformer
            self.encoder = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                             qkv_bias=True)
            hidden_dim = 384
            img_size = 196
        else:
            raise ValueError('')
        self.img_size = img_size

        self.que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim, avg_pool=False)
        self.ctx_que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim, avg_pool=False)
        self.transformer = Transformer(hidden_dim=hidden_dim, avg_pool=False)
        # self.multi_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # self.retrieval_augmented = RetrievalAugmented(k_captions=self.args.k_caption, hidden_dim=hidden_dim)

        # self.CA = idea11CA(hidden_dim)

        self.fusion = Fusion(hidden_dim=hidden_dim, img_size=img_size, avg_pool=True, k_caption=args.k_caption)

    def split_shot_query(self, data, que, ep_per_batch=1):
        args = self.args  # 加上旋转的torch.Size([80, 4, 3, 84, 84])
        img_shape = data.shape[1:]
        data = data.view(ep_per_batch, args.way, args.shot + args.query, *img_shape)
        x_shot, x_query = data.split([args.shot, args.query], dim=2)
        x_shot = x_shot.contiguous()
        x_query = x_query.contiguous().view(ep_per_batch, args.way * args.query, *img_shape)

        que_shape = que.shape[1:]
        que = que.view(ep_per_batch, args.way, args.shot + args.query, *que_shape)
        que_shot, que_query = que.split([args.shot, args.query], dim=2)
        que_shot = que_shot.contiguous()
        que_query = que_query.contiguous().view(ep_per_batch, args.way * args.query, *que_shape)
        return x_shot, x_query, que_shot, que_query

    def get_context(self, img_ids):
        """
        获取上下文
        """
        top_k = self.args.top_k
        contexts = []
        with h5py.File(self.args.ctx_file, "r") as f:
            for img_id in img_ids:
                texts_whole = f[f"{img_id}/whole/texts"][:top_k]
                texts_five = f[f"{img_id}/five/texts"][:, :top_k]
                texts_nine = f[f"{img_id}/nine/texts"][:, :top_k]
                texts_five = list(chain(*texts_five))
                texts_nine = list(chain(*texts_nine))

                texts = list(texts_whole) + list(texts_five) + list(texts_nine)
                texts = [s.decode() for s in texts]
                # texts = [s for s in texts]
                contexts.append(texts)
        # print(len(contexts), len(contexts[0]))  # 80 75
        return contexts

    def get_contexts_feature(self, que_tot, contexts, distinct=True, k=3, img_ids=None):
        texts_fea = []

        # 根据w*s+w*q，每一张图片去筛选上下文
        for que_tot_one_img, context_one_img, img_id in zip(que_tot, contexts, img_ids):
            # 去重
            if distinct:
                context_one_img = list(set(context_one_img))

            texts_fea_one_img = self.process_contexts(context_one_img)

            # 计算欧氏距离
            distances = torch.norm(texts_fea_one_img - que_tot_one_img, dim=(1, 2))

            # 找到最接近的张量的索引
            closest_indices = torch.argsort(distances)[:k].tolist()
            context_k_fea = torch.tensor(
                np.array([item.cpu().detach().numpy() for item in texts_fea_one_img[closest_indices]]))

            texts_fea.append(context_k_fea)

            # print(img_id.item(), end=":")
            # for i in closest_indices:
            #     print(context_one_img[i], end='\t')
            # print("\n")


        ctx = torch.tensor(np.array([item.cpu().detach().numpy() for item in texts_fea])).cuda()

        # 将每一条上下文从15*640平均成了640
        # ctx = torch.mean(ctx, dim=2)  # [80, 3, 640]

        return ctx

    def process_contexts(self, context_list):
        # 将ctx用lstm编码
        texts_fea_one_img = [proc_ques(context, self.args.token_to_ix, max_token=15) for context in context_list]
        texts_fea_one_img = torch.tensor(np.array(texts_fea_one_img)).cuda()
        return self.ctx_que_encoder(texts_fea_one_img)  # [num_contexts, max_token, feature_dim]

    def get_contexts_id(self, que_tot, contexts, distinct=True):
        five_ids = []
        nine_ids = []

        for que_tot_one_img, context_one_img in zip(que_tot, contexts):
            if distinct:
                # 去重
                five_context_one_img = list(set(context_one_img[5:30]))
                nine_context_one_img = list(set(context_one_img[30:75]))
            else:
                five_context_one_img = list(context_one_img[5:30])
                nine_context_one_img = list(context_one_img[30:75])

            # 处理前五个上下文
            texts_five_fea_one_img = self.process_contexts(five_context_one_img)
            five_ids.append(self.find_closest_index(texts_five_fea_one_img, que_tot_one_img, five_context_one_img))

            # 处理接下来的九个上下文
            texts_nine_fea_one_img = self.process_contexts(nine_context_one_img)
            nine_ids.append(self.find_closest_index(texts_nine_fea_one_img, que_tot_one_img, nine_context_one_img))

        return five_ids, nine_ids

    def find_closest_index(self, texts_fea_one_img, que_tot_one_img, context_list):
        distances = torch.norm(texts_fea_one_img - que_tot_one_img, dim=(1, 2))
        closest_index = torch.argsort(distances)[0].item()
        return context_list.index(context_list[closest_index])

    def local_mask(self, ids, part_num=5):
        size, padding = get_padding(self.args.backbone_class, part_num)

        masks = torch.cat([
            torch.nn.functional.pad(torch.zeros((size, size)).cuda(),
                                    padding[id % part_num], value=1).cuda().reshape(1, -1) for id in ids], dim=0)

        masks = torch.eq(masks, 1)
        masks = masks.unsqueeze(1).unsqueeze(2)  # [80, 1, 1, 25]
        return masks

    def forward(self, x, que, support_labels, img_ids, get_feature=False):
        if get_feature:
            # 我不知道为什么会有这个 get_feature 但就运行的顺序来看，这个get_feature这辈子都不可能是true呀
            return self.encoder(x)
        else:
            x_shot, x_query, que_shot, que_query = self.split_shot_query(x, que, self.args.batch)
            # x_shot (1, w, s, 768)
            # x_query (1 , w * q, 768)
            # que_shot (w, 15)
            # que_query (w*q, 15)

            shot_shape = x_shot.shape[:-3]  # (1, way, shot)
            query_shape = x_query.shape[:-3]  # (1, way*query)
            img_shape = x_shot.shape[-3:]  # (3, 224, 224)
            que_shape = que_shot.shape[-1:]  # 文本的特征大小：15

            x_shot = x_shot.view(-1, *img_shape)  # (way, 3, 224, 224)
            x_query = x_query.view(-1, *img_shape)  # (way * query, 3, 224, 224)
            if self.args.backbone_class in ['VitS', 'SwinT']:
                x_tot = self.encoder.forward(torch.cat([x_shot, x_query], dim=0), return_all_tokens=True)[:, 1:]
                # (30, 49, 768)
            else:
                x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
            img_mask = make_mask(x_tot)

            que_shot = que_shot.view(-1, *que_shape)
            que_query = que_query.view(-1, *que_shape)
            que_tot = torch.cat([que_shot, que_query], dim=0)  # (80, 15)
            que_mask = make_mask(que_tot.unsqueeze(2))
            que_tot = self.que_encoder(que_tot)  # (way*(shot + query)=30, 768)

            # ================ context ================
            # print(img_ids.shape)  # 80
            contexts = self.get_context(img_ids)
            ctx_fea = self.get_contexts_feature(que_tot, contexts, k=self.args.k_caption, distinct=True, img_ids=img_ids)  # [80, 3, 15, 640]

            five_ids, nine_ids = self.get_contexts_id(que_tot, contexts)
            five_masks = self.local_mask(five_ids, 5)
            nine_masks = self.local_mask(nine_ids, 9)

            # ori = torch.sum(x_tot, dim=1) + torch.sum(que_tot, dim=1)

            x_tot, que_tot, ctx_fea = self.transformer(x_tot, que_tot, img_mask, que_mask, five_masks, nine_masks, ctx_fea)

            # trans = torch.sum(x_tot, dim=1) + torch.sum(que_tot, dim=1)

            multi_tot = self.fusion(que_tot, x_tot, ctx_fea)

            feat_shape = multi_tot.shape[1:]  # [768]

            x_shot, x_query = multi_tot[:len(x_shot)], multi_tot[-len(x_query):]
            x_shot = x_shot.view(*shot_shape, *feat_shape)  # (1, way=5, shot=1, 768)
            x_query = x_query.view(*query_shape, *feat_shape)  # (1, query=5, 768)
            logits = self._forward(x_shot, x_query)
            return logits
            # return logits, ori, trans, multi_tot

    def _forward(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')


if __name__ == '__main__':
    pass
